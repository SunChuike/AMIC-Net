#!/usr/bin/env python
# -*- coding:utf8 -*-

from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn, _RealValuedColumn

import global_var as gl
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.ops import metrics
from prada_interface.algorithm import Algorithm
from optimizer.adagrad import SearchAdagrad
from optimizer import optimizer_ops as myopt
from model_util.util import *
from model_util.attention import attention as atten_func
from model_util.attention import feedforward

from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError


import numpy as np
from model_util import odps_io as myodps

optimizer_dict = {
    "Adagrad": lambda opt_conf, global_step: SearchAdagrad(opt_conf).get_optimizer(global_step)
}


class ActiveNet(Algorithm):

    def init(self, context):
        self.context = context

        self.config = self.context.get_config()

        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']

        self.main_column_blocks = []

        if self.algo_config.get('main_columns') is not None:
            arr_blocks = self.algo_config.get('main_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.main_column_blocks.append(block)
        else:
            raise RuntimeError("main_columns must be specified.")


        self.seq_column_blocks = []
        self.seq_column_len = {}
        self.seq_column_atten = {}
        self.seq_column_value = {}

        if self.algo_config.get('seq_column_blocks') is not None:
            arr_blocks = self.algo_config.get('seq_column_blocks').split(';', -1)
            for block in arr_blocks:
                if block == "":
                    continue
                arr = block.split(':', -1)
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]
                if len(arr[2]) > 0:
                    self.seq_column_atten[arr[0] + '_atten_item'] = arr[2]
                if len(arr[3]) > 0:
                    self.seq_column_value[arr[0] + '_atten_value'] = self.model_conf["model_hyperparameter"].get(arr[3],
                                                                                                                 [])

        self.layer_dict = {}
        self.sequence_layer_dict = {}

        self.seq_split_num = self.model_conf['model_hyperparameter'].get('seq_split_num')
        self.logits_temperature = self.model_conf['model_hyperparameter'].get('logits_temperature')
        self.logits_lamda = self.model_conf['model_hyperparameter'].get('logits_lamda')
        self.seq_split_logits_list = []


    def variable_scope(self, *args, **kwargs):
        kwargs['partitioner'] = partitioned_variables.min_max_variable_partitioner(
            max_partitions=self.config.get_job_config("ps_num"),
            min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        kwargs['reuse'] = tf.AUTO_REUSE
        return tf.variable_scope(*args, **kwargs)

    def inference(self, features, feature_columns):
        # parse feature, including User/Item/Seq/Context
        self.feature_columns = feature_columns[self.model_name]
        self.features = features[self.model_name]

        # Embedding Layer
        self.embedding_layer(self.features, self.feature_columns)

        # Base Model
        self.sequence_layer()
        self.main_logits = self.main_net() # p

        # AMIC-Net
        ## IMSM
        self.split_sequence_layer()
        self.seq_split_logits_list = self.seq_split_main_net()
        ## AWM
        self.seq_split_merge_logits = self.seq_split_merge_net() # q
        ## CFM
        self.logits = self.main_logits + self.logits_lamda * self.seq_split_merge_logits # p + λ*q
        return self.logits

    def loss(self, logits, label):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            self.label = label
            self.logits = logits
            self.reg_loss_f()

            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.label)

            self.loss_op = tf.reduce_mean(loss) + self.reg_loss
            return self.loss_op

    def predictions(self, logits):
        with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
            self.prediction = tf.sigmoid(logits)
        return


    def optimizer(self, context, loss_op):
        '''
        return train_op
        '''
        with tf.variable_scope(
                name_or_scope="Optimize",
                partitioner=partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.config.get_job_config("ps_num"),
                    min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                ),
                reuse=tf.AUTO_REUSE):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = self.update_op(name=self.model_name)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                if 'scope' not in opt_conf or opt_conf["scope"] == "Global":
                    global_opt_name = opt_name
                    global_optimizer = optimizer
                else:
                    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opt_conf["scope"])
                    if len(vars) != 0:
                        for var in vars:
                            if var in global_opt_vars:
                                global_opt_vars.remove(var)
                        train_op, _, _ = myopt.optimize_loss(
                            loss=loss_op,
                            global_step=self.global_step,
                            learning_rate=opt_conf.get("learning_rate", 0.01),
                            optimizer=optimizer,
                            # update_ops=update_ops,
                            clip_gradients=opt_conf.get('clip_gradients', 5),
                            variables=vars,
                            increment_global_step=False,
                            summaries=myopt.OPTIMIZER_SUMMARIES)
                        train_ops.append(train_op)
            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    # update_ops=update_ops,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op

    def update_op(self, name):
        update_ops = []
        start = ('Share') if name is None else ('Share', name)
        for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            if update_op.name.startswith(start):
                update_ops.append(update_op)
        return update_ops

    def get_optimizer(self, opt_name, opt_conf, global_step):
        optimizer = None
        for name in optimizer_dict:
            if opt_name == name and isinstance(optimizer_dict[name], str):
                optimizer = optimizer_dict[name]
                break
            elif opt_name == name:
                optimizer = optimizer_dict[name](opt_conf, global_step)
                break

        return optimizer

    def build_graph(self, context, features, feature_columns, labels):
        self.set_global_step()
        self.inference(features, feature_columns)
        self.loss(self.logits, labels[self.model_name])
        self.optimizer(context, self.loss_op)
        self.predictions(self.logits)


    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)


    def embedding_layer(self, features, feature_columns):
        with tf.variable_scope(name_or_scope="Embedding_Layer",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for block_name in (self.main_column_blocks
                               + self.seq_column_len.values()):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)

                self.layer_dict[block_name] = layers.input_from_feature_columns(features,
                                                                                feature_columns=feature_columns[
                                                                                    block_name],
                                                                                scope=scope)

        self.sequence_layer_dict = self.build_sequence(self.seq_column_blocks, self.seq_column_len, "seq")

        with tf.variable_scope(name_or_scope="atten_input_from_feature_columns",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for atten_block_name in (self.seq_column_atten.values()):
                if len(atten_block_name) <= 0: continue
                if atten_block_name not in feature_columns or len(feature_columns[atten_block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for atten" % atten_block_name)
                self.layer_dict[atten_block_name] = layers.input_from_feature_columns(features,
                                                                                      feature_columns[atten_block_name],
                                                                                      scope=scope)

    def sequence_layer(self):
        for block_name in self.seq_column_blocks:
            attention_layer = [self.layer_dict[self.seq_column_atten[block_name + '_atten_item']]]
            attention = tf.concat(attention_layer, axis=1)
            target_atten = self.build_query_attention(
                self.sequence_layer_dict,
                self.seq_column_len,
                attention,
                None,
                block_name,
                self.model_conf['model_hyperparameter']['atten_param']['ma_num_units'],
                self.model_conf['model_hyperparameter']['atten_param']['ma_num_output_units'],
                self.model_conf['model_hyperparameter']['atten_param']['num_heads']
            )
            self.layer_dict[block_name] = target_atten

    def split_sequence_layer(self):
        for block_name in self.seq_column_blocks:
            split_item_vec_list, split_stt_vec_list = self.build_split_query_attention(
                self.sequence_layer_dict,
                self.seq_column_len,
                attention,
                None,
                block_name,
                self.model_conf['model_hyperparameter']['atten_param']['ma_num_units'],
                self.model_conf['model_hyperparameter']['atten_param']['ma_num_output_units'],
                self.model_conf['model_hyperparameter']['atten_param']['num_heads']
            )
            for i in range(self.seq_split_num):
                self.layer_dict[block_name + "_" + str(i)] = split_item_vec_list[i]


    def main_net(self):
        main_net_layer = []
        for block_name in (self.main_column_blocks + self.seq_column_blocks):
            if not self.layer_dict.has_key(block_name):
                raise ValueError('[Main net, layer dict] does not has block : {}'.format(block_name))
            main_net_layer.append(self.layer_dict[block_name])

        main_net = tf.concat(values=main_net_layer, axis=1)
        with self.variable_scope(name_or_scope="{}_Main_Score_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                for layer_id, num_hidden_units in enumerate(
                        self.model_conf['model_hyperparameter']['dnn_hidden_units']):
                    with self.variable_scope(
                            name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        main_net = layers.fully_connected(
                            main_net,
                            num_hidden_units,
                            getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.main_collections_dnn_hidden_layer],
                            outputs_collections=[self.main_collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                'batch_norm', True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})

                if self.model_conf['model_hyperparameter']['need_dropout']:
                    main_net = tf.layers.dropout(
                        main_net,
                        rate=self.model_conf['model_hyperparameter']['dropout_rate'],
                        noise_shape=None,
                        seed=None,
                        training=self.is_training,
                        name=None)


        # logit net
        with self.variable_scope(name_or_scope="{}_Logits".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                logits = layers.linear(
                    main_net,
                    1,
                    scope="main_net",
                    variables_collections=[self.main_collections_dnn_hidden_layer],
                    outputs_collections=[self.main_collections_dnn_hidden_output],
                    biases_initializer=None)
                bias = contrib_variables.model_variable(
                    'bias_weight',
                    shape=[1],
                    initializer=tf.zeros_initializer(),
                    trainable=True)
                logits = nn_ops.bias_add(logits, bias)

        return logits


    def seq_split_main_net(self):
        seq_split_logits_list = []
        for i in range(self.seq_split_num):
            seq_split_main_net_layer = []
            for block_name in (self.main_column_blocks):
                if not self.layer_dict.has_key(block_name):
                    raise ValueError('[seq split net, layer dict] does not has block : {}'.format(block_name))
                seq_split_main_net_layer.append(self.layer_dict[block_name])
            for block_name in (self.seq_column_blocks):
                if not self.layer_dict.has_key(block_name):
                    raise ValueError('[seq split net, layer dict] does not has block : {}'.format(block_name))
                seq_split_block_name = block_name + "_" + str(i)
                seq_split_main_net_layer.append(self.layer_dict[seq_split_block_name])

            seq_split_main_net = tf.concat(values=seq_split_main_net_layer, axis=1)

            with self.variable_scope(name_or_scope="{}_Seq_Split_Main_Score_{}".format(self.model_name, i),
                                     partitioner=partitioned_variables.min_max_variable_partitioner(
                                         max_partitions=self.config.get_job_config("ps_num"),
                                         min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                     ):
                with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                    for layer_id, num_hidden_units in enumerate(
                            self.model_conf['model_hyperparameter']['dnn_hidden_units']):
                        with self.variable_scope(
                                name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                            seq_split_main_net = layers.fully_connected(
                                seq_split_main_net,
                                num_hidden_units,
                                getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                                scope=dnn_hidden_layer_scope,
                                variables_collections=[self.main_collections_dnn_hidden_layer],
                                outputs_collections=[self.main_collections_dnn_hidden_output],
                                normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                    'batch_norm', True) else None,
                                normalizer_params={"scale": True, "is_training": self.is_training})

                    if self.model_conf['model_hyperparameter']['need_dropout']:
                        seq_split_main_net = tf.layers.dropout(
                            seq_split_main_net,
                            rate=self.model_conf['model_hyperparameter']['dropout_rate'],
                            noise_shape=None,
                            seed=None,
                            training=self.is_training,
                            name=None)
            # logit net
            with self.variable_scope(name_or_scope="{}_Seq_Split_Weight_Logits_{}".format(self.model_name, i),
                                     partitioner=partitioned_variables.min_max_variable_partitioner(
                                         max_partitions=self.config.get_job_config("ps_num"),
                                         min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                     ):
                with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                    logits = layers.linear(
                        seq_split_main_net,
                        1,
                        scope="main_net",
                        variables_collections=[self.main_collections_dnn_hidden_layer],
                        outputs_collections=[self.main_collections_dnn_hidden_output],
                        biases_initializer=None)
                    bias = contrib_variables.model_variable(
                        'bias_weight',
                        shape=[1],
                        initializer=tf.zeros_initializer(),
                        trainable=True)
                    logits = nn_ops.bias_add(logits, bias)

            seq_split_logits_list.append(logits)

        return seq_split_logits_list

    def seq_split_merge_net(self):
        self.seq_split_logits_layer = tf.concat(values=self.seq_split_logits_list, axis=-1)
        self.seq_split_weight_layer = tf.nn.softmax(self.seq_split_logits_layer / self.logits_temperature, axis = -1)
        seq_split_merge_logits = tf.reshape(tf.reduce_sum(tf.multiply(self.seq_split_logits_layer, self.seq_split_weight_layer), axis=-1),[-1, 1])
        return seq_split_merge_logits

    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_tmp = []
        for reg_loss in reg_losses:
            if reg_loss.name.startswith(self.model_name) or reg_loss.name.startswith('Share'):
                reg_tmp.append(reg_loss)
        self.reg_loss = tf.reduce_sum(reg_tmp)
        self.logger.info('regularization_variable: {}'.format(reg_tmp))


    def run_train(self, context, mon_session, task_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step, self.loss_op, self.metrics, self.label, self.localvar]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    global_step, loss, metrics, labels, flocalv = mon_session.run(
                        run_ops, feed_dict=feed_dict)
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.train_ops)
                    global_step, loss, metrics, labels, flocalv, _ = mon_session.run(
                        run_ops, feed_dict=feed_dict)

            except (ResourceExhaustedError, OutOfRangeError) as e:
                break  # release all
            except ConnectionError as e:
                pass
            except Exception as e:
                pass

    def build_query_attention(self, sequence_layer_dict, seq_column_len, attention_layer, attention_dict, block_name,
                              ma_num_units=64, ma_num_output_units=64, num_heads=1):
        if sequence_layer_dict is None or block_name not in sequence_layer_dict.keys():
            return None

        with arg_scope(
                model_arg_scope(
                    weight_decay=self.model_conf['model_hyperparameter']['atten_param'].get('attention_l2_reg',
                                                                                            0.0))):
            with tf.variable_scope(name_or_scope='Share_Sequence_Layer_{}'.format(block_name),
                                   partitioner=partitioned_variables.min_max_variable_partitioner(
                                       max_partitions=self.config.get_job_config('ps_num'),
                                       min_slice_size=self.config.get_job_config('dnn_min_slice_size')),
                                   reuse=tf.AUTO_REUSE) as (scope):
                max_len = 200
                sequence = sequence_layer_dict[block_name]
                sequence_v = sequence
                sequence_length = self.layer_dict[seq_column_len[block_name]]
                sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                attention = tf.expand_dims(attention_dict[block_name], 1)

                item_vec, stt_vec = atten_func(queries=attention,
                                               keys=sequence,
                                               values=sequence_v,
                                               key_masks=sequence_mask,
                                               query_masks=tf.sequence_mask(
                                                   tf.ones_like(attention[:, 0, 0], dtype=tf.int32), 1),
                                               num_units=ma_num_units,
                                               num_output_units=ma_num_output_units,
                                               scope=block_name + "_query_attention",
                                               atten_mode=self.model_conf['model_hyperparameter']['atten_param'][
                                                   'atten_mode'],
                                               reuse=tf.AUTO_REUSE,
                                               variables_collections=None,
                                               outputs_collections=None,
                                               num_heads=num_heads,
                                               residual_connection=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('residual_connection', False),
                                               attention_normalize=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('attention_normalize', False),
                                               use_atten_linear_project=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('use_atten_linear_project', True))
                if self.model_conf['model_hyperparameter']['atten_param'].get('residual_connection', False):
                    ma_num_output_units = attention.get_shape().as_list()[-1]
                else:
                    ma_num_output_units = ma_num_output_units
                dec = tf.reshape(item_vec, [-1, ma_num_output_units])
                dec = tf.concat(dec, axis=1)

        return dec

    def build_split_query_attention(self, sequence_layer_dict, seq_column_len, attention_layer, attention_dict, block_name,
                              ma_num_units=64, ma_num_output_units=64, num_heads=1):
        if sequence_layer_dict is None or block_name not in sequence_layer_dict.keys():
            return None

        with arg_scope(
                model_arg_scope(
                    weight_decay=self.model_conf['model_hyperparameter']['atten_param'].get('attention_l2_reg',
                                                                                            0.0))):
            with tf.variable_scope(name_or_scope='Share_Split_Sequence_Layer_{}'.format(block_name),
                                   partitioner=partitioned_variables.min_max_variable_partitioner(
                                       max_partitions=self.config.get_job_config('ps_num'),
                                       min_slice_size=self.config.get_job_config('dnn_min_slice_size')),
                                   reuse=tf.AUTO_REUSE) as (scope):
                max_len = 200
                sequence = sequence_layer_dict[block_name]
                sequence_v = sequence
                sequence_length = self.layer_dict[seq_column_len[block_name]]
                sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)
                attention = tf.expand_dims(attention_dict[block_name], 1)

                item_vec, stt_vec = atten_func(queries=attention,
                                               keys=sequence,
                                               values=sequence_v,
                                               key_masks=sequence_mask,
                                               query_masks=tf.sequence_mask(
                                                   tf.ones_like(attention[:, 0, 0], dtype=tf.int32), 1),
                                               num_units=ma_num_units,
                                               num_output_units=ma_num_output_units,
                                               scope=block_name + "_query_attention",
                                               atten_mode=self.model_conf['model_hyperparameter']['atten_param'][
                                                   'atten_mode'],
                                               reuse=tf.AUTO_REUSE,
                                               variables_collections=None,
                                               outputs_collections=None,
                                               num_heads=num_heads,
                                               residual_connection=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('residual_connection', False),
                                               attention_normalize=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('attention_normalize', False),
                                               use_atten_linear_project=self.model_conf['model_hyperparameter'][
                                                   'atten_param'].get('use_atten_linear_project', True))

                # sort seq items based on attention weight scores
                _, sorted_indices = tf.nn.top_k(stt_vec, k=200)
                sorted_seq = tf.batch_gather(sequence, sorted_indices)
                # segment seq partitioned by
                split_seq_list = tf.split(sorted_seq, num_or_size_splits=self.seq_split_num, axis=1)
                sorted_sequence_mask = tf.batch_gather(sequence_mask, sorted_indices)
                split_sequence_mask_list = tf.split(sorted_sequence_mask, num_or_size_splits=self.seq_split_num, axis=1)

            split_item_vec_list = []
            split_stt_vec_list = []
            for i in range(self.seq_split_num):
                split_seq = split_seq_list[i]
                split_sequence_mask = split_sequence_mask_list[i]
                with tf.variable_scope(name_or_scope='Share_Split_Sequence_Layer_{}_{}'.format(block_name, i),
                                       partitioner=partitioned_variables.min_max_variable_partitioner(
                                           max_partitions=self.config.get_job_config('ps_num'),
                                           min_slice_size=self.config.get_job_config('dnn_min_slice_size')),
                                       reuse=tf.AUTO_REUSE) as (scope):
                    split_item_vec, split_stt_vec = atten_func(queries=attention,
                                                   keys=split_seq,
                                                   values=split_seq,
                                                   key_masks=split_sequence_mask,
                                                   query_masks=tf.sequence_mask(
                                                       tf.ones_like(attention[:, 0, 0], dtype=tf.int32), 1),
                                                   num_units=ma_num_units,
                                                   num_output_units=ma_num_output_units,
                                                   scope=block_name + "_query_attention" + "_" + str(i),
                                                   atten_mode=self.model_conf['model_hyperparameter']['atten_param'][
                                                       'atten_mode'],
                                                   reuse=tf.AUTO_REUSE,
                                                   variables_collections=None,
                                                   outputs_collections=None,
                                                   num_heads=num_heads,
                                                   residual_connection=self.model_conf['model_hyperparameter'][
                                                       'atten_param'].get('residual_connection', False),
                                                   attention_normalize=self.model_conf['model_hyperparameter'][
                                                       'atten_param'].get('attention_normalize', False),
                                                   use_atten_linear_project=self.model_conf['model_hyperparameter'][
                                                       'atten_param'].get('use_atten_linear_project', True))
                    dec = tf.reshape(item_vec, [-1, ma_num_output_units])
                    dec = tf.concat(dec, axis=1)
                    split_item_vec_list.append(dec)
                    split_stt_vec_list.append(split_stt_vec)

        return split_item_vec_list, split_stt_vec_list


    def build_sequence(self, seq_column_blocks, seq_column_len, name):
        features = self.features
        feature_columns = self.feature_columns
        sequence_layer_dict = {}
        if seq_column_blocks is None or len(seq_column_blocks) == 0:
            return
        with tf.variable_scope(name_or_scope='{}_seq_input_from_feature_columns'.format(name),
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config('ps_num'),
                                   min_slice_size=self.config.get_job_config('embedding_min_slice_size')),
                               reuse=tf.AUTO_REUSE) as (scope):
            if len(seq_column_blocks) > 0:
                for block_name in seq_column_blocks:
                    if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                        raise ValueError('block_name:(%s) not in feature_columns for seq' % block_name)
                    seq_len = 200
                    sequence_stack = _input_from_feature_columns(features,
                                                                     feature_columns[block_name],
                                                                     weight_collections=None,
                                                                     trainable=True,
                                                                     scope=scope,
                                                                     output_rank=3,
                                                                     default_name='sequence_input_from_feature_columns')
                    sequence_stack = tf.reshape(sequence_stack, [-1, seq_len, sequence_stack.get_shape()[(-1)].value])
                    sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                    if block_name in seq_column_len and seq_column_len[block_name] in self.layer_dict:
                        sequence_length = self.layer_dict[seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                        sequence_stack = tf.reshape(
                            tf.where(tf.reshape(sequence_mask, [-1]), sequence_2d, tf.zeros_like(sequence_2d)),
                            tf.shape(sequence_stack))
                    else:
                        sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                    sequence_layer_dict[block_name] = sequence_stack
        return sequence_layer_dict
