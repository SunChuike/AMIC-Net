# Offical repo for "Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation via Interest Segmentation"

**Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation via Interest Segmentation (DASFAA 2026)**  
*Chuike Sun, Yuhao Chen, Xing Fang, Yang Huang, Songyin Luo, Ruocong Tang, Jing Wang, Junzhou Chen*  
[DOI Link](https://doi.org/10.1007/978-981-92-0378-9_40)  

## **Abstract**  
In e-commerce recommendation, user behavior sequences often contain strong, weak, and irrelevant interests concerning the target item. Existing sequential models tend to overfit weak interests, leading to biased predictions. To address this overfitting challenge, we propose the Adaptive Multi-Interest Counterfactual Network (AMIC-Net). It is designed to amplify strong interest contributions while mitigating weak interest overfitting and noise interference. Specifically, AMIC-Net employs: 1) An Implicit Multi-Interest Segmentation Module (IMSM) that implicitly segments user interests by sorting the item sequence based on continuous, attention-based relevance scores, before partitioning and evaluating the resulting segments. 2) An Adaptive Weighting Module (AWM) that uses the evaluation score of each segment as a dynamic weight, amplifying strong interest signals while attenuating weak and irrelevant ones to generate a core-intent-focused user representation. 3) A Counterfactual Fusion Module (CFM) applying counterfactual reasoning to integrate the base model’s predictions with AWM’s weighted interest representation. This fusion specifically enhances the direct predictive effect of strong interests, thereby reducing the overfitting risk associated with weak interests. Notably, our plug-and-play method easily integrates with existing models, and online A/B tests confirm that AMIC-Net improves recommendation quality in industrial applications. The implementation is available at https://github.com/SunChuike/AMIC-Net.

## Release Notes
The implementation of AMIC-Net is based on our company's customized distributed TensorFlow framework, designed to optimize industrial applications. Due to company policy, this repository provides a carefully extracted and simplified version of the source code. While it is not runnable out-of-the-box, it is intended as supporting material to clearly illustrate the implementation logic of the model architecture and key modules, thereby enhancing the transparency of our method's design.

## Model Architecture & Key Components
The AMIC-Net architecture, highlighting critical modules, is shown below. Accompanying code snippets illustrate their implementation logic.

*   **Overall Model Structure:**

<img width="2566" height="892" alt="model" src="https://github.com/user-attachments/assets/bbeb9f33-3847-4a72-b6ac-63c3facf3b00" />

*   **Annotated Code Snippet for Key Modules (model.py):**
<img src="https://github.com/user-attachments/assets/275d21ce-ef2a-4428-87ad-71f23c66bf10" width="600">

## **Citation**  
If you use AMIC-Net, please cite:  

```bibtex
@InProceedings{10.1007/978-981-92-0378-9_40,
author="Sun, Chuike
and Chen, Yuhao
and Fang, Xing
and Huang, Yang
and Luo, Songyin
and Tang, Ruocong
and Wang, Jing
and Chen, Junzhou",
editor="Jung, Hyungsoo
and Wang, Tianzheng
and Toyoda, Masashi
and Kwon, Hyuk-Yoon
and Lee, Jae-woong",
title="Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation via Interest Segmentation",
booktitle="Database Systems for Advanced Applications",
year="2026",
publisher="Springer Nature Singapore",
address="Singapore",
pages="619--631",
abstract="In e-commerce recommendation, user behavior sequences often contain strong, weak, and irrelevant interests concerning the target item. Existing sequential models tend to overfit weak interests, leading to biased predictions. To address this overfitting challenge, we propose the Adaptive Multi-Interest Counterfactual Network (AMIC-Net). It is designed to amplify strong interest contributions while mitigating weak interest overfitting and noise interference. Specifically, AMIC-Net employs: 1) An Implicit Multi-Interest Segmentation Module (IMSM) that implicitly segments user interests by sorting the item sequence based on continuous, attention-based relevance scores, before partitioning and evaluating the resulting segments. 2) An Adaptive Weighting Module (AWM) that uses the evaluation score of each segment as a dynamic weight, amplifying strong interest signals while attenuating weak and irrelevant ones to generate a core-intent-focused user representation. 3) A Counterfactual Fusion Module (CFM) applying counterfactual reasoning to integrate the base model's predictions with AWM's weighted interest representation. This fusion specifically enhances the direct predictive effect of strong interests, thereby reducing the overfitting risk associated with weak interests. Notably, our plug-and-play method easily integrates with existing models, and online A/B tests confirm that AMIC-Net improves recommendation quality in industrial applications. The implementation is available at https://github.com/SunChuike/AMIC-Net.",
isbn="978-981-92-0378-9"
}
```
## **Contact**  
📧 Email: **sunchuike.sck@taobao.com**  
