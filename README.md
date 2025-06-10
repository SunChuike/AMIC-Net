# AMIC-Net: Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation

This repository contains the official implementation of our proposed method from the paper:
**"Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation via Interest Segmentation"**

## Important Note on Current Release
The full implementation of AMIC-Net is deeply integrated with our proprietary, industrial-scale distributed TensorFlow framework and relies on confidential internal datasets for optimal performance in real-world applications. **Due to these constraints, we are unable to release the complete, directly runnable code.**

**This repository provides a carefully extracted and simplified version of the source code.** While it is not runnable out-of-the-box, it serves as supplementary material to clearly illustrate the core algorithms, the implementation details of our proposed model architecture, and the logic behind key modules. Our goal is to offer transparency into our method's design.

We are committed to providing a fully runnable and reproducible version of AMIC-Net for the community. A future update will include a self-contained implementation based on a publicly available dataset, specifically:

UserBehavior Dataset: https://tianchi.aliyun.com/dataset/649
This future release will allow for local execution, experimentation, and replication of key results, addressing the current limitations. Please stay tuned for updates!

## 🛠️ Model Architecture & Key Components
AMIC-Net  overall architecture is depicted below, with critical modules highlighted. The released code snippets correspond to these highlighted parts, demonstrating their implementation logic.

*   **Overall Model Structure:**
![model](https://github.com/user-attachments/assets/ee9a961e-f3bd-4a51-8cd3-83d72abcf13a)

*   **Interest Segmentation Module Detail:**
![image](https://github.com/user-attachments/assets/275d21ce-ef2a-4428-87ad-71f23c66bf10)

## 🔢 Key Derivations for Counterfactual Prediction
A fundamental aspect of AMIC-Net is its counterfactual reasoning framework. Below are the key derivations for formulas 8 to 9 (as referenced in our paper), which lead to our unique counterfactual prediction:

Given a sequence containing the target item and its interests {X = x,Z = z}, {X = x, Z = z_mask } and {X =x_mask, Z =z}form a partition, which are mutually exclusive and collectively exhaustive. According to the equation of total probability, we have:
(12) P (Y=1|X=x_mask, Z=z) P (X=x_mask, Z=z) = 
P (Y=1|X=x, Z=z) P (X=x, Z=z) – P (Y=1|X=x, Z=z_mask) P (X=x, Z=z_mask)
By dividing both sides by P (X = x_mask, Z = z), we obtain:
(13) P (Y=1|X=x_mask, Z=z) = 
P (X=x, Z=z) P (X=x, Z=z)/ P (X=x_mask, Z=z) P (Y=1|X=x, Z=z) – 
P (X=x, Z=z_mask) / P (X=x_mask, Z=z) P (Y=1|X=x, Z=z_mask)
=α P (Y=1|X=x, Z=z) – β P (Y=1|X=x, Z=z_mask)
where α = P (X=x, Z=z) / P (X=x_mask, Z=z) and β = P (X=x, Z=z_mask) / P (X=x_mask, Z=z) are data-dependent parameters.
Substituting the result of equation Eq.(13) into Eq.(8), we have:
(14) y_^ = P (Y=1|X=x, Z=z) – P (Y=1|X=x_mask, Z=z) 
= (1 - α) P (Y=1|X=x, Z=z) + β P (Y=1|X=x, Z=z_mask)
By dividing both sides by (1 - α) for each example, we obtain y_^ = P(Y=1|X=x, Z=z) + λ P(Y=1|X=x, Z=z_mask), where λ = β / (1 - α). We neglect the denominator (1−α) for y_^ since it does not affect the final result.
