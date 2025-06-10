# AMIC-Net: Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation

This repository contains the official implementation of our proposed method from the paper:
**"Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation via Interest Segmentation"**

## Important Note on Current Release
The full implementation of AMIC-Net is deeply integrated with our proprietary, industrial-scale distributed TensorFlow framework and relies on confidential internal datasets for optimal performance in real-world applications. Due to these constraints, this repository provides a carefully extracted and simplified version of the source code. While it is not runnable out-of-the-box, it serves as supplementary material to clearly illustrate the core algorithms, the implementation details of our proposed model architecture, and the logic behind key modules. Our goal is to offer transparency into our method's design.

We are committed to providing a fully runnable and reproducible version of AMIC-Net for the community. A future update will include a self-contained implementation based on the publicly available "UserBehavior Dataset: https://tianchi.aliyun.com/dataset/649". Please stay tuned for updates!

## Model Architecture & Key Components
AMIC-Net  overall architecture is depicted below, with critical modules highlighted. The released code snippets correspond to these highlighted parts, demonstrating their implementation logic.

*   **Overall Model Structure:**
![model](https://github.com/user-attachments/assets/ee9a961e-f3bd-4a51-8cd3-83d72abcf13a)

*   **Annotated Code Snippet for Key Modules (model.py):**
![image](https://github.com/user-attachments/assets/275d21ce-ef2a-4428-87ad-71f23c66bf10)

## Key Derivations for Counterfactual Causal Fusion

AMIC-Net's counterfactual prediction mechanism is detailed below through the derivations of Formulas 8 to 9 (as presented in our paper):

Given a sequence containing the target item and its interests {X = x,Z = z}, {X = x, Z = z_mask } and {X =x_mask, Z =z}form a partition, which are mutually exclusive and collectively exhaustive. According to the equation of total probability, we have:

P(Y=1|X=x_mask, Z=z) * P(X=x_mask, Z=z) = P(Y=1|X=x, Z=z) * P(X=x, Z=z) – P(Y=1|X=x, Z=z_mask) * P(X=x, Z=z_mask)     (12)

By dividing both sides by P(X=x_mask, Z=z), we obtain:

P(Y=1|X=x_mask, Z=z) = P(X=x, Z=z)P(X=x, Z=z) / P(X=x_mask, Z=z) * P(Y=1|X=x, Z=z) – P(X=x, Z=z_mask) / P(X=x_mask, Z=z) * P(Y=1|X=x, Z=z_mask) 
                     = αP(Y=1|X=x, Z=z) – βP(Y=1|X=x, Z=z_mask)    (13)

where α = P(X=x, Z=z) / P(X=x_mask, Z=z) and β = P(X=x, Z=z_mask) / P(X=x_mask, Z=z) are data-dependent parameters.

Substituting the result of equation Eq.(13) into Eq.(8), we have:

y^ = P(Y=1|X=x, Z=z) – P(Y=1|X=x_mask, Z=z) = (1 - α) * P(Y=1|X=x, Z=z) + β * P(Y=1|X=x, Z=z_mask)

By dividing both sides by (1 - α) for each example, we obtain y^ = P(Y=1|X=x, Z=z) + λ * P(Y=1|X=x, Z=z_mask), where λ = β / (1 - α). We neglect the denominator (1−α) for y_^ since it does not affect the final result.
