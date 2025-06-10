# Offical repo for "Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation"

## Release Notes (Current Version)
The implementation of AMIC-Net is based on our company's customized distributed TensorFlow framework, designed to optimize industrial applications. Due to company policy, this repository provides a carefully extracted and simplified version of the source code. While it is not runnable out-of-the-box, it is intended as supporting material to clearly illustrate the implementation logic of the model architecture and key modules, thereby enhancing the transparency of our method's design.

We aim to provide an out-of-the-box AMIC-Net version for the community. A forthcoming update will feature an implementation leveraging the publicly available [UserBehavior Dataset](https://tianchi.aliyun.com/dataset/649).  

## Model Architecture & Key Components
The AMIC-Net architecture, highlighting critical modules, is shown below. Accompanying code snippets illustrate their implementation logic.

*   **Overall Model Structure:**
![model](https://github.com/user-attachments/assets/ee9a961e-f3bd-4a51-8cd3-83d72abcf13a)

*   **Annotated Code Snippet for Key Modules (model.py):**
![image](https://github.com/user-attachments/assets/275d21ce-ef2a-4428-87ad-71f23c66bf10)

The detailed derivations for Formulas 8 and 9 are provided below:
Given a sequence containing user strong and weak interests: {X = x, Z = z}, {X = x, Z = z_mask} and {X = x_mask, Z = z} form a partition, which are mutually exclusive and collectively exhaustive. According to the equation of total probability, we have:

P(Y=1|X=x_mask, Z=z) * P(X=x_mask, Z=z) = P(Y=1|X=x, Z=z) * P(X=x, Z=z) – P(Y=1|X=x, Z=z_mask) * P(X=x, Z=z_mask)     (12)

By dividing both sides by P(X=x_mask, Z=z), we obtain:
P(Y=1|X=x_mask, Z=z)
= P(X=x, Z=z)P(X=x, Z=z) / P(X=x_mask, Z=z) * P(Y=1|X=x, Z=z) – P(X=x, Z=z_mask) / P(X=x_mask, Z=z) * P(Y=1|X=x, Z=z_mask)
= α * P(Y=1|X=x, Z=z) – β * P(Y=1|X=x, Z=z_mask)    (13)

where α = P(X=x, Z=z) / P(X=x_mask, Z=z) and β = P(X=x, Z=z_mask) / P(X=x_mask, Z=z) are data-dependent parameters.

Substituting the result of equation Eq.(13) into Eq.(8), we have:
y^ = P(Y=1|X=x, Z=z) – P(Y=1|X=x_mask, Z=z) = (1 - α) * P(Y=1|X=x, Z=z) + β * P(Y=1|X=x, Z=z_mask)

By dividing both sides by (1 - α) for each example, we obtain y^ = P(Y=1|X=x, Z=z) + λ * P(Y=1|X=x, Z=z_mask), where λ = β / (1 - α). We neglect the denominator (1 − α) for y^ since it does not affect the final result.
