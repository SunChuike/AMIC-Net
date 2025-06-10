# Offical repo for "Counterfactual Reasoning for Weak Interest Overfitting in Sequential Recommendation"

## Release Notes (Current Version)
The implementation of AMIC-Net is based on our company's customized distributed TensorFlow framework, designed to optimize industrial applications. Due to company policy, this repository provides a carefully extracted and simplified version of the source code. While it is not runnable out-of-the-box, it is intended as supporting material to clearly illustrate the implementation logic of the model architecture and key modules, thereby enhancing the transparency of our method's design.

We aim to provide an out-of-the-box AMIC-Net version for the community. A forthcoming update will feature an implementation leveraging the publicly available [UserBehavior Dataset](https://tianchi.aliyun.com/dataset/649).  

## Model Architecture & Key Components
The AMIC-Net architecture, highlighting critical modules, is shown below. Accompanying code snippets illustrate their implementation logic.

*   **Overall Model Structure:**
<img src="https://github.com/user-attachments/assets/ee9a961e-f3bd-4a51-8cd3-83d72abcf13a" width="600">

*   **Annotated Code Snippet for Key Modules (model.py):**
<img src="https://github.com/user-attachments/assets/275d21ce-ef2a-4428-87ad-71f23c66bf10" width="600">

*   **The detailed derivations for Formulas 8 to 9 are provided below:**
<img src="https://github.com/user-attachments/assets/e1980d8e-62c6-4306-96ff-ff3b7efedc28" width="600">

