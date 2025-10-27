**Deep Learning-Based Histopathologic Classification of Head and Neck Reactive Follicular Hyperplasia and Follicular Lymphoma** 

Author: Lucas Lacerda de Souza

Date: 2025

Language: Python 3.10+ (PyTorch 2.1)
________________________________________
**1. Project Overview**

This project implements a multimodal artificial intelligence pipeline for classifying Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).
It integrates histopathological images, clinicopathologic data, and morphometric nuclear features using XGBoost (SHAP), convolutional neural networks (CNNs) + multilayer perceptron, and explainable AI methods (Grad-CAM).
________________________________________
**2. Pipeline**
 <img width="1317" height="1026" alt="Captura de tela 2025-10-26 135618" src="https://github.com/user-attachments/assets/4d637290-0570-41d6-b5bb-0ec6b5c1a36f" />

________________________________________
**3. Model Architectures**

•	XGBoost

•	U-Net++

•	AlexNet + Multilayer perceptron

•	VGG16 + Multilayer perceptron

•	ResNet18 + Multilayer perceptron

•	GradCam
________________________________________
**4. Features Used**
   
•	Morphometric features (nucleus-based)

•	Clinicopathologic features (age, sex, location)
________________________________________
**5. Evaluation Metrics**
   
•	XGBoost + SHAP – Classification (accuracy, area under the curve (AUC), F1-score, precision, recall and SHAP).

•	U-Net++ (Loss, Accuracy, Precision, Recall, IoU and Dice coefficient).

•	AlexNet (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	VGG16 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	ResNet18 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	GradCam (morphometric)

________________________________________
**6. Repository Structure**
   
Classification-RFH-and-FL/

│

├── 📄 LICENSE

├── 📄 README.md

├── 📄 REQUIREMENTS.txt

│

├── 📁 data/

│   ├── 📄 Supplementary Table 1.xlsx   # Morphometric analysis (traditional statistics)

│   ├── 📄 Supplementary Table 2.xlsx   # Clinicopathologic + morphometric analysis (XGBoost)

│   ├── 📄 Supplementary Table 3.xlsx   # Multimodal (clinical + morphometric factors)

│   ├── 📄 Supplementary Table 4.xlsx   # Grad-CAM data (external validation)

│   └── 📄 README.md                    # Description of supplementary tables and data structure

│

├── 📁 models/

│   ├── 📄 Multimodal AlexNet - Model Level.py

│   ├── 📄 Multimodal AlexNet - Patient-Level.py

│   ├── 📄 Multimodal ResNet18 - Model Level.py

│   ├── 📄 Multimodal ResNet18 - Patient-Level.py

│   ├── 📄 Multimodal VGG16 - Model Level.py

│   ├── 📄 Multimodal VGG16 - Patient-Level.py

│   ├── 📄 XGBoost Classification.py

│   └── 📄 README.md                  

________________________________________
**7. Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL.git
cd Classification-RFH-and-FL

________________________________________
**8. Citation**

@article{delasouza2025,
  title={Deep Learning-Based Histopathologic Classification of Head and Neck Reactive Follicular Hyperplasia and Follicular Lymphoma},
  author={de Souza, Lucas Lacerda, Chen, Zhiyang […] Khurram, Syed Ali and Vargas, Pablo Agustin},
  journal={npj digital medicine},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
________________________________________
**9. License**

MIT License © 2025 Lucas Lacerda de Souza

