Multimodal AI Pipeline for Lymphoid Lesions: RFH vs FL
Author: Lucas Lacerda de Souza
Date: 2025
Language: Python 3.10+ (PyTorch 2.1)
________________________________________
1. Project Overview
This project implements a multimodal artificial intelligence pipeline for classifying Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).
It integrates histopathological images, clinicopathologic data, and morphometric nuclear features using XGBoost (SHAP), convolutional neural networks (CNNs) + multilayer perceptron, and explainable AI methods (Grad-CAM).
________________________________________
2. Pipeline
<img width="1317" height="1026" alt="Captura de tela 2025-10-26 135618" src="https://github.com/user-attachments/assets/be172473-8ce7-414b-9641-d5c6a3d92885" />
_______________________________________
3. Model Architectures
•	XGBoost
•	U-Net++
•	AlexNet + Multilayer perceptron
•	VGG16 + Multilayer perceptron
•	ResNet18 + Multilayer perceptron
•	GradCam
________________________________________
4. Features Used
•	Morphometric features (nucleus-based)
•	Clinicopathologic features (age, sex, location)
________________________________________
5. Evaluation Metrics
•	XGBoost + SHAP – Classification (accuracy, area under the curve (AUC), F1-score, precision, recall and SHAP).
•	U-Net++ (Loss, Accuracy, Precision, Recall, IoU and Dice coefficient).
•	AlexNet (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).
•	VGG16 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).
•	ResNet18 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).
•	GradCam (morphometric)
________________________________________
6. Repository Structure
Classification-RFH-and-FL/
│
├── models/
│   ├── Multimodal AlexNet - Model Level/
│   ├── Multimodal AlexNet - Patient-Level/
│   ├── Multimodal ResNet18 – Model Level/
│   ├── Multimodal ResNet18 - Patient-Level/
│   ├── Multimodal VGG16 – Model Level/
│   ├── Multimodal VGG16 - Patient-Level/
│   ├── XGBoost Classification/
│
├── requirements.txt
├── README.md
└── LICENSE
________________________________________
7. Installation
git clone https://github.com/<your-username>/pathology-multimodal-pipeline.git
cd pathology-multimodal-pipeline
pip install -r requirements.txt
________________________________________
8. Citation
@article{delasouza2025,
  title={Deep Learning-Based Histopathologic Classification of Head and Neck Reactive Follicular Hyperplasia and Follicular Lymphoma},
  author={de Souza, Lucas Lacerda, Chen, Zhiyang […] Khurram, Syed Ali and Vargas, Pablo Agustin},
  journal={npj Precision Oncology},
  year={2025},
  publisher={Nature Publishing Group UK London}
}________________________________________
9. License
MIT License © 2025 Lucas Lacerda de Souza

