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
   
## 📂 Repository Structure

README.md — Documentation and usage instructions
REQUIREMENTS.txt — Dependencies
LICENSE — Project license

models/
 ├── multimodal_alexnet_patch_level.py
 ├── multimodal_alexnet_patient_level.py
 ├── multimodal_resnet18_patch_level.py
 ├── multimodal_resnet18_patient_level.py
 ├── multimodal_vgg16_patch_level.py
 ├── multimodal_vgg16_patient_level.py
 ├── segmentation_unet++.py
 ├── xgboost_classification_cpc_mpa.py
 └── xgboost_classification_gradcam.py

data/
 ├── train/
 │ ├── 0/ → Reactive Follicular Hyperplasia
 │ └── 1/ → Follicular Lymphoma
 ├── val/
 │ ├── 0/
 │ └── 1/
 ├── test/
 │ ├── 0/
 │ └── 1/
 └── clinical_data/
  ├── clinical_data_train.xlsx
  ├── clinical_data_val.xlsx
  └── clinical_data_test.xlsx

patches/
 ├── gradcam/
 │ ├── heatmaps/
 │ │ └── heatmap.png files
 │ └── patches/
 │  └── patch.png files
 ├── masks/
 │ ├── train/
 │ ├── val/
 │ └── test/
 │  └── mask.png files
 └── patches/
  ├── train/
  ├── val/
  └── test/
   └── patch.png files

supplementary_data/
 ├── supplementary_table_1.xlsx → Morphometric analysis (XGBoost)
 ├── supplementary_table_2.xlsx → Clinicopathologic + morphometric data
 ├── supplementary_table_3.xlsx → Multimodal analysis (clinical + morphometric)
 ├── supplementary_table_4.xlsx → Grad-CAM results (external validation)
 └── supplementary_table_5.xlsx → Pathologists’ evaluation
________________________________________

**7. Run models and reproduce tables**


<img width="1632" height="1041" alt="image" src="https://github.com/user-attachments/assets/c60aaf17-10f5-4f2c-b9fa-2fc5eb279eca" />


________________________________________

**8. Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL.git
cd Classification-RFH-and-FL

________________________________________
**9. Citation**

@article{delasouza2025,
  title={Deep Learning-Based Histopathologic Classification of Head and Neck Reactive Follicular Hyperplasia and Follicular Lymphoma},
  author={de Souza, Lucas Lacerda, Chen, Zhiyang […] Khurram, Syed Ali and Vargas, Pablo Agustin},
  journal={npj digital medicine},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
________________________________________
**10. License**

MIT License © 2025 Lucas Lacerda de Souza

