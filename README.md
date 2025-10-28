**Deep Learning-Based Histopathologic Classification of Head and Neck Reactive Follicular Hyperplasia and Follicular Lymphoma** 

Author: Lucas Lacerda de Souza

Date: 2025
________________________________________
**1. Project Overview**

This project implements a multimodal artificial intelligence pipeline for classifying Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).
It integrates histopathological images, clinicopathologic data, and morphometric nuclear features using XGBoost (SHAP), convolutional neural networks (CNNs) + multilayer perceptron, and explainable AI methods (Grad-CAM).
________________________________________
**2. Pipeline**
 <img width="1317" height="1026" alt="Captura de tela 2025-10-26 135618" src="https://github.com/user-attachments/assets/4d637290-0570-41d6-b5bb-0ec6b5c1a36f" />

________________________________________
**3. Environment and Hardware**

All experiments were performed using the following configuration:
**Operating System:** Ubuntu 20.04.1 LTS
**Python Version:** 3.12.11
**PyTorch Version:** 2.8.0 (CUDA 12.8)
**CPU:** Intel Xeon W-2295 (18 cores / 36 threads)
**RAM:** 125 GB
**GPUs:** 3 × NVIDIA GeForce RTX 3090 (24 GB each)
________________________________________
**4. Environment Files**

**Channels:**

  • pytorch
  
  • nvidia
  
  • defaults
  
**Dependencies:**
  • python=3.12.11
  
  • pytorch=2.8.0
  
  • torchvision=0.19.0
  
  • torchaudio=2.8.0
  
  • cudatoolkit=12.8
  
  • numpy=1.26.4
  
  • pandas=2.2.3
  
  • scikit-learn=1.5.2
  
  • matplotlib=3.9.2
  
  • seaborn=0.13.2
  
  • pillow=10.4.0
  
  • tqdm=4.66.5
  
  • openpyxl=3.1.5
________________________________________
**5. Model Architectures**

•	XGBoost

•	U-Net++

•	AlexNet + Multilayer perceptron

•	VGG16 + Multilayer perceptron

•	ResNet18 + Multilayer perceptron

•	GradCam
________________________________________
**6. Features Used**
   
•	Morphometric features (nucleus-based)

•	Clinicopathologic features (age, sex, location)
________________________________________
**7. Evaluation Metrics**
   
•	XGBoost + SHAP – Classification (accuracy, area under the curve (AUC), F1-score, precision, recall and SHAP).

•	U-Net++ (Loss, Accuracy, Precision, Recall, IoU and Dice coefficient).

•	AlexNet (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	VGG16 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	ResNet18 (Loss, Accuracy, Precision, Recall, Confusion matrix (TP, FN, FP, TN), F1-score, Specificity, Receiver operating characteristic – area under the curve (ROC AUC) and Cohen's Kappa).

•	GradCam (morphometric)

________________________________________
**8. Repository Structure**
   
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

**9. Run models and reproduce tables**


<img width="1632" height="1041" alt="image" src="https://github.com/user-attachments/assets/c60aaf17-10f5-4f2c-b9fa-2fc5eb279eca" />


________________________________________

**10. Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL.git
cd Classification-RFH-and-FL

________________________________________

**11. Ethics**

This study was approved by the Ethics Committee of the Piracicaba Dental School, University of Campinas, Piracicaba, Brazil (protocol no. 67064422.9.1001.5418), 
and by the West of Scotland Research Ethics Service (20/WS/0017). The study was performed according to the clinical standards of the 1975 and 1983 Declaration of Helsinki. 
Written consent was not required as data was collected from surplus archived tissue. Data collected were fully anonymised.

________________________________________

**12. Data availability**

All the data derived from this study are included in the manuscript. We are unable to share the whole slide images and clinical data, due to restrictions in the 
ethics applications. However, we created synthetic slides to show the structure of the project.

________________________________________

**13. Code availability**

We have made the codes publicly available online, along with model weights (https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL). All code was written 
with Python Python 3.12.11, along with PyTorch 2.8.0. The full implementation of the model, including the code and documentation, has been deposited in the Zenodo repository 
and is publicly available (https://doi.org/10.1234/RFH_vs_FL_AI_pipeline). 

________________________________________
**14. Citation**

@article{delasouza2025,
  title={Deep Learning-Based Histopathologic Classification of Head and Neck Reactive Follicular Hyperplasia and Follicular Lymphoma},
  author={de Souza, Lucas Lacerda, Chen, Zhiyang […] Khurram, Syed Ali and Vargas, Pablo Agustin},
  journal={npj digital medicine},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
________________________________________
**15. License**

MIT License © 2025 Lucas Lacerda de Souza

