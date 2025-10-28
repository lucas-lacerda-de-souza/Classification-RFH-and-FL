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

**7. Run models and reproduce tables**

Output (Paper files) 	Script/Notebook   	Command  
**Figure 1** - Workflow for morphometric analysis evidenced that whole-slide histological images were divided into tiles and segmented at the patch level.	GraphPad Prism	GraphPad Prism
**Figure 2** - Traditional machine learning for clinicopathological and morphometric data.	`models/XGBoost Classification to xgboost_classification_cpc_mpa.py`	`python "models/XGBoost Classification to xgboost_classification_cpc_mpa.py"`
**Figure 3** - Overview of the explainability-guided pipeline showing Grad-CAM was applied to whole-slide images to localise high-importance regions.	`models/xgboost_classification_gradcam.py`	`python "models/xgboost_classification_gradcam.py"`
**Figure 4** - Clinical and histopathological features of reactive follicular hyperplasia and follicular lymphoma in the oral cavity. 	Not applicable	Not applicable
**Figure 5** - Pipeline of the study.	Not applicable	Not applicable
**Table 1** - Metrics of the models in the multimodal in patch and patient-level analysis.	`models/multimodal_alexnet_patch_level.py`	`python "models/multimodal_alexnet_patch_level.py"`
	`models/multimodal_alexnet_patient_level.py`	`python "models/multimodal_alexnet_patient_level.py"`
	`models/multimodal_resnet18_patch_level.py`	`python "models/multimodal_resnet18_patch_level.py"`
	`models/multimodal_resnet18_patient_level.py`	`python "models/multimodal_resnet18_patient_level.py"`
	`models/multimodal_vgg16_patch_level.py`	`python "models/multimodal_vgg16_patch_level.py"`
	`models/multimodal_vgg16_patient_level.py`	`python "models/multimodal_vgg16_patient_level.py"`
	`models/segmentation_unet++.py`	`python "models/models/segmentation_unet++.py"`
**Supplementary Table 1** - Morphometric analysis of cellular nuclei derived from the whole-slide images used in this study. 	`data/supplementary_table_1.xlsx`	`python "data/supplementary_table_1.xlsx"`
**Supplementary Table 2** - Clinicopathologic and morphometric features used in the XGBoost analysis. 	`data/supplementary_table_2.xlsx`	`python "data/supplementary_table_2.xlsx"`
**Supplementary Table 3** - Clinicopathologic and morphometric features used in the multimodal analysis.	`data/supplementary_table_3.xlsx`	`python "data/supplementary_table_3.xlsx"`
**Supplementary Table 4** - Morphometric measures of the regions after Grad-Cam analysis for each patch. 	`data/supplementary_table_4.xlsx`	`python "data/supplementary_table_4.xlsx"`
**Supplementary Table 5** - Pathologists’ evaluation of the WSIs. 	`data/supplementary_table_5.xlsx`	`python "data/supplementary_table_5.xlsx"`
<img width="1332" height="741" alt="image" src="https://github.com/user-attachments/assets/f0a8059e-7d40-4750-9e89-186120ec858f" />

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

