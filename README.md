**A multimodal explainable ai framework to assist in the differential diagnosis of head and neck reactive follicular hyperplasia and follicular lymphoma: an international multicentre study** 

Author: Lucas Lacerda de Souza

Year: 2026
________________________________________
**1. Project Overview**

This project implements a multimodal artificial intelligence pipeline for the classification of Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).
The framework integrates histopathological image patches, cell-level representations, spatial cell–cell relationships, clinicopathological data, and nuclear morphometric features.

The pipeline combines:
Traditional machine learning (XGBoost with SHAP),
Deep learning (CNNs + multilayer perceptron),
Vision Transformer–based cell modelling (CellViT++),
Graph Neural Networks for spatial analysis (Cell-GNN),
Explainable AI methods (Grad-CAM).
The system is intended for research and proof-of-concept use in computational pathology.
________________________________________
**2. Pipeline**


![Figure 1](https://github.com/user-attachments/assets/d211d37b-958d-42a2-95ce-ad378f49df61)



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

•	XGBoost + SHAP (clinicopathological and morphometric modelling)

•	U-Net++ (tissue-level segmentation for patch extraction)

•	AlexNet + Multilayer Perceptron (multimodal fusion)

•	VGG16 + Multilayer Perceptron (multimodal fusion)

•	ResNet18 + Multilayer Perceptron (multimodal fusion)

•	CellViT / CellViT++ (Vision Transformer–based nuclear segmentation and cell-level embedding extraction)

•	Cell Graph Neural Network (Cell-GNN) (cell–cell spatial modelling using k-nearest neighbour graphs)

•	Grad-CAM (CNN-based interpretability)

Note:
CellViT / CellViT++ and Cell-GNN are established architectures. Their original implementations were used without architectural modification. Only inference, downstream analysis, and integration code are included in this repository.
________________________________________
**6. Features Used**

• H&E image patches (299 × 299 pixels, 20×)

• Segmented patches (U-Net++)

• Cell-level embeddings and nuclear masks (CellViT / CellViT++)

• Cell–cell spatial descriptors (mean intercellular distances from Cell-GNN)

• Nucleus-based morphometric features

• Clinicopathological features (age, sex, lesion location)
________________________________________
**7. Evaluation Metrics**
   
• XGBoost + SHAP:
Accuracy, AUC, F1-score, Precision, Recall, SHAP values

• U-Net++ / CellViT++ (segmentation):
Loss, Accuracy, Precision, Recall, IoU, Dice coefficient

• CNN-based models (AlexNet, VGG16, ResNet18):
Loss, Accuracy, Precision, Recall, Confusion Matrix (TP, FN, FP, TN),
F1-score, Specificity, ROC AUC, Cohen’s Kappa

• Cell-GNN (spatial analysis):
Mean intercellular distances, effect sizes (Cliff’s delta, Hedges’ g),
Mann–Whitney U test and Welch’s t-test

• Grad-CAM:
Accuracy, AUC, F1-score, Precision, Recall (classification-based evaluation)

________________________________________
**8. Repository Structure**
   
## 📂 Repository Structure

DATA - Data used in the training

MODELS - Models used in the study

RESULTS - Results of the study

inferance.py — Inference Script Example

LICENSE.txt — Project license

MODEL_CARD.txt — Description of the essential information of the study 

README.md — Documentation and usage instructions

REQUIREMENTS.txt — Dependencies

________________________________________

**9. Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL.git
cd Classification-RFH-and-FL

________________________________________

**10. Quick Start Guide**

**10.1. Clone the repository**

git clone https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL.git
cd Classification-RFH-and-FL

**10.2. Create and activate the environment**

conda env create -f environment.yml
conda activate rfh-fl-ai

**10.3. Run inference**

python inferance.py --input_dir ./data/test/ --output_dir ./results/

**10.4. Generate Grad-CAM heatmaps**

python scripts/visualize_gradcam.py \
  --model resnet18 \
  --input_dir ./data/test/ \
  --output_dir ./gradcam/heatmaps/
________________________________________

**11. Compliance with TRIPOD-AI and CLAIM 2024 Guidelines**

This repository has been structured to meet the TRIPOD-AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis – 
AI extension) and CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging) requirements for transparent and reproducible AI in healthcare.

**Data Source and Splits**

Detailed in README.md → Dataset Organization and METHODS.md.
Data divided into 80% training, 10% validation, and 10% testing.
Two independent external validation cohorts used to assess generalizability.

**Model Architecture and Training**

Documented in /models and individual training scripts.
Includes optimizer (AdamW), learning rate, batch size, epochs, and loss functions.

**Performance Metrics**

Internal and external validation results summarized in /results
Cross-institutional evaluation demonstrates robustness to domain shifts.

**Interpretability and Explainability**

SHAP feature importance for XGBoost models and Grad-CAM heatmaps for CNNs included.
Code and examples available in /models and /data.

**Clinical and Biological Relevance**

Described in MODEL_CARD.md → Intended Use.
Designed to assist diagnostic workflows, not to replace expert evaluation.

**Limitations and Potential Biases**

Outlined in MODEL_CARD.
Includes dataset size, center-specific staining differences, and potential bias from single-institution data predominance.

**Ethical Considerations**

Discussed in MODEL_CARD.md → Ethical and Practical Considerations.
Model not intended for autonomous clinical use; human oversight required at all stages.

________________________________________

**12. Ethics**

This study was approved by the Ethics Committee of the Piracicaba Dental School, University of Campinas, Piracicaba, Brazil (protocol no. 67064422.9.1001.5418), 
and by the West of Scotland Research Ethics Service (20/WS/0017). The study was performed according to the clinical standards of the 1975 and 1983 Declaration of Helsinki. 
Written consent was not required as data was collected from surplus archived tissue. Data collected were fully anonymised.

________________________________________

**13. Data availability**

All the data derived from this study are included in the manuscript. All the dataset can only be accessed through collaborative investigations and additional ethical approvals. However, we created synthetic slides to show the structure of the project.

________________________________________

**14. Code availability**

We have made the codes publicly available online, along with model weights (https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL). All code was written 
with Python Python 3.12.11, along with PyTorch 2.8.0. The full implementation of the model, including the code and documentation, has been deposited in the Zenodo repository 
and is publicly available (https://doi.org/10.5281/zenodo.18190502). 

________________________________________
**15. Citation**

de Souza LL, Chen Z, de Cáceres CVBL, Rodrigues NG, Pontes HAR, Soares CD, Coracin FL, Kimura I, Siqueira SAC, Mariano FV, Kimura TC, Bonfitto JFL, de Freitas LLL, de Aquino SN, Hankinson P, Mahmood H, Walsh H, Torres Torres IA, Mosqueda-Taylor A, Aleixo PB, Martins MD, Menna-Barreto CL, Hagag A, Nakamura TCR, Dos Santos GC, Moraes MC, Lopes MA, Santos-Silva AR, Fonseca FP, Khurram SA, Vargas PA. A multimodal explainable ai framework to assist in the differential diagnosis of head and neck reactive follicular hyperplasia and follicular lymphoma: an international multicentre study. Virchows Arch. 2026 May 2. doi: 10.1007/s00428-026-04527-w. 

________________________________________
**16. License**

MIT License © 2026 Lucas Lacerda de Souza

