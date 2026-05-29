- Files with training code: multimodal_alexnet_patch_level.py, multimodal_alexnet_patient_level.py, multimodal_resnet18_patch_level.py, multimodal_resnet18_patient_level.py, etc.

How to train (quick steps)

- **1) Set up environment**
  - Linux/macOS:
    ```bash
    chmod +x setup.sh
    ./setup.sh --cpu     # or --cuda for GPU
    ```
  - Windows:
    ```
    setup.bat --cpu
    ```
  - After it finishes: activate the venv
    - Linux/macOS: `source .venv/bin/activate`
    - Windows: `.venv\Scripts\activate.bat`

- **2) Prepare data**
  - Place data under the repository `data/` directory.
  - Required structure for each split (train/val/test):
    - `data/train/`, `data/val/`, `data/test/`
    - Each split contains class subfolders (matching labels used in Excel) and `CaseID` subfolders with image patches, e.g.:
      - `data/train/<Classe>/<CaseID>/*.png`
  - Excel metadata files (required columns):
    - `data/train/clinical_data_train.xlsx`
    - `data/val/clinical_data_val.xlsx`
    - `data/test/clinical_data_test.xlsx`
    - Each Excel file must contain `Classe` and `CaseID` columns plus the clinical/morphometric feature columns used by the dataset.

- **3) Run training**
  - From repository root, run the model script you want. Examples:
    - AlexNet (patch-level):
      ```bash
      python models/multimodal_alexnet_patch_level.py
      ```
    - ResNet18 (patch-level):
      ```bash
      python models/multimodal_resnet18_patch_level.py
      ```
    - Patient-level variants:
      ```bash
      python models/multimodal_alexnet_patient_level.py
      python models/multimodal_resnet18_patient_level.py
      ```
  - Notes:
    - Scripts use fixed hyperparameters (batch size, epochs) inside the file; edit them if you need different values.
    - Scripts automatically use GPU if `torch.cuda.is_available()`; ensure GPU drivers/CUDA are installed for `--cuda` setup.
    - To run on multiple GPUs the scripts already wrap model with `nn.DataParallel()` where present.

- **4) Monitor & resume**
  - Training prints epoch-level progress and saves the best model to a results folder, e.g. `results/follicular_multimodal_alexnet/best_model.pt`.
  - Training history saved as `training_history.xlsx` and metrics as `metrics.csv` / `metrics.txt`.
  - To resume training:
    - Modify the script to load checkpoint before optimizer steps:
      ```python
      model.load_state_dict(torch.load("path/to/checkpoint.pt"))
      ```
    - Recreate optimizer and continue training loop.

- **5) Evaluation & outputs**
  - After training the script evaluates on test set and writes:
    - `results/.../metrics.csv`, `metrics.txt`
    - `results/.../confusion_matrix.png`, `roc_curve.png`
  - Use these files for analysis and reporting.

- **6) Quick troubleshooting**
  - If missing columns error: ensure Excel contains `Classe` and `CaseID`.
  - If OOM on GPU: reduce `batch_size` in the script or switch to `--cpu`.
  - If torchvision/torch version errors: use the `setup` script with `--cuda`/`--cpu` (it installs compatible torch) or install appropriate wheels from PyTorch instructions.

