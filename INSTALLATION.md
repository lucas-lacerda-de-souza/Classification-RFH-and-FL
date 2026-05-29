# Installation Guide

This document provides step-by-step instructions for setting up the RFH-FL Classification project on your system.

## Quick Start (Automated)

### Windows Users

**Interactive mode (prompts for choice):**
```bash
setup.bat
```

**GPU (CUDA 12.8) mode:**
```bash
setup.bat --cuda
```

**CPU-only mode:**
```bash
setup.bat --cpu
```

### Linux/macOS Users

**Interactive mode (prompts for choice):**
```bash
chmod +x setup.sh
./setup.sh
```

**GPU (CUDA 12.8) mode:**
```bash
chmod +x setup.sh
./setup.sh --cuda
```

**CPU-only mode:**
```bash
chmod +x setup.sh
./setup.sh --cpu
```

**Help:**
```bash
./setup.sh --help
```

The automated setup will:
- ✓ Check system requirements (Python 3.12)
- ✓ Detect CUDA/GPU availability (in interactive mode)
- ✓ Create a virtual environment
- ✓ Install all dependencies (PyTorch, scientific libraries, etc.)
- ✓ Verify the installation
- ✓ Provide next steps

---

## Command-Line Arguments

Both `setup.sh` and `setup.bat` support command-line arguments to automate the installation without prompts:

### Available Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--cuda` | Install with CUDA 12.8 GPU support | `setup.bat --cuda` |
| `--cpu` | Install CPU-only version | `./setup.sh --cpu` |
| `--help` | Show help message (Linux/macOS only) | `./setup.sh --help` |
| *(no argument)* | Interactive mode - prompts for choice | `setup.bat` |

### Usage Examples

**Automated GPU setup (no prompts):**
```bash
# Linux/macOS
./setup.sh --cuda

# Windows
setup.bat --cuda
```

**Automated CPU setup (no prompts):**
```bash
# Linux/macOS
./setup.sh --cpu

# Windows
setup.bat --cpu
```

**Interactive mode (recommended for first-time users):**
```bash
# Linux/macOS
./setup.sh

# Windows
setup.bat
```

---

If you prefer manual installation or the automated scripts don't work for you, follow these steps:

### 1. Prerequisites
- **Python:** 3.12 or higher ([Download](https://www.python.org/downloads/))
- **pip:** Latest version (usually comes with Python)
- **NVIDIA drivers** (Optional, for GPU support): CUDA 12.8 compatible drivers
- **Git:** For cloning the repository

### 2. Clone the Repository
```bash
git clone https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL.git
cd Classification-RFH-and-FL
```

### 3. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Upgrade pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Install PyTorch

**For GPU (CUDA 12.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
```

**Note:** The setup scripts use flexible version resolution to ensure compatibility across different PyTorch distributions. Minimum supported versions are PyTorch 2.0.0, torchvision 0.15.0, and torchaudio 2.0.0.

### 6. Install Project Dependencies

Install with development tools:
```bash
pip install -e ".[dev]"
```

Or minimal installation:
```bash
pip install -e .
```

### 7. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

---

## Package Structure

The `pyproject.toml` defines the following extras:

### Core Dependencies
```
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
pillow==10.4.0
tqdm==4.66.5
openpyxl==3.1.5
xgboost>=2.0.0
shap>=0.45.0
```

### Optional Dependencies

**PyTorch (CPU):**
```bash
pip install -e ".[torch-cpu]"
```

**PyTorch (GPU):**
```bash
pip install -e ".[torch-cuda]"
```

**Development Tools:**
```bash
pip install -e ".[dev]"
```
Includes: pytest, pytest-cov, black, flake8, mypy, jupyter, ipython

**Documentation:**
```bash
pip install -e ".[docs]"
```
Includes: sphinx, sphinx-rtd-theme

---

## Troubleshooting

### Virtual Environment Creation Error: ensurepip Failed

**Error:** `Command '[.../.venv/bin/python3', '-m', 'ensurepip', '--upgrade', '--default-pip']' returned non-zero exit status 1.`

**Solution:** This commonly occurs in restricted environments like Kaggle. Our updated setup scripts now handle this automatically:

1. **Automatic fallback:** The setup scripts will try creating a venv without pip first, then install pip manually
2. **Manual fix:** If the automated script still fails, try:

**Linux/macOS:**
```bash
python3 -m venv --without-pip .venv
source .venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
pip install -e .
```

**Windows:**
```powershell
python -m venv --without-pip .venv
.venv\Scripts\activate.bat
powershell -Command "(New-Object System.Net.WebClient).DownloadString('https://bootstrap.pypa.io/get-pip.py') | python"
pip install -e .
```

3. **For Kaggle specifically:** Use `--cpu` argument for non-interactive setup:
```bash
./setup.sh --cpu
```

---

### Python Version Issues
```bash
# Check your Python version
python --version

# If not 3.12, specify the full path
/usr/bin/python3.12 -m venv .venv
```

### CUDA Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Update drivers from: https://www.nvidia.com/Download/driverDetails.aspx
```

### Virtual Environment Not Activating

**Windows - Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

**Linux/macOS - Permission Error:**
```bash
chmod +x .venv/bin/activate
source .venv/bin/activate
```

### PyTorch Version Compatibility

**Error:** `Could not find a version that satisfies the requirement torchvision==X.Y.Z`

**Solution:** The setup scripts now use flexible version specifications to handle different PyTorch distributions. If you encounter version conflicts:

1. **Let pip resolve versions automatically:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

2. **Check available versions:**
   ```bash
   pip index versions torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install compatible group:**
   ```bash
   pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

### PyTorch Not Installing

Try alternative PyTorch index:
```bash
# Official index with all versions
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
```

### Import Errors After Installation

Reinstall development mode:
```bash
pip install --force-reinstall -e .
```

---

## System Requirements

### Minimum (CPU-only)
- **CPU:** 8+ cores recommended
- **RAM:** 8 GB minimum (16 GB+ recommended)
- **Disk:** 10 GB for dependencies + data

### Recommended (GPU)
- **GPU:** NVIDIA with CUDA Compute Capability 7.0+
- **VRAM:** 8 GB minimum (24 GB+ recommended for large batches)
- **RAM:** 32 GB+
- **Disk:** 50+ GB for data and model weights

### Tested Configuration
- **CPU:** Intel Xeon W-2295 (18 cores)
- **RAM:** 125 GB
- **GPU:** 3× NVIDIA GeForce RTX 3090 (24 GB each)
- **CUDA:** 12.8 compatible
- **Python:** 3.12.11

---

## Next Steps

1. **Run Inference:**
   ```bash
   python inferance.py --input_dir ./data/test/ --output_dir ./results/
   ```

2. **View Results:**
   ```bash
   # Results are saved in ./results/ directory
   ls -la results/
   ```

3. **Using Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Run Tests (if available):**
   ```bash
   pytest tests/
   ```

---

## References

- **Project Repository:** https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL
- **Model Weights:** https://doi.org/10.5281/zenodo.18190502
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit
- **Python 3.12:** https://www.python.org/downloads/release/python-3120/

---

## Support

For issues or questions:
1. Check the [README.md](README.md) for project overview
2. Review [MODEL_CARD.txt](MODEL_CARD.txt) for model specifications
3. Open an issue on [GitHub](https://github.com/lucas-lacerda-de-souza/Classification-RFH-and-FL/issues)

---

**Last Updated:** May 2026
