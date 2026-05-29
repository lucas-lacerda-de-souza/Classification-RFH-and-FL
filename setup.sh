#!/bin/bash

###############################################################################
# Setup Script for RFH-FL Classification Project
# Uses uv for fast, reliable environment and dependency management
# Supports both CPU and GPU configurations, and venv or system installs
#
# Usage:
#   ./setup.sh --cuda              # GPU + venv (default)
#   ./setup.sh --cpu               # CPU + venv (default)
#   ./setup.sh --cuda --system     # GPU, install into system/current environment
#   ./setup.sh --cpu  --system     # CPU, install into system/current environment
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_PATH=".venv"
PYTHON_VERSION="3.12"
INSTALL_TYPE=""      # "gpu" | "cpu"
ENV_MODE="venv"      # "venv" | "system"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

print_usage() {
    echo "Usage: $0 --cuda|--cpu [--system] [--help]"
    echo ""
    echo "Required (pick one):"
    echo "  --cuda      Install PyTorch with CUDA 12.8 support (GPU)"
    echo "  --cpu       Install PyTorch CPU-only"
    echo ""
    echo "Optional:"
    echo "  --system    Install into the current/system Python environment instead"
    echo "              of creating a virtual environment. Useful for Kaggle, Colab,"
    echo "              or any managed environment where venvs are not supported."
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --cuda               # GPU install, isolated venv (recommended)"
    echo "  $0 --cpu                # CPU install, isolated venv (recommended)"
    echo "  $0 --cuda --system      # GPU install, no venv (Kaggle / Colab)"
    echo "  $0 --cpu  --system      # CPU install, no venv (Kaggle / Colab)"
}

###############################################################################
# Parse Arguments
###############################################################################

parse_args() {
    if [ $# -eq 0 ]; then
        print_error "No argument provided. You must specify --cuda or --cpu."
        echo ""
        print_usage
        exit 1
    fi

    for arg in "$@"; do
        case "$arg" in
            --cuda)
                INSTALL_TYPE="gpu"
                ;;
            --cpu)
                INSTALL_TYPE="cpu"
                ;;
            --system)
                ENV_MODE="system"
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown argument: $arg"
                echo ""
                print_usage
                exit 1
                ;;
        esac
    done

    if [ -z "$INSTALL_TYPE" ]; then
        print_error "You must specify --cuda or --cpu."
        echo ""
        print_usage
        exit 1
    fi

    # Summary
    local torch_label="CPU"
    [ "$INSTALL_TYPE" = "gpu" ] && torch_label="GPU (CUDA 12.8)"

    local env_label="venv ($VENV_PATH)"
    [ "$ENV_MODE" = "system" ] && env_label="system (no venv)"

    print_info "PyTorch: $torch_label"
    print_info "Environment: $env_label"
}

###############################################################################
# Install uv
###############################################################################

install_uv() {
    print_header "Checking uv"

    if command -v uv &> /dev/null; then
        print_success "uv $(uv --version | awk '{print $2}') already installed"
        return
    fi

    print_info "uv not found — installing..."
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &> /dev/null; then
        wget -qO - https://astral.sh/uv/install.sh | sh
    else
        print_error "Neither curl nor wget found. Install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    # Make uv available in this session immediately
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        print_error "uv installation failed. Install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    print_success "uv $(uv --version | awk '{print $2}') installed"
}

###############################################################################
# Check CUDA
###############################################################################

check_cuda() {
    if [ "$INSTALL_TYPE" = "gpu" ]; then
        print_info "Checking CUDA availability..."
        if command -v nvidia-smi &> /dev/null; then
            CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
            print_success "NVIDIA GPU detected (Driver: $CUDA_DRIVER)"
        else
            print_warning "NVIDIA GPU not detected, but --cuda was requested. Proceeding anyway."
        fi
    fi
}

###############################################################################
# Environment Setup
###############################################################################

setup_environment() {
    if [ "$ENV_MODE" = "venv" ]; then
        print_header "Creating Virtual Environment"

        if [ -d "$VENV_PATH" ]; then
            print_warning "Virtual environment already exists at $VENV_PATH — removing and recreating..."
            rm -rf "$VENV_PATH"
        fi

        print_info "Creating virtual environment with Python $PYTHON_VERSION..."
        uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
        print_success "Virtual environment created at $VENV_PATH"

        print_info "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
        print_success "Virtual environment activated"

    else
        print_header "System Environment Mode"
        print_info "Skipping venv creation — installing into the current environment."
        print_warning "Packages will be installed system-wide (or into the active managed env)."
    fi
}

###############################################################################
# Dependency Installation
###############################################################################

# Build the uv pip install base command.
# In system mode: --system bypasses the "no active venv" safety check in uv,
# which is exactly what Kaggle/Colab need.
# In venv mode: the activated venv is already on PATH, so no extra flag needed.
uv_pip() {
    if [ "$ENV_MODE" = "system" ]; then
        uv pip install --system "$@"
    else
        uv pip install "$@"
    fi
}

install_dependencies() {
    if [ "$INSTALL_TYPE" = "gpu" ]; then
        print_header "Installing dependencies (GPU mode — CUDA 12.8)"
        uv_pip \
            torch==2.8.0 \
            torchvision==0.23.0 \
            torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cu128
        print_success "PyTorch GPU (CUDA 12.8) installed"
    else
        print_header "Installing dependencies (CPU mode)"
        uv_pip \
            torch==2.8.0 \
            torchvision==0.23.0 \
            torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cpu
        print_success "PyTorch CPU installed"
    fi

    uv_pip -e ".[dev]"
    print_success "Project dependencies installed"
}

###############################################################################
# Verification
###############################################################################

verify_installation() {
    print_header "Verifying Installation"

    python3 << 'EOF'
import sys
import importlib

packages = {
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "sklearn": "Scikit-learn",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "openpyxl": "OpenPyXL",
    "xgboost": "XGBoost",
    "shap": "SHAP",
}

print("\nInstalled Packages:")
print("-" * 40)

all_ok = True
for module_name, display_name in packages.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {display_name:<20} {version}")
    except ImportError:
        print(f"✗ {display_name:<20} NOT INSTALLED")
        all_ok = False

if all_ok:
    print("\n✓ All dependencies verified!")
    sys.exit(0)
else:
    print("\n✗ Some dependencies are missing!")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "All dependencies verified"
    else
        print_error "Some dependencies are missing"
        exit 1
    fi
}

###############################################################################
# Final Instructions
###############################################################################

print_final_instructions() {
    print_header "Setup Complete!"
    echo ""
    print_success "Environment setup completed successfully!"
    echo ""

    if [ "$ENV_MODE" = "venv" ]; then
        echo -e "${BLUE}Next Steps:${NC}"
        echo "1. Activate the environment (future sessions):"
        echo -e "   ${YELLOW}source $VENV_PATH/bin/activate${NC}"
        echo ""
        echo "   Or run without activating:"
        echo -e "   ${YELLOW}uv run python inferance.py --input_dir ./data/test/ --output_dir ./results/${NC}"
    else
        echo -e "${BLUE}Next Steps:${NC}"
        echo "   Packages are installed into the current environment — no activation needed."
        echo -e "   ${YELLOW}python inferance.py --input_dir ./data/test/ --output_dir ./results/${NC}"
    fi

    echo ""
    echo "2. Verify PyTorch:"
    echo -e "   ${YELLOW}python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'${NC}"
    echo ""
    echo "3. Check GPU:"
    echo -e "   ${YELLOW}python -c 'import torch; print(f\"GPU available: {torch.cuda.is_available()}\")'${NC}"
    echo ""
    echo -e "${BLUE}Documentation:${NC}"
    echo "  - README.md        - Project overview and usage"
    echo "  - MODEL_CARD.txt   - Model details and specifications"
    echo "  - models/          - Model implementations"
    echo ""
}

###############################################################################
# Main
###############################################################################

main() {
    clear
    print_header "RFH-FL Classification Project Setup"
    echo ""
    echo "This script will set up the complete development environment"
    echo "for the RFH-FL AI classification framework."
    echo ""

    parse_args "$@"
    echo ""

    install_uv
    echo ""

    check_cuda
    echo ""

    setup_environment
    echo ""

    install_dependencies
    echo ""

    verify_installation
    echo ""

    print_final_instructions
}

main "$@"