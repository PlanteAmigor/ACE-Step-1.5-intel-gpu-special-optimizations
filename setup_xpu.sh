#!/usr/bin/env bash
# ACE-Step XPU Environment Setup Script - Linux
# This script creates the venv_xpu virtual environment and installs all dependencies
# For Intel Arc GPUs (A770, A750, A580, A380) and integrated graphics on Linux
#
# Prerequisites:
#   - Python 3.11 installed and in PATH
#   - Intel GPU with latest compute-runtime drivers
#   - Internet connection for first-time installation
#   - ~5-10 GB disk space
#
# Intel GPU driver setup on Linux:
#   https://dgpu-docs.intel.com/driver/installation.html

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo "    ACE-Step 1.5 - Intel XPU Environment Setup (Linux)"
echo "======================================================"
echo ""
echo "This script will:"
echo "  1. Create venv_xpu virtual environment (Python 3.11)"
echo "  2. Install PyTorch XPU (Intel GPU support)"
echo "  3. Install all ACE-Step dependencies"
echo ""
echo "Requirements:"
echo "  - Python 3.11+ installed and in PATH"
echo "  - Intel Arc GPU with latest compute-runtime drivers"
echo "  - Internet connection for first-time installation"
echo "  - ~5-10 GB disk space"
echo ""

# ==================== Check Python ====================
check_python() {
    if ! command -v python3 &>/dev/null; then
        echo "========================================"
        echo " ERROR: Python 3 not found!"
        echo "========================================"
        echo ""
        echo "Please install Python 3.11 from:"
        echo "  https://www.python.org/downloads/"
        echo ""
        echo "Or use your system package manager:"
        echo "  sudo apt install python3.11 python3.11-venv python3.11-dev  # Debian/Ubuntu"
        echo "  sudo dnf install python3.11 python3.11-venv                # Fedora"
        echo "  sudo pacman -S python311                                    # Arch"
        echo ""
        exit 1
    fi

    local py_ver
    py_ver="$(python3 --version 2>&1)"
    echo "[Python] Found: $py_ver"

    # Check Python version is 3.10+
    local py_major py_minor
    py_major="$(python3 -c 'import sys; print(sys.version_info.major)')"
    py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)')"
    if [[ "$py_major" -lt 3 || "$py_minor" -lt 10 ]]; then
        echo ""
        echo "WARNING: Python 3.10+ recommended, but found: $py_ver"
        echo ""
        read -rp "Continue anyway? (y/N): " continue_anyway
        if [[ ! "${continue_anyway,,}" == "y" ]]; then
            echo "Please install Python 3.11 for best compatibility."
            exit 1
        fi
    fi
}

# ==================== Check venv ====================
create_venv() {
    if [[ -d "$SCRIPT_DIR/venv_xpu" ]]; then
        echo ""
        echo "========================================"
        echo "  venv_xpu already exists!"
        echo "========================================"
        echo ""
        echo "Location: $SCRIPT_DIR/venv_xpu"
        echo ""
        read -rp "Recreate virtual environment? (y/N): " recreate
        if [[ "${recreate,,}" == "y" ]]; then
            echo ""
            echo "Removing old venv_xpu..."
            rm -rf "$SCRIPT_DIR/venv_xpu"
        else
            echo ""
            echo "Existing environment will be updated."
            return 0
        fi
    fi

    echo ""
    echo "========================================"
    echo "Step 1: Creating virtual environment"
    echo "========================================"
    echo ""
    echo "Running: python3 -m venv venv_xpu"
    echo ""
    python3 -m venv "$SCRIPT_DIR/venv_xpu"
    echo "Virtual environment created successfully!"
}

# ==================== Install dependencies ====================
install_deps() {
    echo ""
    echo "========================================"
    echo "Step 2: Activating virtual environment"
    echo "========================================"
    echo ""
    source "$SCRIPT_DIR/venv_xpu/bin/activate"

    echo ""
    echo "========================================"
    echo "Step 3: Upgrading pip"
    echo "========================================"
    echo ""
    python3 -m pip install --upgrade pip -q
    echo "pip upgraded successfully!"

    echo ""
    echo "========================================"
    echo "Step 4: Installing PyTorch XPU (Intel GPU)"
    echo "========================================"
    echo ""
    echo "Installing PyTorch with Intel XPU support..."
    echo ""
    # Install PyTorch XPU from official Intel index (+xpu version)
    # --index-url 指定 XPU 源，--extra-index-url 指定公共依赖源
    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/xpu \
      --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 || {
        echo ""
        echo "ERROR: PyTorch XPU installation failed."
        echo "Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu"
        echo ""
    }

    echo ""
    echo "========================================"
    echo "Step 5: Installing ACE-Step dependencies"
    echo "========================================"
    echo ""
    echo "This will take a few minutes on first run..."
    echo ""

    if [[ ! -f "$SCRIPT_DIR/requirements-xpu.txt" ]]; then
        echo "ERROR: requirements-xpu.txt not found!"
        echo "Make sure you are running this script from the ACE-Step-1.5 root directory."
        exit 1
    fi

    # Install dependencies, skipping torch lines and platform-specific markers
    # On Linux, torch is installed separately above; requirements-xpu.txt only pins torch for win32
    pip install -r "$SCRIPT_DIR/requirements-xpu.txt" 2>&1 || {
        echo ""
        echo "WARNING: Some packages may have failed to install."
        echo "This can happen due to network issues or incompatible package versions."
        echo "Trying to continue with available packages..."
    }

    echo ""
    echo "========================================"
    echo "Step 6: Verifying Installation"
    echo "========================================"
    echo ""

    # Verify PyTorch installation
    echo "Checking PyTorch installation..."
    if python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
print(f'XPU available: {xpu_available}')
" 2>&1; then
        echo ""
        echo "PyTorch installed successfully!"
    else
        echo ""
        echo "WARNING: PyTorch verification had issues."
    fi
}

# ==================== Main ====================
check_python
create_venv

# Activate and install
source "$SCRIPT_DIR/venv_xpu/bin/activate"
install_deps

echo ""
echo "======================================================"
echo "     Installation Complete!"
echo "======================================================"
echo ""
echo "Your ACE-Step XPU environment is ready to use!"
echo ""
echo "Next steps:"
echo "  1. Download ACE-Step models to the 'checkpoints' folder"
echo "     (if not already present)"
echo ""
echo "  2. Launch the Gradio UI:"
echo "     ./start_gradio_ui_xpu.sh"
echo ""
echo "  3. Or launch with manual model selection:"
echo "     ./start_gradio_ui_xpu_manual.sh"
echo ""
echo "  4. Or launch the API server:"
echo "     ./start_api_server_xpu.sh"
echo ""
echo "To activate the environment manually:"
echo "  source venv_xpu/bin/activate"
echo ""
echo "======================================================"
echo ""
