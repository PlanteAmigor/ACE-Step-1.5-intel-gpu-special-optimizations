#!/usr/bin/env bash
# ACE-Step Gradio Web UI Launcher - Linux Intel XPU (Manual Mode)
# For Intel Arc GPUs (A770, A750, A580, A380) and integrated graphics on Linux
# Requires: Python 3.11, PyTorch XPU from https://download.pytorch.org/whl/xpu
# IMPORTANT: Uses torch.xpu backend with SYCL/Level Zero acceleration

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== XPU Configuration ====================
# XPU performance optimization (from verified working setup)
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export PYTORCH_DEVICE=xpu

# Disable torch.compile (not fully supported on XPU yet)
export TORCH_COMPILE_BACKEND=eager

# HuggingFace tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Force torchaudio to use ffmpeg backend (torchcodec not available on XPU)
export TORCHAUDIO_USE_BACKEND=ffmpeg

# ==================== Venv Configuration ====================
VENV_DIR="${SCRIPT_DIR}/venv_xpu"

# ==================== Helper Functions ====================

_activate_venv() {
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        echo "Activating XPU virtual environment: $VENV_DIR"
        source "$VENV_DIR/bin/activate"
        return 0
    else
        echo ""
        echo "========================================"
        echo " ERROR: venv_xpu not found!"
        echo "========================================"
        echo ""
        echo "Please create the XPU environment first:"
        echo "  1. Run: python3 -m venv venv_xpu"
        echo "  2. Run: source venv_xpu/bin/activate"
        echo "  3. Run: pip install -r requirements-xpu.txt"
        echo ""
        exit 1
    fi
}

_verify_xpu() {
    if ! python3 -c "
import torch
xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
if xpu_available:
    print(f'XPU: Intel Arc GPU detected')
    print(f'PyTorch XPU version: {torch.__version__}')
else:
    print(f'WARNING: Intel XPU not detected')
    print(f'PyTorch version: {torch.__version__}')
" 2>&1; then
        echo ""
        echo "========================================"
        echo " WARNING: PyTorch XPU verification issue"
        echo "========================================"
        echo ""
    fi
}

# ==================== Interactive Configuration ====================

echo "======================================================"
echo "        ACE-Step XPU Manual Launch Mode (Linux)"
echo "======================================================"
echo ""
echo "This will guide you through configuring ACE-Step options."
echo ""

_activate_venv
_verify_xpu

# Update check prompt
echo "-------------------- Update Settings --------------------"
read -rp "Check for updates before launch? (y/N): " update_choice
CHECK_UPDATE=false
if [[ "${update_choice,,}" == "y" ]]; then
    CHECK_UPDATE=true
    echo "[Update] Update check enabled."
fi
echo ""

# Update check
if [[ "$CHECK_UPDATE" == "true" ]] && command -v git &>/dev/null; then
    cd "$SCRIPT_DIR"
    if git rev-parse --git-dir &>/dev/null 2>&1; then
        local_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
        local_commit="$(git rev-parse --short HEAD 2>/dev/null || echo "")"

        echo "[Update] Checking for updates..."
        git fetch origin --quiet 2>/dev/null || true
        remote_commit="$(git rev-parse --short "origin/$local_branch" 2>/dev/null || echo "")"

        if [[ -n "$remote_commit" && "$local_commit" != "$remote_commit" ]]; then
            echo ""
            echo "========================================"
            echo "  Update available!"
            echo "========================================"
            echo "  Current: $local_commit  ->  Latest: $remote_commit"
            echo ""
            git --no-pager log --oneline "HEAD..origin/$local_branch" 2>/dev/null | head -10
            echo ""
            read -rp "Update now? (Y/N): " do_update
            if [[ "${do_update^^}" == "Y" ]]; then
                git pull --ff-only origin "$local_branch" 2>/dev/null || echo "[Update] Failed."
            fi
        else
            echo "[Update] Already up to date."
        fi
        echo ""
    fi
fi

# -------------------- Select DiT Model --------------------
echo "-------------------- Select DiT Model --------------------"
echo "Scanning available models..."
echo ""

MODEL_COUNT=0
if [[ -d "$SCRIPT_DIR/checkpoints" ]]; then
    for d in "$SCRIPT_DIR/checkpoints"/acestep-v15-*; do
        if [[ -d "$d" ]]; then
            MODEL_COUNT=$((MODEL_COUNT + 1))
        fi
    done
fi

if [[ $MODEL_COUNT -eq 0 ]]; then
    echo "No acestep-v15 models found in checkpoints folder."
    echo "Using default: acestep-v15-turbo"
    CONFIG_PATH="--config_path acestep-v15-turbo"
    CONFIG_PATH_DISPLAY="acestep-v15-turbo (default)"
else
    echo "Available DiT models:"
    echo "1) acestep-v15-turbo  (Recommended - Fast generation)"
    echo "2) acestep-v15-base    (Base model)"
    echo "3) acestep-v15-sft     (Supervised fine-tuned)"
    echo "4) acestep-v15-turbo-rl  (RL optimized)"
    echo ""

    while true; do
        read -rp "Enter selection (1-4): " dit_choice
        case "$dit_choice" in
            1) CONFIG_PATH="--config_path acestep-v15-turbo"; CONFIG_PATH_DISPLAY="acestep-v15-turbo"; break;;
            2) CONFIG_PATH="--config_path acestep-v15-base"; CONFIG_PATH_DISPLAY="acestep-v15-base"; break;;
            3) CONFIG_PATH="--config_path acestep-v15-sft"; CONFIG_PATH_DISPLAY="acestep-v15-sft"; break;;
            4) CONFIG_PATH="--config_path acestep-v15-turbo-rl"; CONFIG_PATH_DISPLAY="acestep-v15-turbo-rl"; break;;
            *) echo "Invalid input. Please enter a number between 1 and 4.";;
        esac
    done
fi
echo ""

# -------------------- Select LM Model --------------------
echo "-------------------- Select LM Model --------------------"
echo "1) acestep-5Hz-lm-0.6B  (Recommended - Fast, low VRAM)"
echo "2) acestep-5Hz-lm-1.7B  (Balanced)"
echo "3) acestep-5Hz-lm-4B    (Best quality - requires CPU offload)"
echo "4) Launch without LM Model (DiT-only mode)"
echo ""

while true; do
    read -rp "Enter selection (1-4): " lm_choice
    case "$lm_choice" in
        1)
            LM_MODEL_PATH="--lm_model_path acestep-5Hz-lm-0.6B"
            LM_MODEL_PATH_DISPLAY="acestep-5Hz-lm-0.6B"
            INIT_LLM="--init_llm true"
            break;;
        2)
            LM_MODEL_PATH="--lm_model_path acestep-5Hz-lm-1.7B"
            LM_MODEL_PATH_DISPLAY="acestep-5Hz-lm-1.7B"
            INIT_LLM="--init_llm true"
            break;;
        3)
            LM_MODEL_PATH="--lm_model_path acestep-5Hz-lm-4B"
            LM_MODEL_PATH_DISPLAY="acestep-5Hz-lm-4B"
            INIT_LLM="--init_llm true"
            break;;
        4)
            LM_MODEL_PATH=""
            LM_MODEL_PATH_DISPLAY="None (DiT-only mode)"
            INIT_LLM="--init_llm false"
            break;;
        *) echo "Invalid input. Please enter a number between 1 and 4.";;
    esac
done
echo ""

# -------------------- CPU Offload Option --------------------
echo "-------------------- CPU Offload Option --------------------"
if [[ "$lm_choice" == "3" ]]; then
    echo "NOTE: 4B LM model requires CPU offload on most GPUs"
    OFFLOAD_TO_CPU="--offload_to_cpu true"
    OFFLOAD_DISPLAY="Enabled (required for 4B LM)"
else
    while true; do
        read -rp "Enable CPU Offload? (y/N): " offload_choice
        if [[ "${offload_choice,,}" == "y" ]]; then
            OFFLOAD_TO_CPU="--offload_to_cpu true"
            OFFLOAD_DISPLAY="Enabled"
            break
        elif [[ "${offload_choice,,}" == "n" || -z "$offload_choice" ]]; then
            OFFLOAD_TO_CPU="--offload_to_cpu false"
            OFFLOAD_DISPLAY="Disabled"
            break
        else
            echo "Invalid input. Please enter y or n."
        fi
    done
fi
echo ""

# ==================== Summary ====================
echo "======================================================"
echo "Configuration Summary"
echo "======================================================"
echo "DiT Model:    $CONFIG_PATH_DISPLAY"
echo "LM Model:     $LM_MODEL_PATH_DISPLAY"
echo "CPU Offload:  $OFFLOAD_DISPLAY"
echo ""
echo "Starting ACE-Step with these settings..."
echo ""

# Build command with optional parameters
CMD="--port 7860 --server-name 127.0.0.1 --language en"
[[ -n "$CONFIG_PATH" ]] && CMD="$CMD $CONFIG_PATH"
[[ -n "$LM_MODEL_PATH" ]] && CMD="$CMD $LM_MODEL_PATH"
[[ -n "$OFFLOAD_TO_CPU" ]] && CMD="$CMD $OFFLOAD_TO_CPU"
[[ -n "$INIT_LLM" ]] && CMD="$CMD $INIT_LLM"
CMD="$CMD --init_service true"

# Run in background so we can handle signals ourselves
cd "$SCRIPT_DIR" && python3 -u acestep/acestep_v15_pipeline.py $CMD &
PID=$!

# Forward Ctrl+C as SIGTERM (bypasses Gradio's SIGINT handler for graceful shutdown)
trap 'echo ""; echo "Shutting down..."; kill -TERM $PID 2>/dev/null; wait $PID; exit 0' SIGINT SIGTERM

# Wait for process to finish
wait $PID
