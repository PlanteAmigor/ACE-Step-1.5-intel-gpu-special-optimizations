#!/usr/bin/env bash
# ACE-Step REST API Server Launcher - Linux Intel XPU
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

# ==================== Server Configuration ====================
HOST="${HOST:-127.0.0.1}"
# HOST="0.0.0.0"
PORT="${PORT:-8001}"

# ==================== Model Configuration ====================
# API key for authentication (optional)
API_KEY="${API_KEY:-}"
# API_KEY="--api-key sk-your-secret-key"

# Download source: auto, huggingface, modelscope
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-}"

# LLM (Language Model) initialization settings
# By default, LLM is auto-enabled/disabled based on GPU VRAM:
#   - <=6GB VRAM: LLM disabled (DiT-only mode)
#   - >6GB VRAM: LLM enabled
# Values: auto (default), true (force enable), false (force disable)
ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"
# ACESTEP_INIT_LLM="true"
# ACESTEP_INIT_LLM="false"

# LM model path (optional, only used when LLM is enabled)
# Available models: acestep-5Hz-lm-0.6B, acestep-5Hz-lm-1.7B, acestep-5Hz-lm-4B
LM_MODEL_PATH="${LM_MODEL_PATH:-}"
# LM_MODEL_PATH="--lm-model-path acestep-5Hz-lm-4B"

# Update check on startup (set to "false" to disable)
: "${CHECK_UPDATE:=true}"

# Skip model loading at startup (models will be lazy-loaded on first request)
# Set to true to start server quickly without loading models
ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-false}"
# ACESTEP_NO_INIT="true"

# ==================== Venv Configuration ====================
VENV_DIR="${SCRIPT_DIR}/venv_xpu"

# ==================== Launch ====================

# ==================== Startup Update Check ====================
_startup_update_check() {
    [[ "$CHECK_UPDATE" != "true" ]] && return 0
    command -v git &>/dev/null || return 0
    cd "$SCRIPT_DIR" || return 0
    git rev-parse --git-dir &>/dev/null 2>&1 || return 0

    local branch commit remote_commit
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
    commit="$(git rev-parse --short HEAD 2>/dev/null || echo "")"
    [[ -z "$commit" ]] && return 0

    echo "[Update] Checking for updates..."

    local fetch_ok=0
    if command -v timeout &>/dev/null; then
        timeout 10 git fetch origin --quiet 2>/dev/null && fetch_ok=1
    else
        git fetch origin --quiet 2>/dev/null && fetch_ok=1
    fi

    if [[ $fetch_ok -eq 0 ]]; then
        echo "[Update] Network unreachable, skipping."
        echo ""
        return 0
    fi

    remote_commit="$(git rev-parse --short "origin/$branch" 2>/dev/null || echo "")"

    if [[ -z "$remote_commit" || "$commit" == "$remote_commit" ]]; then
        echo "[Update] Already up to date ($commit)."
        echo ""
        return 0
    fi

    echo ""
    echo "========================================"
    echo "  Update available!"
    echo "========================================"
    echo "  Current: $commit  ->  Latest: $remote_commit"
    echo ""
    echo "  Recent changes:"
    git --no-pager log --oneline "HEAD..origin/$branch" 2>/dev/null | head -10
    echo ""

    read -rp "Update now before starting? (Y/N): " update_choice
    if [[ "${update_choice^^}" == "Y" ]]; then
        if [[ -f "$SCRIPT_DIR/check_update.sh" ]]; then
            bash "$SCRIPT_DIR/check_update.sh"
        else
            echo "Pulling latest changes..."
            git pull --ff-only origin "$branch" 2>/dev/null || {
                echo "[Update] Update failed. Please run: git pull"
            }
        fi
    else
        echo "[Update] Skipped. Run ./check_update.sh to update later."
    fi
    echo ""
}

_startup_update_check

echo "============================================"
echo "  ACE-Step 1.5 API - Intel XPU Edition (Linux)"
echo "============================================"
echo ""

# Activate venv if it exists
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    echo "Activating XPU virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "========================================"
    echo " ERROR: venv_xpu not found!"
    echo "========================================"
    echo ""
    echo "Please create the XPU virtual environment first:"
    echo ""
    echo "  1. Run: python3 -m venv venv_xpu"
    echo "  2. Run: source venv_xpu/bin/activate"
    echo "  3. Run: pip install -r requirements-xpu.txt"
    echo ""
    echo "Or use the setup script:"
    echo "  ./setup_xpu.sh"
    echo ""
    exit 1
fi
echo ""

# Verify XPU PyTorch is installed
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
echo ""

echo "Starting ACE-Step REST API Server..."
echo "API will be available at: http://${HOST}:${PORT}"
echo "API Documentation: http://${HOST}:${PORT}/docs"
echo ""

# Build command with optional parameters
CMD="--host $HOST --port $PORT"
[[ -n "$API_KEY" ]] && CMD="$CMD $API_KEY"
[[ -n "$DOWNLOAD_SOURCE" ]] && CMD="$CMD $DOWNLOAD_SOURCE"
[[ -n "$LM_MODEL_PATH" ]] && CMD="$CMD $LM_MODEL_PATH"

# Handle ACESTEP_INIT_LLM
if [[ -n "$ACESTEP_INIT_LLM" && "$ACESTEP_INIT_LLM" != "auto" ]]; then
    CMD="$CMD --init-llm $ACESTEP_INIT_LLM"
fi

# Run in background so we can handle signals ourselves
cd "$SCRIPT_DIR" && python3 -u acestep/api_server.py $CMD &
PID=$!

# Forward Ctrl+C as SIGTERM (bypasses Gradio's SIGINT handler for graceful shutdown)
trap 'echo ""; echo "Shutting down..."; kill -TERM $PID 2>/dev/null; wait $PID; exit 0' SIGINT SIGTERM

# Wait for process to finish
wait $PID
