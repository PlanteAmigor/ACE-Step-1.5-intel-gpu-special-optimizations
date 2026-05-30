#!/usr/bin/env bash
# ACE-Step Gradio Web UI Launcher - Linux Intel XPU
# For Intel Arc GPUs (A770, A750, A580, A380) and integrated graphics on Linux
# Requires: Python 3.11, PyTorch XPU from https://download.pytorch.org/whl/xpu
# IMPORTANT: Uses torch.xpu backend with SYCL/Level Zero acceleration
#
# Setup:
#   1. Create venv:  python3 -m venv venv_xpu
#   2. Activate:     source venv_xpu/bin/activate
#   3. Install PyTorch for XPU:
#      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
#   4. Install dependencies:
#      pip install -r requirements-xpu.txt
#   5. Run this script: ./start_gradio_ui_xpu.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Load .env Configuration ====================
_load_env_file() {
    local env_file="${SCRIPT_DIR}/.env"
    if [[ ! -f "$env_file" ]]; then
        return 0
    fi

    echo "[Config] Loading configuration from .env file..."

    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue

        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        case "$key" in
            ACESTEP_CONFIG_PATH)
                [[ -n "$value" ]] && CONFIG_PATH="--config_path $value"
                ;;
            ACESTEP_LM_MODEL_PATH)
                [[ -n "$value" ]] && LM_MODEL_PATH="--lm_model_path $value"
                ;;
            ACESTEP_INIT_LLM)
                if [[ -n "$value" && "$value" != "auto" ]]; then
                    INIT_LLM="--init_llm $value"
                fi
                ;;
            ACESTEP_DOWNLOAD_SOURCE)
                if [[ -n "$value" && "$value" != "auto" ]]; then
                    DOWNLOAD_SOURCE="--download-source $value"
                fi
                ;;
            ACESTEP_API_KEY)
                [[ -n "$value" ]] && API_KEY="--api-key $value"
                ;;
            PORT)
                [[ -n "$value" ]] && PORT="$value"
                ;;
            SERVER_NAME)
                [[ -n "$value" ]] && SERVER_NAME="$value"
                ;;
            LANGUAGE)
                [[ -n "$value" ]] && LANGUAGE="$value"
                ;;
            ACESTEP_BATCH_SIZE)
                [[ -n "$value" ]] && BATCH_SIZE="--batch_size $value"
                ;;
            ACESTEP_OFFLOAD_TO_CPU)
                [[ -n "$value" ]] && OFFLOAD_TO_CPU="--offload_to_cpu $value"
                ;;
        esac
    done < "$env_file"

    echo "[Config] Configuration loaded from .env"
}

_load_env_file

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
: "${PORT:=7860}"
: "${SERVER_NAME:=127.0.0.1}"
# SERVER_NAME="0.0.0.0"
SHARE="${SHARE:-}"
# SHARE="--share"

# Reset LANGUAGE if it contains an invalid value (e.g. system locale like en_CA:en)
case "${LANGUAGE:-}" in
    en|zh|he|ja) ;;
    *) unset LANGUAGE ;;
esac
# UI language: en, zh, he, ja
: "${LANGUAGE:=zh}"

# Batch size: default batch size for generation (1 to GPU-dependent max)
# When not specified, defaults to min(2, GPU_max)
BATCH_SIZE="${BATCH_SIZE:-}"
BATCH_SIZE="--batch_size 1"

# ==================== Model Configuration ====================
: "${CONFIG_PATH:=--config_path acestep-v15-turbo}"
: "${LM_MODEL_PATH:=--lm_model_path acestep-5Hz-lm-1.7B}"

# CPU offload: recommended for 4B LM on GPUs with <=16GB VRAM
# Models shuttle between CPU/GPU as needed (DiT stays on GPU, LM/VAE/text_encoder move on demand)
# Adds ~8-10s overhead per generation but prevents VRAM oversubscription
# Disable if using 1.7B/0.6B LM or if your GPU has >=20GB VRAM
: "${OFFLOAD_TO_CPU:=--offload_to_cpu true}"

# LLM initialization: auto (default), true, false
INIT_LLM="${INIT_LLM:-}"
# INIT_LLM="--init_llm auto"

# Download source: auto, huggingface, modelscope
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-}"

# Auto-initialize models on startup
: "${INIT_SERVICE:=--init_service true}"

# API settings
ENABLE_API="${ENABLE_API:-}"
# ENABLE_API="--enable-api"
API_KEY="${API_KEY:-}"
# API_KEY="--api-key sk-your-secret-key"

# Authentication
AUTH_USERNAME="${AUTH_USERNAME:-}"
# AUTH_USERNAME="--auth-username admin"
AUTH_PASSWORD="${AUTH_PASSWORD:-}"
# AUTH_PASSWORD="--auth-password password"

# Update check on startup (set to "false" to disable)
: "${CHECK_UPDATE:=true}"

# ==================== Venv Configuration ====================
VENV_DIR="/run/media/amigor/Project/AItrain/ACE-Step-1.5/ACE-Step-1.5-0.1.8/venv_xpu"

# ==================== Launch ====================

# 自动释放端口（防止上次退出未清理）
fuser -k ${PORT}/tcp 2>/dev/null || true

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
echo "  ACE-Step 1.5 - Intel XPU Edition (Linux)"
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
    print(f'WARNING: Intel XPU not detected, falling back to CPU')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'MPS available: {hasattr(torch, \"mps\") and torch.mps.is_available()}')
" 2>&1; then
    echo ""
    echo "========================================"
    echo " WARNING: PyTorch XPU verification issue"
    echo "========================================"
    echo ""
    echo "The script will continue, but XPU acceleration may not be available."
    echo "If you see issues, try reinstalling PyTorch XPU."
    echo ""
fi
echo ""

echo "Starting ACE-Step Gradio Web UI..."
echo "Server will be available at: http://${SERVER_NAME}:${PORT}"
echo "Default Model: acestep-v15-turbo"
echo "LM Model: ${LM_MODEL_PATH##* }"
echo ""
echo "Select your model in the UI if needed!"
echo ""

# Build command with optional parameters
CMD="--port $PORT --server-name $SERVER_NAME --language $LANGUAGE"
[[ -n "$SHARE" ]] && CMD="$CMD $SHARE"
[[ -n "$CONFIG_PATH" ]] && CMD="$CMD $CONFIG_PATH"
[[ -n "$LM_MODEL_PATH" ]] && CMD="$CMD $LM_MODEL_PATH"
[[ -n "$OFFLOAD_TO_CPU" ]] && CMD="$CMD $OFFLOAD_TO_CPU"
[[ -n "$INIT_LLM" ]] && CMD="$CMD $INIT_LLM"
[[ -n "$DOWNLOAD_SOURCE" ]] && CMD="$CMD $DOWNLOAD_SOURCE"
[[ -n "$INIT_SERVICE" ]] && CMD="$CMD $INIT_SERVICE"
[[ -n "$BATCH_SIZE" ]] && CMD="$CMD $BATCH_SIZE"
[[ -n "$ENABLE_API" ]] && CMD="$CMD $ENABLE_API"
[[ -n "$API_KEY" ]] && CMD="$CMD $API_KEY"
[[ -n "$AUTH_USERNAME" ]] && CMD="$CMD $AUTH_USERNAME"
[[ -n "$AUTH_PASSWORD" ]] && CMD="$CMD $AUTH_PASSWORD"

# Run in background so we can handle signals ourselves
cd "$SCRIPT_DIR" && python3 -u acestep/acestep_v15_pipeline.py $CMD &
PID=$!

# Forward Ctrl+C as SIGTERM (bypasses Gradio's SIGINT handler for graceful shutdown)
trap 'echo ""; echo "Shutting down..."; kill -TERM $PID 2>/dev/null; wait $PID; exit 0' SIGINT SIGTERM

# Wait for process to finish
wait $PID
