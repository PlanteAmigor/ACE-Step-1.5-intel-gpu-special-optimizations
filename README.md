# ACE-Step 1.5 — Intel GPU Optimized Edition

> **Upstream Repository**: [https://github.com/ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)  
> **This Fork**: [https://github.com/PlanteAmigor/ACE-Step-1.5-intel-gpu-special-optimizations](https://github.com/PlanteAmigor/ACE-Step-1.5-intel-gpu-special-optimizations)  
> **Test Platform**: Intel Core Ultra 9 285H + Arc iGPU (8 Xe-core, 115.2 GB shared memory)  
> **OS**: Ubuntu 26.04 LTS | **PyTorch**: 2.12.0+xpu | **Python**: 3.14  
> **中文版**: [READMEzh-CN.md](https://github.com/PlanteAmigor/ACE-Step-1.5-intel-gpu-special-optimizations/blob/main/READMEzh-CN.md)

---

## Overview

This repository is a deeply optimized fork of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), specifically tailored for **Intel Arc GPU / integrated graphics + Linux XPU** platforms.

ACE-Step 1.5 is primarily designed for NVIDIA CUDA users. The Intel XPU software stack (Level Zero driver, PyTorch XPU backend, SYCL runtime) is relatively new and lacks many protection mechanisms that CUDA has accumulated over 15+ years — such as TDR watchdog, OOM retry chain, expandable segments, and async error reporting.

This fork compensates for these missing protections through **code-level active safeguards**, enabling Intel Arc iGPU to reliably generate up to **600 seconds (10 minutes)** of continuous audio.

> ⚠️ **Currently, only a Linux quick-start script (`start_gradio_ui_xpu.sh`) is provided. No Windows (.bat) version is available.**  
> Windows users can refer to the script's environment variables and configuration for manual setup.
>
> **Note for Windows users**: Windows driver achieves 11–12 tokens/s (vs. 4–5 on Linux), meaning memory fragmentation accumulates ~2.5× faster and instantaneous power draw is significantly higher. Windows may require **more aggressive cooling** than the settings in this fork. See [detailed comparison](https://github.com/PlanteAmigor/intel-gpu-stability-guide/blob/main/Intel-gpu-Ace%20-Step-Test-report.md) for platform-specific test results.

---

## Key Differences from Upstream

### 1. 🧊 Tiered GPU Cooling System

Intel XPU lacks NVIDIA's TDR (Timeout Detection & Recovery) watchdog. Without it, sustained GPU load can lead to an inconsistent driver state, producing NaN or crashes. This fork adds active cooling at all three major computation stages:

| Stage | Cooling Frequency | Rest Duration | Throttling Detection |
|-------|------------------|---------------|---------------------|
| **LLM Inference** (both generation paths) | Every 100 tokens (normal) / 300 tokens (deep) | 5 sec / 30 sec | ✅ 2× median → extra 10 sec |
| **DiT Diffusion** (all 6 model variants) | Every 2 steps | 3 sec | ❌ |
| **VAE Decode** (GPU + CPU Offload) | Every 50 chunks | 5 sec | ✅ 2× median → extra 10 sec |

Each cooling cycle performs:
- `gc.collect()` — Python garbage collection
- `torch.xpu.empty_cache()` — Release XPU cached memory to reduce fragmentation

### 2. 🧠 Memory Fragmentation Monitoring

XPU's caching allocator lacks CUDA's expandable segments and complex OOM retry chain. This fork introduces the `_monitor_mem_frag()` function that:

- Checks `inactive_split_bytes` every 50 LLM steps
- Logs a warning when fragmentation ratio exceeds 10%
- Triggers automatic `empty_cache()` when fragmentation exceeds 30%

### 3. 🛡️ OOM Exception Catch Fix

Upstream code catches only `torch.cuda.OutOfMemoryError` in the VAE decode fallback chain. On XPU, OOM manifests as a `RuntimeError` with Level Zero error codes, making the fallback chain completely non-functional.

This fork fixes the catch to support both CUDA and XPU:
```python
_OOM_ERRORS = (torch.cuda.OutOfMemoryError, RuntimeError)
```

Now the fallback chain works correctly on XPU: GPU decode → CPU offload decode → full CPU VAE decode.

### 4. 🔄 Model Offload Protection

The `model.to("cpu")` operation in `_load_model_context()` can trigger an OOM on XPU due to memory fragmentation. This fork:

- Runs `empty_cache()` + `gc.collect()` **before** `to("cpu")`
- Wraps `to("cpu")` in `try/except` to prevent the process from hanging on cleanup failures

### 5. 🚀 XPU-Specific Launch Script

`start_gradio_ui_xpu.sh` introduces:

```bash
export SYCL_CACHE_PERSISTENT=1          # Persist SYCL kernel cache
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # Lower latency
export PYTORCH_DEVICE=xpu               # Force XPU backend
export TORCH_COMPILE_BACKEND=eager      # torch.compile is incomplete on XPU
export TORCHAUDIO_USE_BACKEND=ffmpeg    # Avoid torchcodec on XPU
```

Additional features:
- Auto-detects and releases occupied ports (`fuser -k`)
- Verifies PyTorch XPU availability before launch
- Enforces UTF-8 encoding

---

## Known Issues

### Attention Backward Pass Crash

When training Transformer models on Intel XPU (including `nn.TransformerEncoderLayer`, `F.scaled_dot_product_attention`), the backward pass crashes under certain configurations:

```
RuntimeError: Trying to create tensor with negative dimension -79243236477491020
```

This is an integer overflow bug in Intel's XPU `mha_bwd` SYCL kernel — likely a stride/padding calculation that overflows for specific tensor sizes. **This only affects training, not inference** (inference uses forward pass only).

Fixing this requires compiling the `intel/torch-xpu-ops` repository from source with the Intel oneAPI DPC++ compiler (which is already installed on this system).

**Detailed analysis**: See `xpu_backward_issue.md`

---

## Quick Start (Linux)

```bash
# 1. Create virtual environment
python3 -m venv venv_xpu
source venv_xpu/bin/activate

# 2. Install PyTorch XPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 3. Install dependencies
pip install -r requirements-xpu.txt

# 4. Launch (auto port kill + XPU detection)
./start_gradio_ui_xpu.sh
```

---

## Test Results

| Duration | Result | Notes |
|----------|--------|-------|
| 360s (6 min) | ✅ Success | Clean output |
| 480s (8 min) | ✅ Success | Clean output |
| **600s (10 min)** | ✅ **Success** | **Upper limit with active cooling** |
| 600s (no cooling) | ❌ NaN | Previous failure cause |

**Key Insight**: Long-audio NaN is **NOT a model limitation or a hardware defect**. It is caused by Intel's immature XPU software stack missing CUDA-grade protection mechanisms. The active cooling strategy in this fork effectively compensates for these missing safeguards.

---

## File Change Log

| File | Changes |
|------|---------|
| `start_gradio_ui_xpu.sh` | **New** — XPU-specific launch script |
| `acestep/llm_inference.py` | Added `_monitor_mem_frag()` + LLM cooling + model offload protection |
| `acestep/core/generation/handler/vae_decode_chunks.py` | VAE decode cooling + OOM exception fix |
| `acestep/models/*/modeling_acestep_v15_*.py` (6 files) | DiT diffusion cooling (every 2 steps, 3 sec) |
