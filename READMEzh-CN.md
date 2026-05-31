# ACE-Step 1.5 — Intel GPU 专用优化版

> **上游仓库**: [https://github.com/ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)  
> **本 Fork**: [https://github.com/PlanteAmigor/ACE-Step-1.5-intel-gpu-special-optimizations](https://github.com/PlanteAmigor/ACE-Step-1.5-intel-gpu-special-optimizations)  
> **硬件平台**: Intel Core Ultra 9 285H + Arc iGPU (8 Xe-core, 115.2 GB 共享内存)  
> **系统**: Ubuntu 26.04 LTS | **PyTorch**: 2.12.0+xpu | **Python**: 3.14  
> **English**: [README.md](https://github.com/PlanteAmigor/ACE-Step-1.5-intel-gpu-special-optimizations/blob/main/README.md)

---

## 概述

本仓库是 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) 的深度优化分支，专门针对 **Intel Arc GPU / 集成显卡 + Linux XPU** 平台。

ACE-Step 1.5 原生主要面向 NVIDIA CUDA 用户。Intel XPU 软件栈（Level Zero 驱动、PyTorch XPU 后端、SYCL 运行时）由于起步较晚，缺少 CUDA 十几年积累的硬件保护机制（如 TDR 看门狗、OOM 重试链、expandable segments、异步错误报告等）。

本分支通过**代码层面的主动保护措施**来补偿这些缺失，使 Intel Arc iGPU 能够稳定完成长达 **600 秒（10 分钟）** 的连续音频生成。

> 本仓库仅提供**一种优化思路**，偏向安全而非速度，不代表最优策略。
> 所有代码修改由 AI 完成，在作者设备上成功运行。**在您的设备上运行之前，请务必进行审查。**

> ⚠️ **目前仅提供 Linux 快速启动脚本 (`start_gradio_ui_xpu.sh`)，暂无 Windows (.bat) 版本。**  
> Windows 用户可参考脚本中的环境变量和配置手动设置。
>
> **Windows 用户注意**: Windows 驱动下生成速度可达 11–12 tokens/s（Linux 仅 4–5），意味着内存碎片累积速度约快 2.5 倍，瞬时功耗也显著更高。Windows 可能需要比本分支更**激进的冷却策略**。详细平台对比见 [Intel-gpu-Ace-Step-Test-report](https://github.com/PlanteAmigor/intel-gpu-stability-guide/blob/main/Intel-gpu-Ace%20-Step-Test-report.md)。

---

## 与上游的主要差异

### 1. 🧊 分层 GPU 冷却系统

Intel XPU 缺乏 NVIDIA TDR（Timeout Detection & Recovery）看门狗机制。当 GPU 长时间满载运行时，可能进入不一致状态导致 NaN 或崩溃。本分支在 ACE-Step 的三个主要计算阶段均加入了主动冷却：

| 阶段 | 冷却频率 | 休息时间 | 降频检测 |
|------|---------|---------|---------|
| **LLM 推理**（两个生成路径） | 每 100 token（常规）/ 300 token（深度） | 5 秒 / 30 秒 | ✅ 超中位数 2 倍 → 额外 10 秒 |
| **DiT 扩散**（全部 6 个模型） | 每 2 步 | 3 秒 | ❌ |
| **VAE 解码**（GPU + CPU Offload） | 每 50 块 | 5 秒 | ✅ 超中位数 2 倍 → 额外 10 秒 |

冷却期间自动执行：
- `gc.collect()` — Python 垃圾回收
- `torch.xpu.empty_cache()` — 释放 XPU 缓存，减少碎片

### 2. 🧠 内存碎片监控

XPU 的缓存分配器缺少 CUDA 的 expandable segments 和复杂 OOM 重试链。本分支新增了 `_monitor_mem_frag()` 函数：

- **每 50 步 LLM 生成时**检查 `inactive_split_bytes`
- 碎片率 > 10% 时记录日志
- 碎片率 > 30% 时主动触发 `empty_cache()` 整理

### 3. 🛡️ OOM 异常捕获修复

上游代码中 VAE 解码的 fallback 链只捕获 `torch.cuda.OutOfMemoryError`，但在 XPU 上 OOM 表现为 `RuntimeError`（Level Zero 错误码），导致 fallback 链完全失效。

本分支将捕获改为同时兼容 CUDA 和 XPU：
```python
_OOM_ERRORS = (torch.cuda.OutOfMemoryError, RuntimeError)
```

### 4. 🔄 模型卸载保护

`_load_model_context()` 中的 `model.to("cpu")` 操作在 XPU 上可能因显存碎片触发 OOM。本分支：

- 在 `to("cpu")` 之前先执行 `empty_cache()` + `gc.collect()`
- 用 `try/except` 包裹 `to("cpu")`，防止卸载阶段崩溃导致进程无法退出

### 5. 🚀 XPU 专用启动脚本

`start_gradio_ui_xpu.sh` 新增了：

```bash
export SYCL_CACHE_PERSISTENT=1          # SYCL 内核缓存持久化
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # 减少延迟
export PYTORCH_DEVICE=xpu               # 强制 XPU 后端
export TORCH_COMPILE_BACKEND=eager      # XPU 上 torch.compile 不完整
export TORCHAUDIO_USE_BACKEND=ffmpeg    # 避免 torchcodec
```

- 自动检测端口占用并释放 (`fuser -k`)
- 验证 PyTorch XPU 是否可用
- UTF-8 编码强制

---

## 已知问题

### 注意力反向传播崩溃

在 Intel XPU 上训练 Transformer 模型时（包括 `nn.TransformerEncoderLayer`、`F.scaled_dot_product_attention`），反向传播在特定配置下会崩溃：

```
RuntimeError: Trying to create tensor with negative dimension -79243236477491020
```

这是 Intel XPU 后端 `mha_bwd` SYCL kernel 中的整型溢出 bug，**与推理无关**（推理只有前向传播）。修复需要编译 `intel/torch-xpu-ops` 仓库的 C++/SYCL 源码。

**详细分析**: 参见 `xpu_backward_issue.md`

---

## 快速开始（Linux）

```bash
# 1. 创建虚拟环境
python3 -m venv venv_xpu
source venv_xpu/bin/activate

# 2. 安装 PyTorch XPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 3. 安装依赖
pip install -r requirements-xpu.txt

# 4. 启动（自动处理端口和 XPU 检测）
./start_gradio_ui_xpu.sh
```

---

## 测试结果

| 测试时长 | 结果 | 说明 |
|---------|------|------|
| 360s（6 分钟） | ✅ 成功 | 无异常 |
| 480s（8 分钟） | ✅ 成功 | 无异常 |
| **600s（10 分钟）** | ✅ **成功** | **有冷却策略下的上限** |
| 600s（无冷却） | ❌ NaN | 之前失败的原因 |

**核心结论**: 长音频 NaN **不是模型限制，也不是硬件缺陷**，而是 Intel XPU 软件栈缺少 CUDA 的保护机制。本分支的主动冷却策略有效补偿了这些缺失。

---

## 文件变更清单

| 文件 | 变更内容 |
|------|---------|
| `start_gradio_ui_xpu.sh` | **新增** — XPU 专用启动脚本 |
| `acestep/llm_inference.py` | 新增 `_monitor_mem_frag()` + LLM 冷却 + 模型卸载保护 |
| `acestep/core/generation/handler/vae_decode_chunks.py` | VAE 解码冷却 + OOM 异常修复 |
| `acestep/models/*/modeling_acestep_v15_*.py`（6 个文件） | DiT 扩散冷却（每 2 步 3 秒） |
