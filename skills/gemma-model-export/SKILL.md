---
name: gemma-model-export
description: Guides agents on exporting fine-tuned Gemma models to multi-format mobile runtimes, comparing memory and hardware target characteristics, and implementing dynamic sandbox download workflows.
compatibility: Requires Python 3.14+, uv, Apple Silicon, and Xcode/Swift compiler
---

# Gemma 4 Multi-Format Mobile Export Skill

This skill provides step-by-step instructions for AI coding assistants and agents to compile, merge, and quantize fine-tuned Gemma models into highly optimized mobile runtimes (GGUF, LiteRT-LM, MLX Swift, and CoreML).

## 🧭 Step-by-Step Workflow

### 1. Identify Your Target Runtime

Before compilation, determine the best on-device runtime using the following decision patterns:

*   **GGUF via `llama.cpp`:** Cross-platform (iOS, Android, Desktop). Supports hot-swapping multiple adapters. Runs on CPU/GPU.
*   **LiteRT-LM (Google):** Cross-platform (iOS, Android). Extreme battery efficiency and low RAM footprint by leveraging NPUs and Vulkan. Adapters must be pre-merged.
*   **MLX Swift:** High-performance, iOS/macOS strictly. Directly accesses Apple Unified Memory on modern A17 Pro or M-series SOCs.
*   **CoreML (Apple Native):** iOS-exclusive. Compiled to run entirely on Apple's dedicated Neural Engine (ANE) for near-zero battery drain. Complex intermediate Graph conversion.

---

### 2. Run Multi-Format Mobile Compiler

The repository includes a premium multi-format model compiler script `scripts/export_formats.py` that streamlines weight pipelines:

#### Option A: GGUF Target
Fuse the base model and adapters dequantizing to 16-bit, copy vocab files, and quantize to GGUF in one command:
```bash
uv run scripts/export_formats.py gguf \
  --base ./model \
  --adapter ./adapters \
  --dest ./my_model.gguf \
  --outtype q4_k_m
```

#### Option B: LiteRT-LM Target (.litertlm flatbuffer)
Natively merges the LoRA adapter weights inside PyTorch and compiles them using `litert-torch` with highly optimized dynamic 8-bit quantization (`dynamic_wi8_afp32`):
```bash
uv run scripts/export_formats.py litert \
  --base ./model \
  --adapter ./adapters \
  --dest ./models/litert
```
*   **Performance Optimization:** If you already generated fused, dequantized weights using `mlxtune fuse`, pass `--prefused ./fused_model_dequantized` to bypass expensive on-the-fly PEFT reloading and save over `5GB` of RAM overhead.

#### Option C: MLX Format (for Swift iOS app loading)
Bakes adapters back and outputs a clean, MLX-compatible directory:
```bash
uv run scripts/export_formats.py mlx \
  --base ./model \
  --adapter ./adapters \
  --dest ./models/mlx_model
```

---

## 📱 Guidelines for Swift/iOS App Integration & Storage

When building local mobile applications that load these exported binaries, you must comply with strict Apple Sandbox policies:

1.  **Never bundle model files inside the App Store binary:** Adding a ~1.4GB model directly to your Xcode asset catalog will exceed cellular OTA limits and block downloads.
2.  **Dynamic Onboarding Download Pattern:**
    *   Present an onboarding wizard displaying the download size and persistent storage path.
    *   Store the downloaded model files inside the **Application Support Directory** (`FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)`). Apple preserves this directory but automatically excludes it from standard iCloud backups.
    *   Validate file integrity on boot using an MD5 or SHA256 checksum before initializing the runtime.
3.  **Graceful Offline Fallback:** Always allow the application to operate in an offline, dictionary-only, or cached-prompt mode if a download is interrupted.
