# Scripts Directory

This directory contains utility scripts for interacting with models and testing the MLX fine-tuning pipeline.

## Audio Testing (`ask_audio.py`)

This script is designed to test the native multimodal audio capabilities of Gemma 4 (or other multimodal models) by executing them directly on the Mac's GPU using Apple's official `mlx-vlm` Python library. 

By using `mlx-vlm`, we bypass `llama.cpp` entirely, which allows us to test cutting-edge multimodal architectures before the C++ ecosystem catches up.

### Obtaining and Converting Models for MLX

To run inference natively using Apple Silicon unified memory, you need models in MLX format (rather than GGUF which is for `llama.cpp`). If you want to download and convert the official uncompressed Google models yourself instead of relying on community conversions:

#### 1. Download the Official Model
You must accept the license agreement on Hugging Face first and ensure you are logged in (`hf auth login`). Then, download the raw PyTorch Safetensors:

```bash
# E2B Model (2 Billion Parameters)
hf download google/gemma-4-e2b-it --local-dir ~/projects/gemma/google-gemma-4-e2b

# E4B Model (4 Billion Parameters)
hf download google/gemma-4-e4b-it --local-dir ~/projects/gemma/google-gemma-4-e4b
```

#### 2. Convert PyTorch to MLX
The downloaded models contain standard PyTorch tensor layouts. You must convert them to MLX format to run optimally on the Mac GPU. We use the `mlx_vlm.convert` tool to do this.

*(Ensure your previous `hf` downloads have finished completely before running this).*

```bash
# Convert the downloaded E2B model
uv run --with mlx-vlm -m mlx_vlm.convert \
  --hf-path ~/projects/gemma/google-gemma-4-e2b \
  --mlx-path ~/projects/gemma/mlx-gemma-4-e2b
```
*(Optional: You can add `-q` to the conversion command to quantize the model to 4-bit, saving RAM at the cost of a slight drop in accuracy).*

### How to Use the Script

The script uses `uv` to manage its dependencies (like the `mlx-vlm` library) automatically via inline script metadata. You don't need to manually install anything.

From the root of the project, run:

```bash
# Basic usage (defaults to looking in the ./model directory)
uv run scripts/ask_audio.py /path/to/your/audio.wav

# Specify a custom prompt
uv run scripts/ask_audio.py /path/to/your/audio.wav --prompt "Translate this audio to Spanish."

# Run with your newly converted local model
uv run scripts/ask_audio.py /path/to/your/audio.wav --model ~/projects/gemma/mlx-gemma-4-e2b

# Or specify a Hugging Face MLX repo (downloads automatically)
uv run scripts/ask_audio.py /path/to/your/audio.wav --model mlx-community/gemma-4-E2B-it-4bit
```

### Interpreting the Results
*   **Success:** The script loads the `.safetensors` files natively, processes the audio waveform, and generates a textual response describing the audio on your GPU.
*   **Failure:** If the script throws an error during loading or generation, it usually means the model directory is missing the audio projector weights, or you need to update to the latest version of `mlx-vlm` to get the newest Gemma 4 audio tensor mappings.

---

## Multi-Format Mobile & Desktop Export (`export_formats.py`)

This utility provides the tuning team with an automated compiler suite to export fine-tuned weights into three standard runtime targets:

1.  **GGUF (`gguf`):** For desktop companion servers like `llama-server`.
2.  **LiteRT-LM (`litert`):** For highly optimized, dynamically quantized 8-bit flatbuffers (`.litertlm` model package) optimized for sandboxed offline mobile execution on iOS (Swift) and Android (Kotlin).
3.  **MLX (`mlx`):** For native, unified memory swift environments on macOS/iOS.

### How to Use the Script

The script runs in the gemmma project environment (Python 3.12, `uv run` from the repo root). GGUF and MLX use
the same mlx-tune code paths as `mlxtune gguf` / `mlxtune fuse`. GGUF needs the `llama.cpp/.venv` converter
environment (main README, step 7).

```bash
# Display help and usage options
uv run scripts/export_formats.py --help

# 1. Compile for GGUF (Desktop & llama-server); add --qat for Gemma QAT checkpoints (strict q4_0)
uv run scripts/export_formats.py gguf \
  --base ./model \
  --adapter ./adapters \
  --dest my_custom_model.gguf \
  --outtype q4_k_m

# 2. Compile for LiteRT-LM (iOS & Android offline flatbuffers). litert-torch is not a project
# dependency, so add it for this run. Point --prefused at the output of `mlxtune fuse` to skip
# on-the-fly PEFT merging (saves over 5GB of RAM). E2B/E4B only: 12B needs >90GB RAM (FL-004).
uv run --with litert-torch scripts/export_formats.py litert \
  --base ./model \
  --adapter ./adapters \
  --dest ./models/litert \
  --prefused ./fused_model_dequantized

# 3. Compile for Standard MLX (Apple Silicon Swift pipelines)
uv run scripts/export_formats.py mlx \
  --base ./model \
  --adapter ./adapters \
  --dest ./models/mlx_model
```

For a comprehensive guide comparing file sizes, memory footprints, processing bottlenecks, and battery characteristics across these targets, see [`docs/MOBILE_EXPORT.md`](../docs/MOBILE_EXPORT.md).

