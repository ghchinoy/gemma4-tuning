# Gemmmma Models with MLX and llama.cpp

This repository contains an end-to-end pipeline (`mlxtune`) for downloading, preparing, fine-tuning, and exporting Gemma models locally on Apple Silicon using MLX. The final output is a highly optimized GGUF file ready for use in Go applications or others via `llama.cpp`.

## Prerequisites

You need the `uv` Python package manager, the `hf` (Hugging Face) CLI, and `llama.cpp` installed on your system.

```bash
brew install huggingface-cli
brew install llama.cpp
# Ensure uv is installed: curl -LsSf https://astral.sh/uv/install.sh | sh
```

## The 8-Step `mlxtune` Pipeline

The `mlxtune` script provides an "easy button" CLI for the entire fine-tuning lifecycle. Because this is an installable `uv` project, you can run the command natively.

### 1. Download the Base Model
Downloads a quantized MLX model from Hugging Face to act as your base.
```bash
uv run mlxtune download --model "mlx-community/gemma-2-2b-it-4bit"
```

### 2 & 3. Prepare the Dataset
Downloads a dataset, shuffles it, takes a sample subset, and formats it into the ChatML JSONL format expected by MLX.
```bash
uv run mlxtune prep --dataset "yahma/alpaca-cleaned" --samples 1000
```

### 4. Train the Model (LoRA)
Runs Low-Rank Adaptation (LoRA) fine-tuning on your dataset. This freezes the base model and only trains a tiny adapter, making it extremely memory efficient.
```bash
uv run mlxtune train --iters 200 --batch-size 2
```

### 5. Evaluate the Adapter
Tests the trained adapter by applying it to the base model on the fly and generating a response.
```bash
uv run mlxtune eval --prompt "Write a short poem about MLX."
```

### 6. Fuse the Model
Bakes the trained LoRA adapter weights permanently into the base model. This step *dequantizes* the MLX model back to 16-bit so that it can be cleanly converted to GGUF later.
```bash
uv run mlxtune fuse
```

### 7. Export to GGUF
Packages the fused weights, the tokenizer, and chat templates into a single, quantized `.gguf` file using `llama.cpp` conversion scripts.
```bash
uv run mlxtune gguf --outtype q8_0
```

### 8. Clean Artifacts
Quickly cleans out temporary training directories (like `adapters/` and `fused_model_dequantized/`) so you can run fresh experiments.
```bash
uv run mlxtune clean
```

### 9. Multi-Format Mobile Export
Compiles and quantizes your fine-tuned model into mobile-optimized runtimes (LiteRT-LM and native MLX formats) for completely offline sandboxed app execution.
```bash
uv run scripts/export_formats.py --help
```

## MLXMonitor (Live Visual Dashboard)

This repository also includes a native macOS Swift application called **MLXMonitor** that provides a real-time, interactive dashboard for your training runs. 

<img width="1012" height="644" alt="Image" src="https://github.com/user-attachments/assets/490736f0-f1f9-4fa1-992f-f0798eb2cb66" />

As `mlxtune train` runs in your terminal, it emits metrics to a `training_log.jsonl` file. You can open MLXMonitor to see a live-updating chart of your Train and Validation loss, allowing you to instantly identify the exact iteration where your model hits its "sweet spot" before overfitting.

To run the monitor (in a separate terminal tab):
```bash
cd MLXMonitor
swift run
```
*See [MLXMonitor/README.md](MLXMonitor/README.md) for more details.*

## Serving the Model

Once exported, you can serve your custom-trained model via the `llama-server`. This spins up a local HTTP server that mimics the OpenAI API, perfect for Go apps.

```bash
llama-server -m my-custom-model.gguf -c 4096 --port 8080
```

## Reference Documentation & Tradeoffs

To ensure your fine-tuning pipeline matches your target deployment architectures, consult our specialized guides:
*   **Detailed Lifecycle Explanations:** See [`docs/PIPELINE.md`](docs/PIPELINE.md) for data selection, quantization, and cloud scale rules.
*   **On-Device Mobile Deployments:** See [`docs/MOBILE_EXPORT.md`](docs/MOBILE_EXPORT.md) for our premium comparative analysis, memory footprints, battery constraints, and a complete **Decision Tree** for choosing GGUF vs. LiteRT-LM vs. MLX Swift.
*   **Multimodal Models:** Check out [`docs/MULTIMODAL_PIPELINE.md`](docs/MULTIMODAL_PIPELINE.md) for Gemma 4 audio/vision training setups.