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
*   **Standard Run:**
    ```bash
    uv run mlxtune train --iters 200 --batch-size 2
    ```
*   **Gemma 4 QAT Run (Enables conservative adapter configurations):**
    ```bash
    uv run mlxtune train --model "~/projects/gemma/google-gemma-4-E2B-it-qat-q4_0-unquantized" --qat --rank 8 --lora-layers 16
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
Exports to a single quantized `.gguf` via mlx-tune's `export_to_gguf` (llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`).
Uses the `llama.cpp/` checkout in this repo (or `$LLAMA_CPP_PATH` / `--llama-cpp`) and its own converter venv; one-time setup:
```bash
uv venv --python 3.12 llama.cpp/.venv
uv pip install --python llama.cpp/.venv/bin/python \
    -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --index-strategy unsafe-best-match
```
*   **Standard Export:**
    ```bash
    uv run mlxtune gguf --outtype q8_0
    ```
*   **Gemma 4 QAT Export (Strictly enforces `q4_0` to match QAT pre-conditioned parameters):**
    ```bash
    uv run mlxtune gguf --qat
    ```
*   **Straight from a base model + adapter (no separate fuse step):**
    ```bash
    uv run mlxtune gguf --model <base> --adapter ./adapters --qat
    ```

### 8. Clean Artifacts
Quickly cleans out temporary training directories (like `adapters/` and `fused_model_dequantized/`) so you can run fresh experiments.
```bash
uv run mlxtune clean
```

### 9. Benchmark Quantization Drift & Perplexity
Programmatically evaluates the semantic "drift" introduced by compressing your model, comparing the high-precision reference outputs directly to the low-precision quantized GGUF outputs.
```bash
uv run mlxtune benchmark --reference-model "./fused_model_dequantized" --gguf-model "my-custom-model.gguf"
```
*Note: This command will automatically run Jaccard vocabulary similarity matching and output a complete tutorial on executing mathematical validation via `llama-perplexity`.*

### 10. Multi-Format Mobile Export
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

## 🤖 Agent Purposes, Capabilities & Specialized Guides

To maximize **didactic ease of use** and support automated AI coding assistants, the repository houses both comprehensive markdown documentation and structured AI "Agent Skills" (conforming to the [agentskills.io specification](https://agentskills.io/specification.md)). 

Here is how you can leverage and map these resources based on your specific engineering or training goals:

### 1. Standard LoRA Fine-Tuning
*   **Purpose:** Fine-tuning base/instruction-tuned text models on conversational prompt-response data.
*   **Detailed Guide:** [`docs/PIPELINE.md`](docs/PIPELINE.md) (covers dataset philosophies like LIMA vs. Alpaca, and basic LoRA parameters).
*   **Agent Skill:** [`skills/gemma-fine-tuning/SKILL.md`](skills/gemma-fine-tuning/SKILL.md) — instructions for agents to prepare, train, monitor, and clean standard runs.

### 2. Quantization-Aware Training (QAT) Alignment
*   **Purpose:** Compiling high-fidelity models for edge/mobile devices using pre-conditioned QAT base models, avoiding the semantic accuracy drops introduced by Post-Training Quantization (PTQ).
*   **Detailed Guide:** [`docs/PIPELINE.md#gemma-4-qat-quantization-aware-training-alignment`](docs/PIPELINE.md#L26) (concepts of QAT unquantized checks, strict `q4_0` GGUF alignment, and rank conservatism).
*   **Agent Skill:** [`skills/gemma-qat-tuning/SKILL.md`](skills/gemma-qat-tuning/SKILL.md) — instructions for executing clean QAT runs with stable parameters, strict `q4_0` quantization, and drift benchmarking.

### 3. Natively Multimodal Training (Vision & Audio)
*   **Purpose:** Training models to receive interleaved text, raw audio, and image arrays in the new ChatML standard.
*   **Detailed Guide:** [`docs/MULTIMODAL_PIPELINE.md`](docs/MULTIMODAL_PIPELINE.md) (interleaved ChatML format, `mlx-vlm` library integration, and audio conformer module targeting).
*   **Agent Skill:** [`skills/gemma-multimodal-tuning/SKILL.md`](skills/gemma-multimodal-tuning/SKILL.md) — instructions for formatting content arrays, launching `--multimodal` training, and executing key-sanitized weights fusion.

### 4. On-Device Mobile Compilation & Runtimes
*   **Purpose:** Merging, converting, and packaging model checkpoints into sandboxed mobile apps (iOS & Android).
*   **Detailed Guide:** [`docs/MOBILE_EXPORT.md`](docs/MOBILE_EXPORT.md) (performance matrices, battery/RAM tradeoffs, storage guidelines, and the **Mobile Export Decision Tree**).
*   **Agent Skill:** [`skills/gemma-model-export/SKILL.md`](skills/gemma-model-export/SKILL.md) — instructions for compiling GGUF, LiteRT-LM (`.litertlm` package), or native MLX formats, and implementing safe sandbox dynamic-download onboarding flows.

---

## Servicing and Tracking Tools

*   **Experiment Journal:** [`docs/EXPERIMENT_JOURNAL.md`](docs/EXPERIMENT_JOURNAL.md) — track hyperparameter configurations, GPU memory profiles, validation losses, and quantization drift metrics.
*   **Upstream Friction Log:** [`docs/FRICTION_LOG.md`](docs/FRICTION_LOG.md) — check active workarounds for upstream bugs (e.g. Apple Metal uncatchable OOMs or MLX key-prefix mangling).
*   **Real-time Monitoring:** Check out [`MLXMonitor/README.md`](MLXMonitor/README.md) to launch the native Swift charts dashboard for live training loss.