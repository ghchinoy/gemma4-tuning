---
name: gemma-fine-tuning
description: Guides agents on preparing datasets, running parameter-efficient LoRA fine-tuning, and tracking telemetry metrics using the mlxtune CLI on Apple Silicon. Use when an agent wants to fine-tune standard Gemma 4 base models on text instructions.
compatibility: Requires Python 3.14+, uv, and Apple Silicon Mac
---

# Gemma 4 LoRA Fine-Tuning Skill

This skill provides step-by-step instructions for AI coding assistants and agents to fine-tune standard Gemma 4 models on text datasets using the local `mlxtune` tool.

## 🧭 Step-by-Step Workflow

### 1. Download the Base Model
Download a high-precision or pre-quantized base model from Hugging Face:
```bash
uv run mlxtune download --model mlx-community/gemma-2-2b-it-4bit --dest ./model
```

### 2. Prepare the Training Dataset
Convert custom instruction-response pairs from Hugging Face or local data into the standard ChatML JSONL format:
```bash
uv run mlxtune prep --dataset yahma/alpaca-cleaned --dest ./data --samples 1000
```
This splits data into `data/train.jsonl` and `data/valid.jsonl`.

### 3. Execute the LoRA Training Loop
Run the parameter-efficient LoRA fine-tuning loop on Apple Metal GPU:
```bash
uv run mlxtune train --model ./model --data ./data --iters 200 --batch-size 2 --rank 8 --lr 1e-5
```
*   **Key Parameters:**
    *   `--iters`: Total training steps.
    *   `--batch-size`: Batch size per step.
    *   `--rank`: LoRA adapter dimension (default 8).
    *   `--lr`: Learning rate (default 1e-5).

### 4. Monitor Live Telemetry
Inspect the real-time logging output written to `training_log.jsonl`. The log format emits JSON objects representing training steps:
```json
{"type": "train", "iter": 10, "loss": 2.214, "learning_rate": 1e-5, "peak_mem_gb": 10.4}
```
This log is read dynamically by the native macOS companion app `MLXMonitor`.

---

## ⚠️ Common Edge Cases & Workarounds

*   **GPU Out-of-Memory (OOM):**
    If the unified memory pressure triggers an uncatchable Metal allocation crash, reduce `--batch-size` to `1` or decrease `--lora-layers` from 16 to 8.
*   **Data Pollution:**
    Always run `uv run mlxtune clean` before starting a new experiment to ensure older adapter weights or logs do not pollute the workspace.
