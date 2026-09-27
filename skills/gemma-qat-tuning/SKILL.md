---
name: gemma-qat-tuning
description: Guides agents on fine-tuning Gemma 4 Quantization-Aware Training (QAT) unquantized checkpoints, maintaining mathematical alignment during LoRA, and exporting to GGUF using strict q4_0 parameters to avoid post-training quantization drift on edge devices.
compatibility: Requires Python 3.14+, uv, Apple Silicon, and Metal GPU
---

# Gemma 4 QAT Alignment and Tuning Skill

This skill provides step-by-step instructions for AI coding assistants and agents to fine-tune Gemma 4 Quantization-Aware Training (QAT) models on Apple Silicon using the local `mlxtune` pipeline.

## 🧠 Why QAT Fine-Tuning is Different

Quantization-Aware Training (QAT) pre-conditions high-precision weights during pre-training to model quantization noise. This allows compressing the model afterwards with almost zero accuracy loss. However, standard post-training quantization (PTQ) fine-tuning rules break QAT:

1.  **Always Tune Unquantized:** Run LoRA on the high-precision unquantized QAT base checkpoint (e.g. `google-gemma-4-E2B-it-qat-q4_0-unquantized`). Fine-tuning pre-quantized models causes major gradient misalignment and destroys pre-conditioned paths.
2.  **LoRA Rank Conservatism:** Standard MLX does not model quantization noise for new adapters. Keep LoRA rank conservative (`--rank 8` or `16`) so the model leverages the pre-conditioned base weights rather than relying too heavily on unquantized adapter weights.
3.  **Strict GGUF Alignment:** QAT weights are pre-conditioned **exclusively for the 4-bit `q4_0` scheme**. Exporting to other types like `q8_0` or `q4_k_m` completely bypasses QAT benefits, resulting in severe performance loss. Use the `--qat` flag to enforce `q4_0`.

---

## 🧭 Step-by-Step Workflow

### 1. Reset Workspace
Always run `clean` first to avoid mixing adapter weights or corrupting logs:
```bash
uv run mlxtune clean
```

### 2. Prepare the Training Dataset
Prep custom instruction-response pairs into ChatML JSONL formatting:
```bash
uv run mlxtune prep --dataset yahma/alpaca-cleaned --dest ./data --samples 1000
```

### 3. Execute the QAT LoRA Loop
Use a **conservative learning rate** (`2e-6`) to prevent training divergence (which causes repetitive loops like `"plays it plays it"`):
```bash
uv run mlxtune train \
  --model ./model/google-gemma-4-E2B-it-qat-q4_0-unquantized \
  --data ./data \
  --qat \
  --iters 20 \
  --batch-size 2 \
  --rank 8 \
  --lora-layers 16 \
  --lr 2e-6
```
*Note: A learning rate of `2e-6` yields stable validation loss drops (e.g., from `2.26` to `0.70` in dry runs) with zero text repetition.*

### 4. Fuse Weights
Fuse the trained LoRA adapter weights back into the unquantized base weights, dequantizing to 16-bit to allow clean `llama.cpp` conversion:
```bash
uv run mlxtune fuse \
  --model ./model/google-gemma-4-E2B-it-qat-q4_0-unquantized \
  --adapter ./adapters \
  --dest ./fused_model_dequantized
```

### 5. GGUF Export (Enforcing Q4_0)
Compile the fused 16-bit safetensors back into a quantized GGUF, enforcing `q4_0` via the `--qat` flag:
```bash
uv run mlxtune gguf \
  --model ./fused_model_dequantized \
  --dest my-custom-model.gguf \
  --qat
```
Or skip the fuse step: `--model <unquantized QAT base> --adapter ./adapters --qat`.
Tokenizer files come only from the model itself; never copy in a `tokenizer.model` from another checkpoint (FL-005).
Verify the result: `token_embd.weight` vocab must be 262144 for Gemma 4.

### 6. Benchmark Semantic Drift
Evaluate the semantic "drift" introduced by low-precision compression. This compares high-precision reference outputs directly to the low-precision quantized GGUF outputs:
```bash
uv run mlxtune benchmark \
  --reference-model ./fused_model_dequantized \
  --gguf-model my-custom-model.gguf \
  --prompt "Describe the core benefits of using Quantization-Aware Training (QAT) for edge deployments."
```

---

## ⚠️ Common Edge Cases & Workarounds

*   **Training Loss Divergence (Repetitive Outputs):**
    If the model generates repetitive outputs or validation loss climbs, the learning rate is too high (e.g., `2e-5`). Run `mlxtune clean` and retrain with a learning rate of `2e-6`.
*   **Stale Adapter Pollution:**
    Failing to clean out older adapters in `./adapters` before retuning can cause silent model degradation. Always run `uv run mlxtune clean` first.
