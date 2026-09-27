# MLOps Advice: Fine-Tuning Gemma 4 QAT Models with MLX

This document contains strategic engineering guidelines and configurations for fine-tuning Google's **Gemma 4 Quantization-Aware Training (QAT)** unquantized checkpoints using the `mlxtune` pipeline in `gemmma`.

---

## 1. Core Model & Precision Setup

* **Base Model Selection:** Always target the unquantized QAT safe-tensors directory (e.g., `google-gemma-4-E2B-it-qat-q4_0-unquantized`).
* **Preserve Base Precision:** Fine-tuning must run on top of the unquantized `bfloat16`/`float16` weights. Do not download pre-quantized MLX weights for training. Fine-tuning on high-precision weights preserves gradient flow and ensures the highest adapter quality.

---

## 2. Mandatory Downstream Quantization Alignment

Because the base weights were mathematically optimized during Google's pre-training to withstand the specific mathematical constraints of the 4-bit classic quantization format, **you must target `q4_0` during the GGUF export step.**

* **The Rule:** In Step 7 of the `mlxtune` pipeline, change the output format from the default `q8_0` or `q4_k_m` to **`q4_0`**.
* **Why?** Using any other quantization format (such as `q8_0` or `q4_k_m`) will completely bypass the QAT optimizations, negating the accuracy and intelligence retention benefits.

Execute your export command like this:
```bash
uv run mlxtune gguf --outtype q4_0
```

---

## 3. Multimodal Routing: Audio & Vision

Since Gemma 4 E2B and E4B are fully multimodal models:

* **Use `mlx-vlm`:** Ensure the CLI backend routes training through `mlx-vlm.lora` rather than `mlx-lm.lora`. This ensures the audio Conformer blocks and projection layers are correctly loaded, routed, and backpropagated.
* **LoRA Targeting Strategy:**
  * **Language Model Only:** Attach adapters to the self-attention/feed-forward projections (`gate_up_proj`, `down_proj`). This is sufficient if teaching the model to follow specific output templates or formatting styles.
  * **Multimodal LoRA:** Attach adapters to both the language model and the `Conformer` encoder inside the audio stack. This is required if you are introducing specialized acoustic conditions (e.g., background noise, low-bitrate radio signals, unique vocal accents).

---

## 4. Mitigating Adapter Quantization Loss

Standard MLX fine-tuning applies post-training LoRA; it does **not** simulate quantization noise during training. When you permanently fuse your FP16 LoRA adapter weights back into the base model and compress the unified file to `q4_0`, the adapter layers will undergo standard naive quantization and may suffer slight accuracy degradation.

### Recommended Mitigations:
1. **Conservative Rank:** Keep LoRA rank low (e.g., `rank=8` or `16`). A lower rank restricts the model from over-indexing on unquantized weights, forcing it to rely on the QAT-optimized base model pathways.
2. **Lower Alpha:** Set a lower `lora_alpha` (e.g., `alpha=16` or `32` for `rank=16`) to ensure the adapter's influence scales smoothly without causing stark weight disparities that might blow up during quantization.
