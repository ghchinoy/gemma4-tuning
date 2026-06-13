# The MLX Fine-Tuning Pipeline: Concepts & Nuances

*Note: If you are fine-tuning natively multimodal models (like Gemma 4) with Audio or Image datasets, please read the [Multimodal Pipeline Guide](MULTIMODAL_PIPELINE.md) first!*

This document explains the "why" behind the steps in the `mlxtune` pipeline, providing crucial context for training AI models locally.

## 1. Datasets: Quality vs. Quantity
When fine-tuning an already capable instruction-tuned model (like Gemma 4), the goal is usually **style, tone, or specific formatting**, rather than teaching it new facts.

*   **Alpaca (50k+ examples):** Great for taking a raw "base" model and turning it into a conversational assistant. However, it can teach the model to give generic, shallow answers because the dataset was synthetically generated.
*   **LIMA (1k examples):** "Less Is More for Alignment." Proves that 1,000 meticulously crafted, perfect human examples produce a vastly superior, highly structured assistant compared to 50,000 mediocre examples. 
*   **Takeaway:** When you want to teach Gemma to code in a specific style, curate 500 perfect examples rather than scraping 10,000 average ones.

## 2. LoRA (Low-Rank Adaptation)
Training a 2-billion or 9-billion parameter model requires massive memory. **LoRA** solves this by freezing the massive original neural network and attaching a tiny, trainable "sidecar" (the adapter).
*   During training, only this tiny adapter learns and changes.
*   This drops the memory requirement from hundreds of gigabytes down to something an Apple Silicon Mac can easily handle.

## 3. Quantization Clashes (MLX vs GGUF) & QAT

Different ecosystems handle "quantization" (compressing model weights from 16-bit to 4-bit or 8-bit) differently.
*   **MLX:** Uses highly specific 4-bit matrix scales tailored for Apple's Metal framework.
*   **llama.cpp:** Uses a format called GGUF. The conversion scripts for GGUF expect clean, standard 16-bit PyTorch tensors.
*   **The Pipeline Fix:** To bridge these ecosystems, the `mlxtune fuse` command actively **dequantizes** the MLX model back to 16-bit (`--dequantize`). Then, the `mlxtune gguf` command passes it to `llama.cpp` to be **re-quantized** into a standard `q8_0` or `q4_k_m` GGUF file.

### 🧠 Gemma 4 QAT (Quantization-Aware Training) Alignment

When using natively pre-conditioned QAT base models (such as `google/gemma-4-E2B-it-qat-q4_0-unquantized`), the mathematical alignment requires special handling:

1.  **What is QAT?**
    Standard post-training quantization (PTQ) takes a model trained in 16-bit float and truncates the weights afterwards, introducing rounding errors ("quantization noise") that degrade accuracy. QAT, however, models this low-precision noise *during pre-training/alignment*. This pre-conditions the high-precision weights so that when they are compressed, they experience almost zero accuracy loss.
2.  **Why load the unquantized checkpoint?**
    Always fine-tune the high-precision unquantized QAT checkpoint. Running LoRA on a pre-quantized MLX community model causes gradient misalignment and disrupts the pre-conditioned weight paths.
3.  **Strict GGUF Alignment (Enforcing `Q4_0`):**
    QAT models are pre-conditioned **specifically and exclusively** for the 4-bit `q4_0` quantization scheme. If you run `mlxtune gguf` and request `q8_0` or `q4_k_m` (standard PTQ parameters), you completely bypass the QAT optimization, losing its accuracy benefits. For this reason, the `--qat` flag in `mlxtune gguf` actively overrides the export parameters and enforces `q4_0`.
4.  **LoRA Rank Conservatism:**
    Because standard MLX does not model quantization noise for your newly trained adapters, fusing them and then compressing them to 4-bit will subject the adapter weights to naive post-training quantization. To mitigate this, keep your LoRA rank conservative (`--rank 8` or `16`) so the model relies on the pre-conditioned QAT base weights rather than over-indexing on unquantized adapter paths.

---

## 4. Fusing
When LoRA training finishes, you don't actually have a new model; you just have the original base model plus a tiny `adapters.safetensors` file. "Fusing" mathematically multiplies these two matrices together to create a single, permanently altered model directory. This is mandatory if you want to export the model out of the MLX ecosystem.

---

## 5. Scaling to the Cloud (JAX/TPU)
MLX is brilliant for local prototyping on Macs. However, if you want to train on massive datasets, use the 26B parameter model, or scale to clusters of Google Cloud TPUs, you will need to transition.
*   **The Path:** Learn **JAX** (Google's high-performance numerical framework) and the **Kauldron** library.
*   **The Benefit:** JAX supports `DeviceMesh` and automatic sharding, meaning it can mathematically slice a massive model across 8 or 64 TPU cores automatically without you having to write complex distribution logic. Keras 3 (with the JAX backend) is the easiest bridge to this cloud-scale ecosystem.

## 6. Artifact Management & Cleaning
Machine learning pipelines generate a massive amount of temporary data (adapter checkpoints, dequantized weights, and real-time JSONL logs). The `mlxtune clean` command is a crucial MLOps practice for maintaining a healthy local environment. It ensures that subsequent experimental runs do not accidentally mix adapter weights, pollute your charts, or exhaust your Mac's SSD storage with gigabytes of intermediate Safetensors.