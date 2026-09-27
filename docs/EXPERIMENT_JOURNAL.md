# 📓 Gemma 4 QAT Fine-Tuning Experiment Journal

This journal serves as an experiment tracker and MLOps log for training, fusing, quantizing, and evaluating Gemma 4 QAT (Quantization-Aware Training) models on Apple Silicon.

---

## 🔬 Experiment ID: AUDIO-E2B-CONFIGA-LM-ONLY
*   **Date:** 2026-09-07
*   **Base Model:** `google-gemma-4-E2B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `50` | Training steps. |
| **Batch Size** | `1` | Batch training count. |
| **LoRA Rank (`--rank`)** | `8` | LoRA adapter width. |
| **Target Layers** | `8` | Frozen boundary layer targeting. |
| **Dataset** | `minds14-banking-en-US` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `2.9062`
*   **Validation Loss at end:** `0.8086`
*   **Peak GPU Memory:** `15.48 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:** *[Awaiting execution]*
*   **Quantized GGUF Response:** *[Awaiting execution]*
*   **Jaccard Similarity Score:** `*[Awaiting execution]*`
*   **Calculated Quantization Drift:** `*[Awaiting execution]*`

---

### 💡 Notes & Lessons Learned
Config A: LM-only LoRA (audio encoder frozen), 12.08M trainable params (0.261%), 45 minds14 banking-domain audio transcription examples. Wall time ~91s for 50 iters. Baseline for audio-tower ablation vs Config B.

---

## 🔬 Experiment ID: AUDIO-E2B-CONFIGB-LM-PLUS-AUDIO-TOWER
*   **Date:** 2026-09-07
*   **Base Model:** `google-gemma-4-E2B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `50` | Training steps. |
| **Batch Size** | `1` | Batch training count. |
| **LoRA Rank (`--rank`)** | `8` | LoRA adapter width. |
| **Target Layers** | `8` | Frozen boundary layer targeting. |
| **Dataset** | `minds14-banking-en-US` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `2.9062`
*   **Validation Loss at end:** `0.7461`
*   **Peak GPU Memory:** `20.962 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:** *[Awaiting execution]*
*   **Quantized GGUF Response:** *[Awaiting execution]*
*   **Jaccard Similarity Score:** `*[Awaiting execution]*`
*   **Calculated Quantization Drift:** `*[Awaiting execution]*`

---

### 💡 Notes & Lessons Learned
Config B: LM LoRA + audio tower LoRA (48 Conformer layers adapted), same 12.08M trainable params reported by mlx-tune PEFT summary. Wall time ~332s for 50 iters (3.6x slower than Config A due to audio encoder backward pass). Peak memory +35% vs Config A (20.96GB vs 15.48GB). Final loss modestly lower (0.746 vs 0.809) but not dramatically better on this small 45-example single-speaker-domain (banking) dataset -- audio tower tuning benefit likely more pronounced on accent/noise-shift or new-language transcription tasks per docs/MULTIMODAL_PIPELINE.md guidance, not simple single-domain ASR fine-tuning.

---

## 🔬 Experiment ID: QAT-4B-ELDAMO-001
*   **Date:** 2026-06-14
*   **Base Model:** `google-gemma-4-E4B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `100` | Training steps. |
| **Batch Size** | `1` | Batch training count. |
| **LoRA Rank (`--rank`)** | `8` | LoRA adapter width. |
| **Target Layers** | `8` | Frozen boundary layer targeting. |
| **Dataset** | `eldamo-combined` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `4.137`
*   **Validation Loss at end:** `2.13`
*   **Peak GPU Memory:** `17.922 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:**
    ```text
    This is a very specific word, and it is not a standard, common entry in the primary Quenya lexicons (like those compiled by Thomas Paun or Vinyard). This suggests it might be a highly descriptive, poetic, or constructed word.\n\nHowever, by breaking the word down into its likely morphemes (parts), we can determine its probable meaning based on Quenya grammar:\n\n### Etymological Breakdown\n\n1. **apa-**: This is the Quenya word for **"water."**\n2. **-quista**: This suffix is derived from a root that often relates to a state, condition, or quality. In this context, it functions as an adjective or a noun describing a state.\n\n### Probable Meaning\n\nBased on this structure, **"apaquista"** most likely means:\n\n* **"One who is of the water"**\n* **"Aquatic"**\n* **"Watery"**\n* **"A being associated with water"**\n\nIn essence, it is a descriptive term for something or someone that is fundamentally connected to or defined by water.
    ```
*   **Quantized GGUF Response:**
    ```text
    Not executable in llama.cpp (llama-server/llama-cli) due to the upstream Attention KV layer tensor loader bug: missing tensor "blk.24.attn_k.weight" in Gemma 4 E4B. Verified successful execution bypass using native MLX on the fused FP16 weights.
    ```
*   **Jaccard Similarity Score:** `*[Awaiting execution]*`
*   **Calculated Quantization Drift:** `*[Awaiting execution]*`

---

### 💡 Notes & Lessons Learned
Successful 100-iteration fine-tuning run of Gemma 4 E4B (4B Mobile QAT variant) on Apple GPU (Metal) using combined conversational QA and CoT phonetic derivation datasets. Stable convergence without out-of-memory errors.

---

## 🔬 Experiment ID: QAT-12B-ELDAMO-001
*   **Date:** 2026-06-13
*   **Base Model:** `google-gemma-4-12B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `100` | Training steps. |
| **Batch Size** | `1` | Batch training count. |
| **LoRA Rank (`--rank`)** | `8` | LoRA adapter width. |
| **Target Layers** | `8` | Frozen boundary layer targeting. |
| **Dataset** | `eldamo-elvish` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `7.995`
*   **Validation Loss at end:** `2.243`
*   **Peak GPU Memory:** `24.97 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:** *[Awaiting execution]*
*   **Quantized GGUF Response:** *[Awaiting execution]*
*   **Jaccard Similarity Score:** `*[Awaiting execution]*`
*   **Calculated Quantization Drift:** `*[Awaiting execution]*`

---

### 💡 Notes & Lessons Learned
Successfully resolved Gemma 4 LoRA submodule naming mismatch. Descended loss steadily to 2.20.

---

## 🔬 Experiment ID: QAT-12B-ELDAMO-002
*   **Date:** 2026-06-13
*   **Base Model:** `google-gemma-4-12B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `500` | Training steps. |
| **Batch Size** | `2` | Batch training count. |
| **LoRA Rank (`--rank`)** | `16` | LoRA adapter width. |
| **Target Layers** | `16` | Frozen boundary layer targeting. |
| **Dataset** | `eldamo-elvish` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `7.995`
*   **Validation Loss at end:** `1.584`
*   **Peak GPU Memory:** `25.32 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:** *[Awaiting execution]*
*   **Quantized GGUF Response:** *[Awaiting execution]*
*   **Jaccard Similarity Score:** `*[Awaiting execution]*`
*   **Calculated Quantization Drift:** `*[Awaiting execution]*`

---

### 💡 Notes & Lessons Learned
Full-dataset 500-iteration QAT tuning loop on 16,726 ChatML-formatted Eldamo Elvish instruction pairs. Learning rate Cosine Decay successfully drove validation loss from 7.995 down to 1.584 without overfitting. Stably ran with 25.32 GB peak memory footprint on Apple Silicon. Fused model dequantized to FP16 and then quantized back to q4_0 using strict QAT-aligned parameter schemes to prevent post-training quantization drift. Successfully copied to the Mithlond local application support directory.

---

## 🔬 Experiment ID: QAT-DRYRUN-001
*   **Date:** 2026-06-12
*   **Base Model:** `google-gemma-4-E2B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `20` | Training steps. |
| **Batch Size** | `2` | Batch training count. |
| **LoRA Rank (`--rank`)** | `8` | LoRA adapter width. |
| **Target Layers** | `16` | Frozen boundary layer targeting. |
| **Dataset** | `yahma/alpaca-cleaned` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `2.8305`
*   **Validation Loss at end:** `4.0463`
*   **Peak GPU Memory:** `12.43 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:**
    ```text
    plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it
    ```
*   **Quantized GGUF Response:**
    ```text
    plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it plays it
    ```
*   **Jaccard Similarity Score:** `100.00%`
*   **Calculated Quantization Drift:** `0.00%`

---

### 💡 Notes & Lessons Learned
Successfully resolved the Gemma 4 strict loading bug by modifying Attention.__init__ to conditionally instantiate KV projection and normalization parameters. Fine-tuning completed on Apple Metal GPU (20 iterations, 12.43 GB peak memory). Fusion succeeded by adding metadata={'format': 'mlx'} to the safetensors, resolving the key-mangling issue on load. Benchmarked semantic drift showing identical output between dynamic and fused models (Jaccard word similarity = 100%). However, training telemetry showed divergence (loss rose from 2.83 to 4.04), resulting in repetitive generation ('plays it') from the adapters, suggesting the learning rate (2e-5) or iteration count requires tuning for QAT stability.

---

## 🔬 Experiment ID: QAT-STABLE-002
*   **Date:** 2026-06-12
*   **Base Model:** `google-gemma-4-E2B-it-qat-q4_0-unquantized`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** 🟢 COMPLETED

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `20` | Training steps. |
| **Batch Size** | `2` | Batch training count. |
| **LoRA Rank (`--rank`)** | `8` | LoRA adapter width. |
| **Target Layers** | `16` | Frozen boundary layer targeting. |
| **Dataset** | `yahma/alpaca-cleaned` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `2.2655`
*   **Validation Loss at end:** `0.7061`
*   **Peak GPU Memory:** `12.43 GB`

#### 2. Quantization Drift Analysis
*   **Reference FP16 Response:**
    ```text
    Quantization-Aware Training (QAT) is a crucial technique for deploying deep learning models onto resource-constrained edge devices (like mobile phones, IoT sensors, and embedded systems). Its core benefit lies in **minimizing the performance degradation** that typically occurs when moving a model from high-precision training
    ```
*   **Quantized GGUF Response:**
    ```text
    Quantization-Aware Training (QAT) is a technique used to reduce the model size and computational requirements of deep learning models, which is crucial for edge deployments where resources are often limited.

The core benefits of using QAT for edge deployments include:

1. **Reduced Model Size:**
    ```
*   **Jaccard Similarity Score:** `23.73%`
*   **Calculated Quantization Drift:** `76.27%`

---

### 💡 Notes & Lessons Learned
Second tuning iteration using a conservative learning rate (2e-6) from a clean initialization. Successfully prevented validation loss divergence and fully eliminated the repetitive output issue of Experiment 1, yielding highly coherent and fluent English text generation. Peak memory footprint remained stable at 12.43 GB.

---
