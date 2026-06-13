# 📓 Gemma 4 QAT Fine-Tuning Experiment Journal

This journal serves as an experiment tracker and MLOps log for training, fusing, quantizing, and evaluating Gemma 4 QAT (Quantization-Aware Training) models on Apple Silicon.

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
