# Multimodal Fine-Tuning Pipeline: Audio & Vision

As we transition `gemmmma` from text-only models to natively multimodal models like **Gemma 4**, our fine-tuning pipeline (`cli.py`) and dataset preparation must evolve. 

Gemma 4 natively supports interleaved text, audio, and images. This document explains how we adapt our MLOps pipeline to handle these new modalities, focusing primarily on Audio.

## 1. Preparing Multimodal Datasets

Historically, our datasets were simple JSONL files containing text conversations:
```json
{"role": "user", "content": "Translate this to French: Hello"}
{"role": "assistant", "content": "Bonjour"}
```

### The New ChatML Standard (Interleaved)
For Gemma 4, the input is no longer just a string. It is an **interleaved list of content blocks**. This allows the model to receive a picture or an audio file at any point in the conversation.

To fine-tune the model to understand a specific speaker's tone, or to transcribe audio in a specific format, your dataset must structure the `content` as an array:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio": "path/to/local/audio/speaker_1.wav"},
        {"type": "text", "text": "Transcribe this audio and describe the speaker's emotional state."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "[Emotional State: Frustrated] I cannot get this pipeline to work."}
      ]
    }
  ]
}
```

**Key Dataset Rules:**
*   **Absolute or Relative Paths:** The `"audio"` or `"image"` keys must point to valid file paths on your local machine during the training phase.
*   **Audio Format:** Stick to standard `16kHz` mono `.wav` files to avoid real-time resampling overhead during training.

## 2. Adapting `gemmmma/cli.py`

Our current Python pipeline uses `mlx-lm`, which expects flat text. To support the new architecture, we need to make the following architectural shifts:

### A. Switch to `mlx-vlm`
Text-only models train through `mlx-lm`. Multimodal models need `mlx-vlm`, which routes audio/image tensors through the Vision/Audio encoders before passing the embeddings to the language model. `mlxtune train --multimodal` drives this through mlx-tune's `FastVisionModel` (it applies the LoRA layers and builds batches from the interleaved content blocks), while gemmma's own loop keeps streaming telemetry to `training_log.jsonl`.

### B. Updating `format_row`
The dataset parsing logic in our CLI needs to recursively check for `"type"` keys. 
When `cli.py` loads the JSONL file, it must:
1.  Read the file paths.
2.  (Optional but recommended) Validate that the `.wav` files actually exist before kicking off a 10-hour training run.
3.  Pass the structured dictionary directly to the `mlx_vlm` processor's `apply_chat_template`, rather than trying to join the text manually.

### C. LoRA Target Modules
When fine-tuning text models, we target the `q_proj` and `v_proj` (Attention) layers of the language model. 

For multimodal fine-tuning, you have a strategic choice:
*   **Tune the Language Model Only:** Freeze the audio encoder, and only attach LoRA adapters to the language model. The model will learn to *speak* differently based on what it hears.
*   **Tune the Audio Encoder (Multimodal LoRA):** Attach LoRA adapters to the `ConformerBlocks` inside the audio encoder. This is necessary if you are teaching the model to understand a completely new language, a heavy accent, or specific acoustic environments (like radio static) that it wasn't originally trained on.

`mlxtune train --multimodal` tunes the language model by default; add `--tune-audio-encoder` to also attach LoRA to the audio tower. Measured on MInDS-14 banking ASR (E2B, 50 iterations): the audio tower adds ~35% memory (15.5 → 21.0GB) and ~3.6x time for a small loss gain, so reserve it for real acoustic domain shifts (see `docs/EXPERIMENT_JOURNAL.md`).

## 3. Telemetry & The `MLXMonitor`

Because processing raw audio waveforms consumes significantly more GPU memory than text tokens, our telemetry system (`training_log.jsonl`) needs to track memory spikes closely.

Audio training runs at batch size 1 (the collator processes one audio sample at a time); the `config` event records `multimodal: true`, and each `train` event carries `peak_mem_gb` for `MLXMonitor`.

## 4. Exporting & Fusing

Just like text models, once the LoRA training finishes, the adapter weights must be fused back into the base model.
`mlxtune fuse --multimodal` fuses the adapter into *both* the language model and (if tuned) the audio encoder. It loads the base with mlx-tune's `FastVisionModel`, fuses each LoRA layer dequantized, and saves a single `.safetensors` under the base checkpoint's key layout, so the result loads like the original checkpoint (`mlx_vlm.load`, `ask_audio.py`). A fused checkpoint reproduces the adapter model's transcriptions exactly.

### Export Format Compatibility Matrix
*   **GGUF `q4_0`:** Verified for E2B, E4B and 12B (Sept 2026, via mlx-tune `export_to_gguf(qat=True)`): tensor layout and types match Google's official QAT GGUFs, and all three load and generate in llama.cpp. Text model only; audio/vision need a separate `mmproj` file. Check vocab = 262144 (see FL-005).
*   **Apple MLX 4-bit SafeTensors:** Supported for all architectures. Native high-throughput Apple Silicon inference.
*   **LiteRT-LM (`.litertlm`):** Optimized for edge NPU deployment on **E2B** and **E4B** mobile models. Note: Exporting 12B models via `litert-torch` requires >90GB RAM due to upstream FP32 graph materialization (see `docs/FRICTION_LOG.md` entry **FL-004**).
