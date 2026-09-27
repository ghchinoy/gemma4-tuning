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
Instead of using `mlx_lm.lora`, our pipeline must invoke `mlx_vlm.lora` (or its equivalent training loop). `mlx-vlm` is Apple's library explicitly designed to handle the heavy lifting of routing audio/image tensors through the Vision/Audio encoders before passing the embeddings to the language model.

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

We will update the `gemmmma` CLI arguments to allow users to specify `--tune-audio-encoder` or `--tune-language-model`.

## 3. Telemetry & The `MLXMonitor`

Because processing raw audio waveforms consumes significantly more GPU memory than text tokens, our telemetry system (`training_log.jsonl`) needs to track memory spikes closely.

When training with audio, batch sizes usually need to drop from `4` or `8` down to `1` or `2` to avoid `Out of Memory` (OOM) errors on Mac. The Swift `MLXMonitor` dashboard should be updated to highlight `peak_memory` distinctly when `multimodal=True` is detected in the config event.

## 4. Exporting & Fusing

Just like text models, once the LoRA training finishes, the adapter weights must be fused back into the base model.
Because we are using `mlx-vlm`, the `mlxtune fuse` command will fuse adapters into *both* the language model and the audio encoder simultaneously, producing a single, monolithic `.safetensors` directory ready for our `ask_audio.py` script or the upcoming native Swift implementation.

### Export Format Compatibility Matrix
*   **GGUF `q4_0`:** Supported for all architectures (E2B, E4B, 12B). Preserves QAT calibration.
*   **Apple MLX 4-bit SafeTensors:** Supported for all architectures. Native high-throughput Apple Silicon inference.
*   **LiteRT-LM (`.litertlm`):** Optimized for edge NPU deployment on **E2B** and **E4B** mobile models. Note: Exporting 12B models via `litert-torch` requires >90GB RAM due to upstream FP32 graph materialization (see `docs/FRICTION_LOG.md` entry **FL-004**).
