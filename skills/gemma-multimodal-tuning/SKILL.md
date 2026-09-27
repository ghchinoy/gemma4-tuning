---
name: gemma-multimodal-tuning
description: Guides agents on preparing multimodal datasets in interleaved ChatML format, executing audio/vision fine-tuning with mlx-vlm, targeting specific layers, and performing key-sanitized weights fusion to avoid silent blank weights loading.
compatibility: Requires Python 3.12 (uv), Apple Silicon, and mlx-vlm
---

# Gemma 4 Multimodal (Audio & Vision) Tuning Skill

This skill provides step-by-step instructions for AI coding assistants and agents to fine-tune natively multimodal models (like Gemma 4) with interleaved text, audio, and vision inputs on Apple Silicon.

## 🧠 Why Multimodal Fine-Tuning is Different

Fine-tuning a multimodal network requires routing raw audio waveforms or image grids through specialized encoders before injecting their representations as token embeddings into the main Language Model (LM):

1.  **Interleaved ChatML Standard:** The training dataset must structure the `messages` content as an array of structured blocks (rather than a simple text string), supporting paths to files:
    ```json
    {
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "audio", "audio": "/abs/path/to/data_audio/wavs/speaker_1.wav"},
            {"type": "text", "text": "Transcribe this audio."}
          ]
        }
      ]
    }
    ```
2.  **Audio File Standards:** Use **absolute paths** to the audio files (bare filenames fail in the collator). Stick to standard `16kHz` mono `.wav` files. This completely eliminates expensive real-time resampling CPU overhead during high-speed training loops.
3.  **Strategic Target Modules:**
    *   **Language Model Only:** Attach LoRA adapters to the Language Model attention projection layers (e.g., `q_proj`, `v_proj`). Best for learning style, dialect, and formatting constraints.
    *   **Language Model + Audio Encoder (`--tune-audio-encoder`):** Also attaches LoRA to the audio tower's Conformer attention layers. Worth it for new acoustic conditions, accents or environments; on a single-domain ASR task (MInDS-14 banking) it cost +35% memory and 3.6x time for a small loss gain (see `docs/EXPERIMENT_JOURNAL.md`).
4.  **Fusing:** Use `mlxtune fuse --multimodal`, never a hand-rolled `save_safetensors`. It saves in the base checkpoint's key layout and carries audio-tower LoRA across. Hand-saving with `metadata={"format": "mlx"}` and non-leaf keys silently loads blank weights (infinite `<pad>`, FL-002).
5.  **Export limits:** A GGUF holds the text model only (audio/vision need a separate llama.cpp `mmproj` file). Raw-text (non-chat) training data needs a leading `<bos>` (FL-006).

---

## 🧭 Step-by-Step Workflow

### 1. Reset Workspace
```bash
uv run mlxtune clean
```

### 2. Prepare Multimodal Dataset
Verify that your local dataset contains files matching standard audio configurations (`16kHz` mono `.wav`) and format the JSONL lines into interleaved block arrays.

### 3. Run Multimodal LoRA Loop
Invoke training with the `--multimodal` flag, targeting either the language model or the audio encoder:
```bash
uv run mlxtune train \
  --model ./model \
  --data ./data_audio \
  --multimodal \
  --iters 200 \
  --batch-size 1 \
  --rank 8 \
  --lr 1e-5
```
*   Audio training uses batch size 1 (the collator processes one audio sample at a time).
*   Add `--tune-audio-encoder` to also apply LoRA to the audio tower's Conformer attention layers (in addition to the language model).

### 4. Evaluate Multimodal Adapters
Evaluate the trained multimodal adapter by passing a prompt and the `--multimodal` option:
```bash
uv run mlxtune eval \
  --model ./model \
  --adapter ./adapters \
  --prompt "Transcribe the following audio." \
  --multimodal
```

### 5. Multimodal Fusion
Bake the LoRA weights into the multimodal base. `mlxtune fuse --multimodal` loads the base with mlx-tune's
`FastVisionModel`, applies language-model LoRA and (for `--tune-audio-encoder` adapters) the audio-tower LoRA
plus its trained tensors, fuses them dequantized, and saves under the base checkpoint's key layout:
```bash
uv run mlxtune fuse \
  --model ./model \
  --adapter ./adapters \
  --dest ./fused_model_dequantized \
  --multimodal
```

---

## ⚠️ Common Edge Cases & Workarounds

*   **Apple Metal Driver OOM Crashes:**
    Processing audio/image frames consumes highly variable GPU memory. If a training run triggers an uncatchable Metal driver memory crash (`Command buffer execution failed`), reduce `--batch-size` to `1` and reduce targeted layers.
*   **Infinite Pad/Blank Outputs:**
    If a fused model produces endless `<pad>` tokens, the weights were saved with the wrong keys and loaded as blanks (FL-002). Re-fuse with `mlxtune fuse --multimodal` rather than a custom script.
*   **Adapter Loads But Fuse Changes Nothing:**
    `mlxtune fuse --multimodal` stops if no LoRA layers were applied. Check the adapter directory contains `adapters.safetensors` + `adapter_config.json` from `mlxtune train --multimodal`.
