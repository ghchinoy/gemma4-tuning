---
name: gemma-multimodal-tuning
description: Guides agents on preparing multimodal datasets in interleaved ChatML format, executing audio/vision fine-tuning with mlx-vlm, targeting specific layers, and performing key-sanitized weights fusion to avoid silent blank weights loading.
compatibility: Requires Python 3.14+, uv, Apple Silicon, and mlx-vlm
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
            {"type": "audio", "audio": "data/audio/speaker_1.wav"},
            {"type": "text", "text": "Transcribe this audio."}
          ]
        }
      ]
    }
    ```
2.  **Audio File Standards:** Stick to standard `16kHz` mono `.wav` files. This completely eliminates expensive real-time resampling CPU overhead during high-speed training loops.
3.  **Strategic Target Modules:**
    *   **Language Model Only:** Attach LoRA adapters to the Language Model attention projection layers (e.g., `q_proj`, `v_proj`). Best for learning style, dialect, and formatting constraints.
    *   **Audio/Vision Encoder Only:** Attach LoRA adapters directly to the encoder layers (e.g., `ConformerBlocks` inside the audio tower). Required when introducing entirely new acoustic conditions, accents, or specific environments.
4.  **Weight Key-Sanitization on Save:** MLX skips automatic key sanitization when saving fused weights with `save_safetensors` and `metadata={"format": "mlx"}`. You MUST flat-map in-memory parameter keys to exact leaf paths (e.g., remapping `language_model.model.layers` to `model.language_model.layers`) to avoid silent zero/blank weight loads resulting in infinite `<pad>` outputs.

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
  --batch-size 2 \
  --rank 8 \
  --lr 1e-5
```
*   Add `--tune-audio-encoder` if you want to apply LoRA layers to the audio tower conformer layers rather than language projections.

### 4. Evaluate Multimodal Adapters
Evaluate the trained multimodal adapter by passing a prompt and the `--multimodal` option:
```bash
uv run mlxtune eval \
  --model ./model \
  --adapter ./adapters \
  --prompt "Transcribe the following audio." \
  --multimodal
```

### 5. Multimodal Key-Sanitized Fusion
Bake the LoRA weights into the multimodal base. The `mlxtune` CLI handles the manual key remapping internally:
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
    If the fused model produces repetitive or endless `<pad>` tokens on inference, the weights saved in the safetensors file are not correctly prefix-mapped. Ensure your fusion script flat-maps the `flat_params` keys precisely before saving.
