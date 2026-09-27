# Gemma 4 Multimodal End-to-End Test Plan

This document outlines the complete workflow for testing the `gemmmma` pipeline with a native multimodal model (like Gemma 4) on Apple Silicon. It covers data preparation, LoRA training, evaluation, and native Swift inference.

## Prerequisites

Ensure you have the official Google Gemma 4 model downloaded and converted to MLX format. 

```bash
# 1. Download the uncompressed 16-bit model
hf download google/gemma-4-e2b-it --local-dir ~/projects/gemma/google-gemma-4-e2b

# 2. Convert to Apple Silicon MLX format
uv run --with mlx-vlm -m mlx_vlm.convert \
  --hf-path ~/projects/gemma/google-gemma-4-e2b \
  --mlx-path ~/projects/gemma/mlx-gemma-4-e2b
```

---

## Phase 1: The MLX Fine-Tuning Pipeline (`mlxtune`)

### Step 1: Prepare the Multimodal Dataset
We will use the open `PolyAI/minds14` audio instruction dataset. This step downloads the audio `.wav` files into your Hugging Face cache and generates `train.jsonl` and `valid.jsonl` files with the interleaved ChatML structure expected by Gemma 4.

```bash
uv run mlxtune prep \
  --dataset "PolyAI/minds14" \
  --dataset-config "en-US" \
  --samples 50 \
  --dest ./data_audio
```
*Verify: Check `./data_audio/train.jsonl` to ensure the `content` array contains `{"type": "audio", "audio": "<absolute_path>"}`. Bare filenames fail in the collator, and the audio files must exist on disk (not only inside the Hugging Face cache's arrow files).*

### Step 2: Train the Multimodal LoRA Adapter
Run the fine-tuning process. The `--multimodal` flag instructs the pipeline to use `mlx-vlm` to process the audio tensors.
*   **Batch Size:** Use `1` (the collator processes one audio sample at a time). Add `--tune-audio-encoder` to also train the audio tower (≈ +35% memory).
*   **Telemetry:** Open the `MLXMonitor` Swift app in another window to watch the `Peak mem (GB)` and `Train loss` metrics live.

```bash
uv run mlxtune train \
  --model ~/projects/gemma/mlx-gemma-4-e2b \
  --data ./data_audio \
  --batch-size 1 \
  --iters 40 \
  --multimodal
```
*Verify: An `adapters/` directory should be created containing `adapters.safetensors`.*

### Step 3: Evaluate the Fine-Tuned Model
Test the newly trained adapter on the fly using a sample audio file.

```bash
uv run mlxtune eval \
  --model ~/projects/gemma/mlx-gemma-4-e2b \
  --adapter ./adapters \
  --prompt "Listen to this and transcribe it." \
  --multimodal
```
*(Note: As `mlx-vlm` generation scripts evolve, you may need to pass `--audio /path/to/file.wav` if the basic `eval` command does not automatically extract it from the prompt).*

### Step 4: Fuse the Adapters
Bake the LoRA weights permanently into the base model. The fused checkpoint keeps the base's key layout, so it loads with `mlx_vlm.load` like the original, and it should reproduce the adapter's transcriptions exactly.
```bash
uv run mlxtune fuse \
  --model ~/projects/gemma/mlx-gemma-4-e2b \
  --adapter ./adapters \
  --dest ./fused_gemma_4_audio \
  --multimodal
```

---

## Phase 2: Native Inference Validation

Once you have a fully fused model (or if you just want to test the base model without adapters), you can validate the native Apple Silicon audio execution.

### Python Validation (`mlx-vlm`)
Run the standalone Python script to execute the audio pipeline entirely on the Mac GPU.

```bash
uv run scripts/ask_audio.py ~/projects/kokoro-rs/test.wav \
  --prompt "Describe the speaker's tone in this audio and translate it to English." \
  --model ~/projects/gemma/mlx-gemma-4-e2b
```

### Swift Validation (`mlx-swift`)
Run the native macOS UI to validate the translation of the Conformer neural network layers into Swift. 

```bash
cd MLXAudioUI
swift run
```
1. Click **Browse...** and select a `.wav` file.
2. Click **Run Gemma 4 Inference**.
3. Verify the Output Stats. You should see the Spectrogram shape (`[1, 99, 128]`) successfully shrink into the Encoded Features shape (`[1, 25, 1024]`), proving the native Swift audio pipeline is mathematically sound.
