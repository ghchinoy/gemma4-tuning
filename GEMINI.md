# AI Assistant Guidelines for Gemmmma

Welcome to the `gemmmma` project. This file contains the core mandates, architectural goals, and pedagogical philosophies you must follow when assisting with this repository.

## 🎯 Project Goals

The primary objective of this project is **didactic ease of use**. It is a learning environment designed to demystify the process of fine-tuning Large Language Models—specifically Google's Gemma 4—on Apple Silicon using the MLX framework.

As we expand this project to include **multimodal tuning** (images, audio, video), the tooling must remain accessible, heavily documented, and operationally clean.

## 🛠️ Architectural Philosophy

We maintain a strict separation of concerns between the machine learning logic and the operational monitoring:

1.  **The ML Pipeline (`src/gemmmma/cli.py`):** Written in Python, managed by `uv`. This is the "Easy Button." It wraps [mlx-tune](https://github.com/ARahim3/mlx-tune) (training, merge, GGUF export via llama.cpp) into simple, linear steps (prep, train, fuse, gguf, benchmark). mlx-tune is pinned to our fork (`ghchinoy/mlx-tune`) until [ARahim3/mlx-tune#23](https://github.com/ARahim3/mlx-tune/pull/23) is released; gemmma adds what mlx-tune lacks (telemetry, journal, drift benchmark, QAT guardrails).
2.  **The Telemetry System (`training_log.jsonl`):** The Python pipeline must emit structured, self-documenting JSONL logs. This acts as the MLOps experiment tracker.
3.  **The Monitor (`MLXMonitor/`):** Written in native Swift and SwiftUI. It reads the telemetry system to provide a real-time macOS dashboard. Native code is used here because it excels at native UI and charting, keeping the Python ML code free of heavy UI libraries.
4.  **The Upstream Friction Log (`docs/FRICTION_LOG.md`):** Maintain a structured log tracking blocker and major severity issues encountered in upstream dependencies (e.g., Apple Metal, MLX, `llama.cpp`). This ensures reproducibility and provides concrete reports we can share with upstream maintainers or publish.

## 💻 Coding & Development Guidelines

When writing code, proposing architectures, or executing commands in this workspace, you must adhere to the following rules:

### 1. Explain the "Why" (Pedagogy First)
Never just write a script. Always explain the underlying ML concepts before execution. 
*   If we change datasets, explain the difference in philosophy (e.g., LIMA's "quality over quantity" vs. Alpaca).
*   If we change quantization formats, explain the clash between MLX 4-bit and standard PyTorch 16-bit tensors.
*   **Code Comments:** Python functions must have docstrings explaining *why* the step exists in the ML lifecycle, not just what the code does.

### 2. Prioritize MLOps Hygiene
Training generates gigabytes of temporary data. 
*   Always ensure `.gitignore` is updated to catch new artifact types (like `.safetensors` or `.gguf`).
*   Ensure tools have cleanup commands (like `mlxtune clean`) so the user's SSD doesn't fill up with dead experiments.
*   Ensure telemetry logs are self-documenting (e.g., injecting a `config` event at the start of a run).

### 3. Anticipate Multimodal Gemma 4
Gemma 4 is natively multimodal. When modifying data preparation scripts (like `format_row` in `cli.py`), always anticipate the nested ChatML structure. Text is no longer just a string; it is a content block:
```json
{"role": "user", "content": [{"type": "text", "text": "Hello"}]}
```
Future scripts will need to gracefully handle `{"type": "image", "image": "path.jpg"}` without breaking the existing text-only pipelines.

### 4. Tooling Constraints
*   **Python:** Always use `uv` for dependency management and running scripts (`uv run ...`). Do not use `pip` directly.
*   **Swift:** Keep the `MLXMonitor` isolated as an executable Swift Package. Avoid heavy third-party dependencies; rely on Apple's native frameworks (like `Charts` and `Combine`).

### 5. Hardware Constraints & Upstream Failures
Details for every item live in `docs/FRICTION_LOG.md` (FL-xxx); read it before debugging export or loading problems.
*   **Scripted GGUF inference uses `llama-completion`, never `llama-cli`:** `llama-cli` (llama.cpp build 10330) is chat-only; even with `-no-cnv` it sits in its `>` prompt loop at ~100% CPU and never exits (FL-001). Use `llama-completion -m <gguf> -bf <prompt_file> -n N --temp 0 -no-cnv --no-display-prompt -ngl 99`. Use `-bf`, not `-f` (`-f` drops the trailing newline after `<|turn>model`), and strip the template's leading `<bos>` (llama.cpp adds it).
*   **Apple Metal OOM:** High unified memory pressure causes uncatchable `[METAL] Command buffer execution failed` crashes (FL-003). Run each heavy step (12B training, 12B export) in its own process, and ask the user to close memory-heavy apps first. `MLX_GPU_DISABLE=1` forces a slow CPU fallback for debugging.
*   **Weight key layout on save:** Don't hand-roll `save_safetensors` for fused models. Use `mlxtune fuse` (mlx-tune `save_pretrained_merged`, or the multimodal fuse that saves in the base checkpoint's key layout). Saving with `metadata={"format": "mlx"}` and non-leaf keys silently loads blank weights (infinite `<pad>`, FL-002).

## ⚙️ Tactical Operations (Agent Checklist)

Hard-won rules from real runs on the 32GB M5. Follow them unless the user says otherwise.

**Environment**
*   Python **3.12** (`.python-version`); `uv sync` builds `.venv`. Requires `mlx-lm>=0.31.3` (0.31.2 can't load Gemma 4 E2B/E4B, FL-008).
*   GGUF export needs the llama.cpp checkout at `./llama.cpp` (or `$LLAMA_CPP_PATH` / `--llama-cpp`) **and** its own converter venv at `llama.cpp/.venv` with llama.cpp's pinned requirements. The main venv's newer transformers breaks the converter (`KeyError: 'global_head_dim'`, FL-007). Setup is in `README.md` step 7.
*   `llama-quantize` / `llama-completion` come from Homebrew `llama.cpp`.

**Training data (Gemma 4)**
*   Prefer chat-format `messages` data. The Gemma 4 chat template starts with `<bos>`.
*   Raw `{"text": ...}` data never gets a `<bos>` token (the tokenizer has `add_bos_token: false`); prepend `<bos>` or the LoRA won't transfer to llama.cpp/HF (FL-006).
*   Audio datasets must reference audio files by **absolute path**; bare filenames fail in the collator.
*   QAT checkpoints: train the **unquantized** QAT base, rank ≤ 16 (8 preferred), lr ~2e-6; export with `--qat` (strict Q4_0).

**Export & verification**
*   `mlxtune gguf --model <fused> --qat`, or in one step: `mlxtune gguf --model <base> --adapter ./adapters --qat`.
*   Never copy a `tokenizer.model` into a model directory from another checkpoint. Gemma 4 ships only `tokenizer.json`; a stray Gemma 2 file produced an unloadable GGUF (FL-005).
*   Verify every Gemma 4 GGUF before calling it done:
    *   architecture `gemma4`, vocab / `token_embd` second dimension **262144**;
    *   tensor names/types/shapes match Google's official QAT GGUF (`~/projects/gemma/gemma-4-*-it-qat-q4_0.gguf`); only the LoRA-trained layers should differ;
    *   it generates coherent text via `llama-completion`.
*   A GGUF contains the **text model only**; audio/vision need a separate llama.cpp `mmproj` file.
*   LiteRT-LM export: E2B/E4B only. 12B needs >90GB RAM (FL-004).
*   `mlxtune benchmark` compares MLX (reference) vs GGUF with identical tokens and greedy decoding; big drift usually means a prompt/token mismatch, not quantization.

**Resource hygiene**
*   Peak memory: 12B LoRA ~25.6GB, 12B export ~26.8GB, E4B export ~15–17GB, E2B audio + audio-tower LoRA ~21GB. Check `memory_pressure` before heavy steps.
*   Fused models are ~10–24GB and GGUFs 3–7GB: delete test artifacts when done, and keep them out of git (see `.gitignore`).
*   Record runs in the experiment journal (`mlxtune journal init/update`), and new upstream issues in `docs/FRICTION_LOG.md`.

**Upstream work**
*   Contributions to mlx-tune go through our fork, one feature per issue/PR. Local notes and validation logs live in `upstream_contributions/` (gitignored). When #23 is released, switch `pyproject.toml` back to the PyPI `mlx-tune` and drop the fork pin.

## 🤝 Interaction Protocol
When the user asks "How do I do X?", first outline the conceptual steps and the tools required. Wait for confirmation, then implement the code, explaining the specific ML parameters (like LoRA rank, batch size, or learning rate) you chose and why.