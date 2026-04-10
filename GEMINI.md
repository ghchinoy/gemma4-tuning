# AI Assistant Guidelines for Gemmmma

Welcome to the `gemmmma` project. This file contains the core mandates, architectural goals, and pedagogical philosophies you must follow when assisting with this repository.

## 🎯 Project Goals

The primary objective of this project is **didactic ease of use**. It is a learning environment designed to demystify the process of fine-tuning Large Language Models—specifically Google's Gemma 4—on Apple Silicon using the MLX framework.

As we expand this project to include **multimodal tuning** (images, audio, video), the tooling must remain accessible, heavily documented, and operationally clean.

## 🛠️ Architectural Philosophy

We maintain a strict separation of concerns between the machine learning logic and the operational monitoring:

1.  **The ML Pipeline (`src/gemmmma/cli.py`):** Written in Python, managed by `uv`. This is the "Easy Button." It wraps complex MLX and `llama.cpp` commands into simple, linear steps (prep, train, fuse, gguf).
2.  **The Telemetry System (`training_log.jsonl`):** The Python pipeline must emit structured, self-documenting JSONL logs. This acts as the MLOps experiment tracker.
3.  **The Monitor (`MLXMonitor/`):** Written in native Swift and SwiftUI. It reads the telemetry system to provide a real-time macOS dashboard. Native code is used here because it excels at native UI and charting, keeping the Python ML code free of heavy UI libraries.

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

## 🤝 Interaction Protocol
When the user asks "How do I do X?", first outline the conceptual steps and the tools required. Wait for confirmation, then implement the code, explaining the specific ML parameters (like LoRA rank, batch size, or learning rate) you chose and why.