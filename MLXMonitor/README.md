# MLXMonitor

A native macOS application built with Swift and SwiftUI Charts for real-time monitoring of MLX fine-tuning processes.

<img width="1012" height="644" alt="Image" src="https://github.com/user-attachments/assets/490736f0-f1f9-4fa1-992f-f0798eb2cb66" />

## Overview
When running the `mlxtune train` pipeline, the training script emits rich metrics (Train loss, Validation loss, learning rate, memory usage, and tokens per second) into a `training_log.jsonl` file. 

This native macOS app attaches a file watcher to that JSONL file and provides a live-updating, visual dashboard of your training progress.

## How to Run

Because this is built as a Swift Package, you don't even need to open Xcode. You can compile and run the application directly from your terminal.

1. Open a new terminal tab (so your training can continue in the background).
2. Navigate to the MLXMonitor directory:
   ```bash
   cd MLXMonitor
   ```
3. Run the application:
   ```bash
   swift run
   ```

## Features
*   **Live Loss Charting:** Instantly plots Train Loss (blue line) and Validation Loss (red dots/line) so you can easily spot when your model hits the "sweet spot" and begins overfitting.
*   **Performance Metrics:** Displays real-time training speed (tokens/sec) and peak memory usage directly in the bottom dashboard.
*   **Status Tracking:** Shows when a training run started, whether it is currently active, and when it completed based on timestamp events.
*   **Historical Review:** You can load past `training_log.jsonl` files from previous runs (e.g., `lima_log.jsonl`) to instantly visualize and review their performance.