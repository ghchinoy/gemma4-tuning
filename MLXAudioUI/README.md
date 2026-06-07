# MLXAudioUI (SwiftUI App)

This is a native macOS SwiftUI application designed to test the Gemma 4 Multimodal Audio pipeline using Apple's official `mlx-swift` libraries and the `Gemma4Audio` port.

## Purpose
This app provides a user-friendly graphical interface to test the DSP operations (like Mel-spectrogram extraction) and the execution of the Gemma 4 audio models natively on the Mac GPU. It is structured so that it can be contributed back to the `ml-explore/mlx-swift-examples` repository once the full Gemma 4 model class is complete.

## Features
1. **Audio File Selection:** A native `NSOpenPanel` to browse and select `.wav` files.
2. **Prompt Editing:** A multiline text editor to customize the prompt sent to the LLM alongside the audio.
3. **Native MLX Execution:** Connects directly to the `Gemma4AudioFeatureExtractor` to process the audio arrays into spectrograms using `MLXFFT`.
4. **Stats & Output:** Displays execution time, tensor shape output, and the final generated text response.

## Current State
The UI is fully functional and successfully executes the native `Gemma4AudioFeatureExtractor` to process raw audio arrays into the `[1, 99, 128]` 128-bin Mel-spectrogram tensors on the GPU.

It then successfully passes those spectrograms through the `AudioEncoder` (the native Swift port of the Conformer blocks) to produce the `[1, 25, 1024]` acoustic embeddings. 

*Note:* The final text generation step is currently a stub. It will execute the full `mlx-swift` generation loop as soon as the `Gemma4Omni` model wrapper is merged into the examples repository.

## How to Run

You can run this SwiftUI application directly from the command line using Swift Package Manager. Since it's a UI app, it bypasses the `.metallib` path issue that sometimes affects pure CLI binaries.

```bash
cd MLXAudioUI
swift run
```
*(Wait a moment for it to compile, and the macOS window will appear on your screen).*

Alternatively, you can open the `Package.swift` file in **Xcode** and click the Play button to build and run it with full debugging support.
