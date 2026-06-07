# MLXAudioInfer (Swift CLI)

This is a native Apple Silicon Swift executable designed to run the Gemma 4 Multimodal Audio pipeline using Apple's official `mlx-swift` libraries and the `Gemma4Audio` classes ported by the community.

## Purpose
While Python (`mlx-vlm`) is fantastic for fast prototyping and fine-tuning, deploying a model inside a macOS or iOS application requires it to be compiled in Swift using `MLXNN`. This CLI serves as the testing ground for the native port of the Audio DSP logic and the neural network arrays.

## Current State
This executable successfully imports the `Gemma4AudioFeatureExtractor` and can compute the `mel-spectrogram` tensors natively on the Mac GPU from raw audio. The final Language Model execution is currently mocked (a stub) while the full `Gemma4Omni` wrapper is being ported to `mlx-swift-examples`.

## How to Run

### The Metal Library Path Issue
Because this is a standalone Swift Package Manager (SPM) executable that links against the MLX C++ Core (`Cmlx`), running it directly from the terminal via `swift run` often results in a missing `.metallib` error (e.g., `MLX error: Failed to load the default metallib. library not found`). 

This is a known issue with how SPM bundles resources for CLI binaries. 

**To test this code properly:**
We highly recommend using the Xcode GUI (or creating a full `.app` bundle, like the `MLXAudioUI` sample provided) which handles the Metal shader compilation and resource linking automatically.

If you just want to verify compilation from the terminal:
```bash
cd MLXAudioInfer
swift build --configuration release
```
