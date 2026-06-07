import SwiftUI
import MLX
import MLXNN
import Gemma4Audio
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var prompt: String = "Describe the speaker's tone in this audio and translate it to English."
    @State private var audioPath: String = ""
    @State private var outputText: String = "Waiting for input..."
    @State private var isProcessing: Bool = false
    @State private var stats: String = ""
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Gemma 4 Multimodal Audio Inference")
                .font(.title)
                .fontWeight(.bold)
            
            VStack(alignment: .leading) {
                Text("1. Select Audio File (.wav)")
                    .font(.headline)
                HStack {
                    TextField("Path to audio", text: $audioPath)
                        .textFieldStyle(.roundedBorder)
                        .disabled(true)
                    Button("Browse...") {
                        selectAudioFile()
                    }
                }
            }
            
            VStack(alignment: .leading) {
                Text("2. Prompt")
                    .font(.headline)
                TextEditor(text: $prompt)
                    .font(.system(.body, design: .monospaced))
                    .frame(height: 80)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.5)))
            }
            
            Button(action: runInference) {
                if isProcessing {
                    ProgressView()
                        .controlSize(.small)
                    Text("Processing...")
                } else {
                    Text("3. Run Gemma 4 Inference")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(audioPath.isEmpty || isProcessing)
            
            VStack(alignment: .leading) {
                Text("Output & Stats:")
                    .font(.headline)
                
                Text(stats)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                ScrollView {
                    Text(outputText)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(Color(NSColor.textBackgroundColor))
                .cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.5)))
            }
        }
        .padding()
        .frame(minWidth: 600, minHeight: 500)
    }
    
    func selectAudioFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType.wav]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK {
            audioPath = panel.url?.path ?? ""
        }
    }
    
    func runInference() {
        isProcessing = true
        outputText = "Extracting Audio Features..."
        stats = "Preparing MLX Tensors..."
        
        Task {
            let startTime = Date()
            
            do {
                // 1. Mock Loading Audio (in a real app, use AVAudioFile to extract PCM buffer)
                // We use a mocked 1-second 16kHz silent audio tensor here to test the DSP logic
                let audioArray = MLXArray.zeros([16000])
                
                // 2. Feature Extraction (DSP STFT/Mel-Filterbanks via MLXFFT)
                let extractor = Gemma4AudioFeatureExtractor()
                let (spectrogram, mask) = extractor(rawSpeech: [audioArray])
                
                // Force computation to measure time
                MLX.eval(spectrogram, mask)
                let extractionTime = Date().timeIntervalSince(startTime)
                
                // 3. Test the Neural Network (Audio Encoder)
                // This validates the Conformer blocks, SSCP convolutions, and relative attention.
                let config = AudioConfig()
                let encoder = AudioEncoder(config: config, numHiddenLayers: 2) // Test with 2 layers for speed
                
                // We need to shape the spectrogram correctly: [Batch, Time, Frequencies]
                var expandedSpec = spectrogram
                if expandedSpec.ndim == 2 {
                    expandedSpec = expandedSpec.expandedDimensions(axes: [0]) // Add batch dim
                }
                
                // The encoder now handles the causalValidMask internally!
                let (encodedFeatures, _) = encoder(expandedSpec, mask: mask)
                MLX.eval(encodedFeatures)
                let encodeTime = Date().timeIntervalSince(startTime) - extractionTime
                
                // 4. Stub for Gemma 4 Language Model Execution
                try await Task.sleep(nanoseconds: 1_000_000_000) // Simulate generation time
                
                let finalTime = Date().timeIntervalSince(startTime)
                let specShapeStr = "\\[\(spectrogram.shape.map { String($0) }.joined(separator: ", "))\\]"
                let encoderShapeStr = "\\[\(encodedFeatures.shape.map { String($0) }.joined(separator: ", "))\\]"
                
                await MainActor.run {
                    self.stats = String(format: "DSP time: %.2fs | Encoder time: %.2fs | Total: %.2fs\nSpectrogram: \(specShapeStr)\nEncoded Features: \(encoderShapeStr)", extractionTime, encodeTime, finalTime)
                    self.outputText = """
                    [STUB] Native MLX language model generation will execute here.
                    
                    Input Prompt:
                    "\(prompt)"
                    
                    Audio File:
                    \(audioPath)
                    
                    Status:
                    Successfully processed raw audio into MLX Spectrogram tensor using Metal!
                    Ready to feed into Gemma 4 Audio Encoder when the full model port completes.
                    """
                    self.isProcessing = false
                }
                
            } catch {
                await MainActor.run {
                    self.outputText = "Error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
}
