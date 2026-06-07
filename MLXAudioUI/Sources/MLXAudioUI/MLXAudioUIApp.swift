import SwiftUI
import MLX

@main
struct MLXAudioUIApp: App {
    init() {
        // MLX defaults to the GPU automatically on Apple Silicon.
        // Device.withDefaultDevice(...) can be used if specific scoped changes are needed.
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
