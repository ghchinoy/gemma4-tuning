// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXMonitor",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "MLXMonitor",
            path: "Sources"
        ),
    ]
)
