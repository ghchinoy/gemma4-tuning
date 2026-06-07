// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "MLXAudioInfer",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.11.0"),
        .package(path: "../../mlx-swift-examples")
    ],
    targets: [
        .executableTarget(
            name: "MLXAudioInfer",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Gemma4Audio", package: "mlx-swift-examples")
            ]
        ),
    ]
)
