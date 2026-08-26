// swift-tools-version:5.9
import PackageDescription

// ScreenCaptureKit audio capture requires macOS 13 (Ventura) or newer.
let package = Package(
    name: "lectern-audio-capture",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "lectern-audio-capture",
            path: "Sources/LecternAudioCapture"
        )
    ]
)
