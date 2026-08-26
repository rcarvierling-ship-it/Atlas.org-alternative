// Lectern system-audio helper.
//
// Captures everything the Mac is playing using ScreenCaptureKit's audio tap
// (macOS 13+) and streams it to Lectern over stdout. This exists because macOS
// has no public API for tapping system output from Python, and because Lectern
// refuses to make users install a virtual audio driver such as BlackHole.
//
// Wire protocol — kept deliberately trivial so the Python side can be tested
// against a stub:
//
//   stdout : raw little-endian Float32 mono PCM at the requested sample rate.
//   stderr : one JSON object per line, {"event": "...", "message": "..."}.
//   exit 13: screen-recording permission was denied.
//
// Usage: lectern-audio-capture [--sample-rate 16000] [--format f32]

import AVFoundation
import CoreGraphics
import Foundation
import ScreenCaptureKit

let permissionDeniedExit: Int32 = 13
let startupFailureExit: Int32 = 1

// ScreenCaptureKit always delivers 48 kHz float PCM; the helper converts down
// to whatever Lectern asked for (16 kHz, to match Whisper's input).
let captureSampleRate: Double = 48_000
let captureChannelCount: AVAudioChannelCount = 2

struct Options {
    var sampleRate: Double = 16_000

    static func parse(_ arguments: [String]) -> Options {
        var options = Options()
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--sample-rate":
                index += 1
                if index < arguments.count, let value = Double(arguments[index]) {
                    options.sampleRate = value
                }
            case "--format":
                // Only f32 is implemented; the flag exists so the protocol can
                // grow without changing Lectern's invocation.
                index += 1
            default:
                break
            }
            index += 1
        }
        return options
    }
}

/// Emit a structured log line on stderr. stdout carries audio only.
func emit(event: String, message: String) {
    let payload: [String: String] = ["event": event, "message": message]
    guard let data = try? JSONSerialization.data(withJSONObject: payload),
          var line = String(data: data, encoding: .utf8) else { return }
    line += "\n"
    FileHandle.standardError.write(Data(line.utf8))
}

/// Write bytes to stdout, tolerating short writes on a pipe.
func writeStdout(_ data: Data) {
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        guard var pointer = raw.baseAddress else { return }
        var remaining = raw.count
        while remaining > 0 {
            let written = write(STDOUT_FILENO, pointer, remaining)
            if written > 0 {
                pointer = pointer.advanced(by: written)
                remaining -= written
            } else if errno == EINTR {
                continue
            } else {
                // Lectern closed the pipe: nothing left to stream to.
                exit(0)
            }
        }
    }
}

final class AudioCapture: NSObject, SCStreamDelegate, SCStreamOutput {
    private let options: Options
    private var stream: SCStream?
    private var converter: AVAudioConverter?
    private let outputFormat: AVAudioFormat
    private let sampleQueue = DispatchQueue(label: "com.lectern.audio-capture")

    init(options: Options) {
        self.options = options
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: options.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            emit(event: "error", message: "could not build the output audio format")
            exit(startupFailureExit)
        }
        self.outputFormat = format
        super.init()
    }

    func start() async {
        // Preflight rather than waiting for SCShareableContent to fail, so the
        // exit code tells Lectern precisely that permission is the problem.
        if !CGPreflightScreenCaptureAccess() {
            CGRequestScreenCaptureAccess()
            emit(event: "error", message: "screen recording permission is not granted")
            exit(permissionDeniedExit)
        }

        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: false
            )
        } catch {
            let text = "\(error)"
            if text.lowercased().contains("declined") || text.lowercased().contains("permission") {
                emit(event: "error", message: "screen recording permission is not granted")
                exit(permissionDeniedExit)
            }
            emit(event: "error", message: "could not query shareable content: \(text)")
            exit(startupFailureExit)
        }

        guard let display = content.displays.first else {
            emit(event: "error", message: "no display available to attach the audio tap to")
            exit(startupFailureExit)
        }

        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.sampleRate = Int(captureSampleRate)
        configuration.channelCount = Int(captureChannelCount)
        // Without this, Lectern's own output (were it ever to play audio) would
        // be captured and fed back into the transcript.
        configuration.excludesCurrentProcessAudio = true
        // Video frames are not used; keep them minimal so the tap is cheap.
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        configuration.queueDepth = 5

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        self.stream = stream

        do {
            try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
            try await stream.startCapture()
        } catch {
            emit(event: "error", message: "could not start the capture stream: \(error)")
            exit(startupFailureExit)
        }

        emit(event: "started", message: "capturing system audio at \(Int(options.sampleRate)) Hz mono")
    }

    func stop() async {
        guard let stream else { return }
        try? await stream.stopCapture()
        emit(event: "stopped", message: "capture stopped")
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferDataIsReady(sampleBuffer) else { return }
        guard let pcmBuffer = makePCMBuffer(from: sampleBuffer) else { return }
        guard let mono = downmixAndResample(pcmBuffer) else { return }
        writeStdout(mono)
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        emit(event: "error", message: "capture stopped unexpectedly: \(error)")
        exit(startupFailureExit)
    }

    // MARK: - Conversion

    /// Wrap a CMSampleBuffer's audio in an AVAudioPCMBuffer without copying twice.
    private func makePCMBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(description)
        else { return nil }

        let format = AVAudioFormat(streamDescription: streamDescription)
        guard let format else { return nil }

        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frameCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
        else { return nil }
        buffer.frameLength = frameCount

        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: buffer.mutableAudioBufferList
        )
        return status == noErr ? buffer : nil
    }

    /// Convert captured stereo 48 kHz audio to mono at the requested rate.
    private func downmixAndResample(_ input: AVAudioPCMBuffer) -> Data? {
        if converter == nil || converter?.inputFormat != input.format {
            converter = AVAudioConverter(from: input.format, to: outputFormat)
            // AVAudioConverter handles both the channel downmix and the sample
            // rate conversion, including the anti-alias filtering that a naive
            // decimation would skip.
            converter?.downmix = true
        }
        guard let converter else { return nil }

        let ratio = outputFormat.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded(.up) + 16)
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            return nil
        }

        var consumed = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, statusPointer in
            if consumed {
                statusPointer.pointee = .noDataNow
                return nil
            }
            consumed = true
            statusPointer.pointee = .haveData
            return input
        }

        if status == .error || output.frameLength == 0 {
            if let conversionError {
                emit(event: "error", message: "audio conversion failed: \(conversionError)")
            }
            return nil
        }

        guard let channel = output.floatChannelData?[0] else { return nil }
        return Data(bytes: channel, count: Int(output.frameLength) * MemoryLayout<Float>.size)
    }
}

// MARK: - Entry point

let options = Options.parse(Array(CommandLine.arguments.dropFirst()))
let capture = AudioCapture(options: options)

// Terminate cleanly when Lectern stops the session.
let signalQueue = DispatchQueue(label: "com.lectern.audio-capture.signals")
for signalNumber in [SIGINT, SIGTERM] {
    signal(signalNumber, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: signalQueue)
    source.setEventHandler {
        Task {
            await capture.stop()
            exit(0)
        }
    }
    source.resume()
    // Keep the source alive for the process lifetime.
    withExtendedLifetime(source) {}
}

Task {
    await capture.start()
}

// ScreenCaptureKit delivers samples on its own queue; park the main thread.
RunLoop.main.run()
