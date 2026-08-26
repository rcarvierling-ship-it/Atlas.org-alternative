#!/usr/bin/env bash
# Build the Swift ScreenCaptureKit helper that captures macOS system audio.
#
# Only needed for the "System audio" and "Microphone + system audio" sources.
# Microphone recording works without it.
#
# Requirements: macOS 13+, Xcode command line tools (xcode-select --install).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$HERE/../native/audio-capture"
OUTPUT_DIR="$PROJECT/build"
BINARY_NAME="lectern-audio-capture"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: the system-audio helper is macOS-only (found $(uname -s))." >&2
  echo "       Microphone capture works on this machine without it." >&2
  exit 1
fi

MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if (( MACOS_MAJOR < 13 )); then
  echo "error: ScreenCaptureKit audio capture needs macOS 13 or newer (found $(sw_vers -productVersion))." >&2
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "error: 'swift' not found. Install the Xcode command line tools:" >&2
  echo "       xcode-select --install" >&2
  exit 1
fi

echo "Building $BINARY_NAME…"
cd "$PROJECT"
swift build -c release

mkdir -p "$OUTPUT_DIR"
cp "$(swift build -c release --show-bin-path)/$BINARY_NAME" "$OUTPUT_DIR/$BINARY_NAME"
chmod +x "$OUTPUT_DIR/$BINARY_NAME"

echo
echo "Built: $OUTPUT_DIR/$BINARY_NAME"
echo
echo "Next: the first time you select a system-audio source, macOS will ask for"
echo "Screen & System Audio Recording permission for your terminal app."
echo "Grant it in System Settings → Privacy & Security, then restart the terminal —"
echo "macOS only applies the permission to newly launched processes."
