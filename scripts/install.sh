#!/usr/bin/env bash
# Install Lectern so that typing `lectern` in any terminal starts it.
#
#   ./scripts/install.sh
#
# Installs the `lectern` command onto your PATH and offers to install the local
# AI pieces it needs. Nothing is installed without you saying yes first.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
dim()  { printf '\033[2m%s\033[0m\n' "$1"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '\033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '\033[31m✗\033[0m %s\n' "$1"; }

# Ask a yes/no question. Defaults to no, and to no when not interactive.
confirm() {
  local prompt="$1"
  if [[ ! -t 0 ]]; then
    dim "   (not an interactive terminal — skipping: $prompt)"
    return 1
  fi
  read -r -p "   $prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

echo
bold "Installing Lectern"
echo

# --- 1. uv ------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  warn "uv is not installed (Lectern uses it to manage its Python environment)."

  # Homebrew first: it verifies what it downloads, which piping a script
  # straight into a shell cannot do.
  if command -v brew >/dev/null 2>&1; then
    echo "   Install command: brew install uv"
    if confirm "Run it now?"; then
      brew install uv
    else
      bad "uv is required. Install it and re-run this script."
      exit 1
    fi
  else
    # No Homebrew. Fall back to Astral's installer, but download it to a file
    # first so it can be read before it runs — never curl | sh, which executes
    # whatever the network returned, unseen.
    installer="$(mktemp -t lectern-uv-install)"
    trap 'rm -f "$installer"' EXIT
    echo "   Downloading https://astral.sh/uv/install.sh"
    if ! curl -fLsS https://astral.sh/uv/install.sh -o "$installer"; then
      bad "Could not download the uv installer. Install uv yourself: https://docs.astral.sh/uv/"
      exit 1
    fi
    printf '   Saved to %s (%s lines)\n' "$installer" "$(wc -l < "$installer" | tr -d ' ')"
    if confirm "Show it before running?"; then
      "${PAGER:-less}" "$installer" || cat "$installer"
    fi
    if confirm "Run this downloaded installer?"; then
      sh "$installer"
    else
      bad "uv is required. Install it and re-run this script."
      exit 1
    fi
  fi

  # uv installs to ~/.local/bin; make it visible to the rest of this script.
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# --- 2. the lectern command -------------------------------------------------
# Installed as a uv tool: its own isolated environment, one command on PATH.
# --editable keeps this checkout as the source, so `git pull` updates Lectern.
echo
dim "Installing the 'lectern' command…"
uv tool install --force --editable "$REPO_DIR" --with sounddevice >/dev/null 2>&1 || {
  # sounddevice needs PortAudio; without it Lectern still runs, minus the mic.
  warn "Could not install microphone support (PortAudio missing?). Installing without it."
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "   Fix with: brew install portaudio && ./scripts/install.sh"
  fi
  uv tool install --force --editable "$REPO_DIR" >/dev/null
}
ok "lectern command installed"

# --- 3. PATH ----------------------------------------------------------------
if ! command -v lectern >/dev/null 2>&1; then
  warn "The install directory is not on your PATH yet."
  uv tool update-shell >/dev/null 2>&1 || true
  echo
  echo "   Add this line to your shell profile (~/.zshrc for zsh, ~/.bashrc for bash):"
  bold '     export PATH="$HOME/.local/bin:$PATH"'
  echo
  echo "   Then open a new terminal, or run:  source ~/.zshrc"
else
  ok "'lectern' is on your PATH ($(command -v lectern))"
fi

# --- 4. local AI ------------------------------------------------------------
echo
bold "Local AI"

if [[ "$(uname -s)" == "Darwin" ]] && ! command -v brew >/dev/null 2>&1; then
  warn "Homebrew is not installed; skipping the automatic offers below."
  echo "   See https://brew.sh, then re-run this script."
fi

if ! command -v whisper-server >/dev/null 2>&1; then
  warn "whisper.cpp is not installed (needed to transcribe speech)."
  if command -v brew >/dev/null 2>&1 && confirm "Install it with 'brew install whisper-cpp'?"; then
    brew install whisper-cpp
    ok "whisper.cpp installed"
  else
    echo "   Install later with: brew install whisper-cpp"
  fi
else
  ok "whisper.cpp ($(command -v whisper-server))"
fi

if command -v lectern >/dev/null 2>&1; then
  MODEL="$(lectern config path >/dev/null 2>&1 && echo small.en || echo small.en)"
  if ! lectern models whisper 2>/dev/null | grep -q "installed"; then
    warn "No Whisper model is downloaded yet (~466 MB for $MODEL)."
    if confirm "Download $MODEL now?"; then
      lectern models whisper --download "$MODEL"
    else
      echo "   Download later with: lectern models whisper --download $MODEL"
    fi
  else
    ok "Whisper model present"
  fi
fi

if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama is not installed (needed to generate notes)."
  if command -v brew >/dev/null 2>&1 && confirm "Install it with 'brew install ollama'?"; then
    brew install ollama
    ok "Ollama installed"
  else
    echo "   Install later from https://ollama.com, or: brew install ollama"
  fi
else
  ok "Ollama ($(command -v ollama))"
fi

# Lectern starts the Ollama daemon itself, but it cannot invent a model.
if command -v ollama >/dev/null 2>&1; then
  if ! ollama list 2>/dev/null | tail -n +2 | grep -q .; then
    warn "Ollama has no models installed (a few GB to download)."
    if confirm "Pull qwen3:8b now?"; then
      ollama pull qwen3:8b
      command -v lectern >/dev/null 2>&1 && lectern config set ollama.notes_model=qwen3:8b >/dev/null
      command -v lectern >/dev/null 2>&1 && lectern config set ollama.final_model=qwen3:8b >/dev/null
      ok "qwen3:8b ready and selected"
    else
      echo "   Pull one later with: ollama pull qwen3:8b"
    fi
  else
    ok "Ollama has models installed"
  fi
fi

# --- 5. done ----------------------------------------------------------------
echo
bold "Done"
echo
echo "   Start Lectern by typing:"
bold "     lectern"
echo
echo "   It starts Ollama and whisper.cpp for you — nothing else to launch."
dim  "   Check your setup any time with: lectern doctor"
echo
