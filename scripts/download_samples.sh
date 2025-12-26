#!/bin/bash
# Download sample audio files for testing
# These are short public domain audio clips

set -e

SAMPLES_DIR="$(dirname "$0")/../samples"
mkdir -p "$SAMPLES_DIR"

cd "$SAMPLES_DIR"

echo "Downloading sample audio files..."

# JFK "Ask not" speech excerpt (public domain)
if [ ! -f "jfk.wav" ]; then
    echo "Downloading jfk.wav..."
    curl -L -o jfk.wav "https://upload.wikimedia.org/wikipedia/commons/e/e0/En-us-president.ogg" 2>/dev/null || \
    curl -L -o jfk.wav "https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav" 2>/dev/null || \
    echo "Could not download jfk.wav - create manually or use whisper.cpp samples"
fi

# Generate a simple test tone if downloads fail
if [ ! -f "test_tone.wav" ]; then
    echo "Generating test tone..."
    python3 -c "
import numpy as np
try:
    from scipy.io import wavfile
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Simple multi-frequency signal
    audio = 0.3 * np.sin(2*np.pi*440*t) + 0.2 * np.sin(2*np.pi*880*t)
    audio = (audio * 32767).astype(np.int16)
    wavfile.write('test_tone.wav', sr, audio)
    print('Created test_tone.wav')
except ImportError:
    print('scipy not installed - skipping test tone generation')
" 2>/dev/null || echo "Skipping test tone (scipy not installed)"
fi

# Copy from whisper.cpp if available
WHISPER_SAMPLES="$HOME/whisper.cpp/samples"
if [ -d "$WHISPER_SAMPLES" ]; then
    echo "Copying samples from whisper.cpp..."
    for f in "$WHISPER_SAMPLES"/*.wav; do
        if [ -f "$f" ]; then
            base=$(basename "$f")
            if [ ! -f "$base" ]; then
                cp "$f" .
                echo "  Copied $base"
            fi
        fi
    done
fi

echo ""
echo "Sample files in $SAMPLES_DIR:"
ls -la *.wav 2>/dev/null || echo "  (no .wav files found)"
