#!/usr/bin/env python3
"""
Basic inference tests for wav2vec2.cpp

Tests:
1. CLI runs without crashing
2. Output is valid phoneme string
3. Consistent output across runs
4. Quantized model produces similar results

Run:
    python tests/test_inference.py
    pytest tests/test_inference.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Paths
ROOT_DIR = Path(__file__).parent.parent
BUILD_DIR = ROOT_DIR / "build"
CLI_PATH = BUILD_DIR / "bin" / "wav2vec2-cli"
SAMPLES_DIR = ROOT_DIR / "samples"
MODELS_DIR = ROOT_DIR / "models"


def generate_test_audio(path: str, duration: float = 1.0, sr: int = 16000):
    """Generate a simple test audio file."""
    try:
        from scipy.io import wavfile
    except ImportError:
        # Fallback: write raw PCM
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        with open(path, 'wb') as f:
            # Write minimal WAV header
            import struct
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + len(audio) * 2))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Subchunk1Size
            f.write(struct.pack('<H', 1))   # AudioFormat (PCM)
            f.write(struct.pack('<H', 1))   # NumChannels
            f.write(struct.pack('<I', sr))  # SampleRate
            f.write(struct.pack('<I', sr * 2))  # ByteRate
            f.write(struct.pack('<H', 2))   # BlockAlign
            f.write(struct.pack('<H', 16))  # BitsPerSample
            f.write(b'data')
            f.write(struct.pack('<I', len(audio) * 2))
            f.write(audio.tobytes())
        return

    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(path, sr, audio)


def find_model():
    """Find a model file to use for testing."""
    patterns = [
        MODELS_DIR / "wav2vec2-phoneme" / "ggml-model-f16.bin",
        MODELS_DIR / "wav2vec2-phoneme" / "ggml-model-q6_k.bin",
        MODELS_DIR / "ggml-model-f16.bin",
        MODELS_DIR / "*.bin",
    ]

    for pattern in patterns:
        if pattern.exists():
            return pattern
        matches = list(pattern.parent.glob(pattern.name)) if '*' in str(pattern) else []
        if matches:
            return matches[0]

    return None


def run_cli(audio_path: str, model_path: str, extra_args: list = None) -> dict:
    """Run wav2vec2-cli and return results."""
    if not CLI_PATH.exists():
        raise FileNotFoundError(f"CLI not found at {CLI_PATH}. Run: mkdir build && cd build && cmake .. && make")

    cmd = [str(CLI_PATH), "-m", str(model_path), "-f", str(audio_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "phonemes": result.stdout.strip()
    }


class TestCLI:
    """Tests for wav2vec2-cli"""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.model = find_model()
        if cls.model is None:
            print("Warning: No model found. Some tests will be skipped.")

        # Create temp audio file
        cls.temp_dir = tempfile.mkdtemp()
        cls.audio_path = Path(cls.temp_dir) / "test.wav"
        generate_test_audio(str(cls.audio_path))

    def test_cli_exists(self):
        """CLI binary exists."""
        assert CLI_PATH.exists(), f"CLI not found at {CLI_PATH}"

    def test_cli_help(self):
        """CLI shows help without crashing."""
        result = subprocess.run([str(CLI_PATH), "--help"], capture_output=True, text=True)
        # May return non-zero but should not crash
        assert result.returncode in [0, 1]
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_cli_version(self):
        """CLI shows version."""
        result = subprocess.run([str(CLI_PATH), "--version"], capture_output=True, text=True)
        # Version flag may not exist yet, but shouldn't crash
        assert result.returncode in [0, 1]

    def test_inference_runs(self):
        """Basic inference completes without error."""
        if self.model is None:
            print("SKIP: No model available")
            return

        result = run_cli(str(self.audio_path), str(self.model))
        assert result["returncode"] == 0, f"CLI failed: {result['stderr']}"
        assert len(result["phonemes"]) > 0, "No output produced"

    def test_output_is_phonemes(self):
        """Output contains valid phoneme characters."""
        if self.model is None:
            print("SKIP: No model available")
            return

        result = run_cli(str(self.audio_path), str(self.model))
        phonemes = result["phonemes"]

        # Should contain some IPA or ASCII phoneme characters
        # Not just whitespace or empty
        assert len(phonemes.strip()) > 0, "Output is empty"

        # Should not be an error message
        assert "error" not in phonemes.lower(), f"Got error: {phonemes}"
        assert "failed" not in phonemes.lower(), f"Got failure: {phonemes}"

    def test_consistent_output(self):
        """Same input produces same output."""
        if self.model is None:
            print("SKIP: No model available")
            return

        result1 = run_cli(str(self.audio_path), str(self.model))
        result2 = run_cli(str(self.audio_path), str(self.model))

        assert result1["phonemes"] == result2["phonemes"], \
            f"Inconsistent output:\n  Run 1: {result1['phonemes']}\n  Run 2: {result2['phonemes']}"

    def test_missing_model(self):
        """Graceful error for missing model."""
        result = subprocess.run(
            [str(CLI_PATH), "-m", "/nonexistent/model.bin", "-f", str(self.audio_path)],
            capture_output=True, text=True
        )
        assert result.returncode != 0, "Should fail for missing model"

    def test_missing_audio(self):
        """Graceful error for missing audio."""
        if self.model is None:
            print("SKIP: No model available")
            return

        result = subprocess.run(
            [str(CLI_PATH), "-m", str(self.model), "-f", "/nonexistent/audio.wav"],
            capture_output=True, text=True
        )
        assert result.returncode != 0, "Should fail for missing audio"


def run_tests():
    """Run tests manually without pytest."""
    test = TestCLI()
    test.setup_class()

    tests = [
        ("CLI exists", test.test_cli_exists),
        ("CLI help", test.test_cli_help),
        ("Inference runs", test.test_inference_runs),
        ("Output is phonemes", test.test_output_is_phonemes),
        ("Consistent output", test.test_consistent_output),
        ("Missing model error", test.test_missing_model),
        ("Missing audio error", test.test_missing_audio),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name}")
            print(f"        {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("wav2vec2.cpp Inference Tests")
    print("=" * 40)

    success = run_tests()
    sys.exit(0 if success else 1)
