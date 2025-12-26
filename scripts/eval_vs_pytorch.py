#!/usr/bin/env python3
"""
Evaluate wav2vec2.cpp output against PyTorch reference implementation.

This script compares the C++ port against the original PyTorch implementation
to validate correctness. It does NOT compare against ground truth transcriptions.

Usage:
    python scripts/eval_vs_pytorch.py \
        --audio samples/jfk.wav \
        --cpp-model models/wav2vec2-phoneme/ggml-model-f16.bin \
        --pytorch-model facebook/wav2vec2-lv-60-espeak-cv-ft

    # Batch evaluation
    python scripts/eval_vs_pytorch.py \
        --audio-dir samples/ \
        --cpp-model models/wav2vec2-phoneme/ggml-model-f16.bin \
        --output results/eval.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Try to import optional dependencies
try:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch/transformers not installed. Install with: pip install torch transformers")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Install with: pip install librosa")


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to 16kHz mono."""
    if HAS_LIBROSA:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32)
    else:
        # Fallback: use scipy
        from scipy.io import wavfile
        sample_rate, audio = wavfile.read(path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sample_rate != sr:
            # Simple resampling
            ratio = sr / sample_rate
            new_length = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )
        return audio.astype(np.float32) / 32768.0


def run_pytorch(
    audio: np.ndarray,
    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
) -> dict:
    """Run PyTorch inference and return results."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

    return {
        "phonemes": transcription,
        "logits_shape": list(logits.shape),
        "logits_sample": logits[0, :5, :5].tolist(),  # First 5x5 for comparison
        "predicted_ids": predicted_ids[0].tolist()
    }


def run_cpp(audio_path: str, model_path: str, cli_path: str = "./build/bin/wav2vec2-cli") -> dict:
    """Run wav2vec2.cpp inference and return results."""
    cli = Path(cli_path)
    if not cli.exists():
        # Try alternate locations
        for alt in ["./bin/wav2vec2-cli", "./wav2vec2-cli", "build/bin/wav2vec2-cli"]:
            if Path(alt).exists():
                cli = Path(alt)
                break

    if not cli.exists():
        raise FileNotFoundError(f"wav2vec2-cli not found at {cli_path}")

    result = subprocess.run(
        [str(cli), "-m", model_path, "-f", audio_path, "-oj"],  # -oj for JSON output
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        # Try without JSON flag
        result = subprocess.run(
            [str(cli), "-m", model_path, "-f", audio_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        # Parse plain text output
        return {
            "phonemes": result.stdout.strip(),
            "raw_output": result.stdout
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "phonemes": result.stdout.strip(),
            "raw_output": result.stdout
        }


def edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def phoneme_error_rate(hypothesis: str, reference: str) -> float:
    """Calculate Phoneme Error Rate."""
    # Remove spaces for character-level comparison
    h = hypothesis.replace(" ", "").strip()
    r = reference.replace(" ", "").strip()

    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0

    return edit_distance(h, r) / len(r)


def compare_outputs(pytorch_result: dict, cpp_result: dict) -> dict:
    """Compare PyTorch and C++ outputs."""
    pytorch_phonemes = pytorch_result.get("phonemes", "")
    cpp_phonemes = cpp_result.get("phonemes", "")

    per = phoneme_error_rate(cpp_phonemes, pytorch_phonemes)
    exact_match = cpp_phonemes.strip() == pytorch_phonemes.strip()

    return {
        "pytorch_phonemes": pytorch_phonemes,
        "cpp_phonemes": cpp_phonemes,
        "per": per,
        "exact_match": exact_match,
        "edit_distance": edit_distance(
            cpp_phonemes.replace(" ", ""),
            pytorch_phonemes.replace(" ", "")
        ),
        "pytorch_length": len(pytorch_phonemes.replace(" ", "")),
        "cpp_length": len(cpp_phonemes.replace(" ", ""))
    }


def evaluate_single(
    audio_path: str,
    cpp_model_path: str,
    pytorch_model: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
    cli_path: str = "./build/bin/wav2vec2-cli"
) -> dict:
    """Evaluate a single audio file."""
    print(f"Evaluating: {audio_path}")

    # Load audio
    audio = load_audio(audio_path)
    print(f"  Audio: {len(audio)/16000:.2f}s @ 16kHz")

    # Run PyTorch
    print("  Running PyTorch reference...")
    pytorch_result = run_pytorch(audio, pytorch_model)
    print(f"  PyTorch: {pytorch_result['phonemes'][:50]}...")

    # Run C++
    print("  Running wav2vec2.cpp...")
    cpp_result = run_cpp(audio_path, cpp_model_path, cli_path)
    print(f"  C++:     {cpp_result['phonemes'][:50]}...")

    # Compare
    comparison = compare_outputs(pytorch_result, cpp_result)

    status = "PASS" if comparison["per"] < 0.05 else "WARN" if comparison["per"] < 0.10 else "FAIL"
    print(f"  PER: {comparison['per']:.2%} [{status}]")
    print(f"  Exact match: {comparison['exact_match']}")

    return {
        "audio_path": audio_path,
        "audio_duration": len(audio) / 16000,
        **comparison
    }


def evaluate_directory(
    audio_dir: str,
    cpp_model_path: str,
    pytorch_model: str,
    cli_path: str,
    extensions: tuple = (".wav", ".flac", ".mp3", ".m4a")
) -> list:
    """Evaluate all audio files in a directory."""
    audio_dir = Path(audio_dir)
    results = []

    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    audio_files = sorted(audio_files)
    print(f"Found {len(audio_files)} audio files")

    for audio_path in audio_files:
        try:
            result = evaluate_single(
                str(audio_path),
                cpp_model_path,
                pytorch_model,
                cli_path
            )
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "audio_path": str(audio_path),
                "error": str(e)
            })

    return results


def summarize_results(results: list) -> dict:
    """Generate summary statistics."""
    valid_results = [r for r in results if "per" in r]

    if not valid_results:
        return {"error": "No valid results"}

    pers = [r["per"] for r in valid_results]
    exact_matches = sum(1 for r in valid_results if r["exact_match"])

    return {
        "total_files": len(results),
        "valid_files": len(valid_results),
        "errors": len(results) - len(valid_results),
        "mean_per": sum(pers) / len(pers),
        "max_per": max(pers),
        "min_per": min(pers),
        "exact_match_rate": exact_matches / len(valid_results),
        "pass_rate_5pct": sum(1 for p in pers if p < 0.05) / len(pers),
        "pass_rate_10pct": sum(1 for p in pers if p < 0.10) / len(pers)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate wav2vec2.cpp against PyTorch reference"
    )
    parser.add_argument(
        "--audio", "-a",
        help="Single audio file to evaluate"
    )
    parser.add_argument(
        "--audio-dir", "-d",
        help="Directory of audio files to evaluate"
    )
    parser.add_argument(
        "--cpp-model", "-m",
        required=True,
        help="Path to wav2vec2.cpp GGML model"
    )
    parser.add_argument(
        "--pytorch-model",
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="HuggingFace model ID for PyTorch reference"
    )
    parser.add_argument(
        "--cli-path",
        default="./build/bin/wav2vec2-cli",
        help="Path to wav2vec2-cli executable"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for results"
    )
    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        parser.error("Specify --audio or --audio-dir")

    # Run evaluation
    if args.audio:
        results = [evaluate_single(
            args.audio,
            args.cpp_model,
            args.pytorch_model,
            args.cli_path
        )]
    else:
        results = evaluate_directory(
            args.audio_dir,
            args.cpp_model,
            args.pytorch_model,
            args.cli_path
        )

    # Generate summary
    summary = summarize_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: wav2vec2.cpp vs PyTorch Reference")
    print("=" * 60)
    print(f"Files evaluated: {summary.get('valid_files', 0)}/{summary.get('total_files', 0)}")
    print(f"Mean PER:        {summary.get('mean_per', 0):.2%}")
    print(f"Exact match:     {summary.get('exact_match_rate', 0):.1%}")
    print(f"Pass (<5% PER):  {summary.get('pass_rate_5pct', 0):.1%}")
    print(f"Pass (<10% PER): {summary.get('pass_rate_10pct', 0):.1%}")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "summary": summary,
            "results": results,
            "config": {
                "cpp_model": args.cpp_model,
                "pytorch_model": args.pytorch_model
            }
        }

        # Also add top-level per for CI checks
        output_data["per"] = summary.get("mean_per", 1.0)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Exit with error if PER too high
    mean_per = summary.get("mean_per", 1.0)
    if mean_per > 0.10:
        print(f"\nFAIL: Mean PER {mean_per:.2%} exceeds 10% threshold")
        sys.exit(1)
    else:
        print(f"\nPASS: Mean PER {mean_per:.2%} within acceptable range")


if __name__ == "__main__":
    main()
