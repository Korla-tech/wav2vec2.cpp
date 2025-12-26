# wav2vec2.cpp

High-performance C/C++ implementation of [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) for phoneme recognition, using the [GGML](https://github.com/ggerganov/ggml) tensor library.

Wav2Vec 2.0 is a self-supervised speech representation learning framework from Facebook AI Research that achieves state-of-the-art results with minimal labeled data.

> **Note:** This project was vibe coded with an AI assistant and draws heavily from [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

## Features

- Plain C/C++ implementation without dependencies
- Apple Silicon first-class support (via Metal)
- Mixed F16/F32 precision
- Quantization support (Q4, Q5, Q6, Q8)
- Phoneme recognition with timing information
- CTC decoding with configurable options

## Quick Start

### Build

```bash
mkdir build && cd build
cmake ..
make -j

# With Metal support (macOS/iOS)
cmake -DGGML_METAL=ON ..
make -j
```

### Convert Model

```bash
# Install dependencies
pip install torch transformers

# Convert HuggingFace model to GGML format
python models/convert-wav2vec2-to-ggml.py \
    facebook/wav2vec2-lv-60-espeak-cv-ft \
    models/wav2vec2-phoneme
```

### Run

```bash
# Basic phoneme recognition
./bin/wav2vec2-cli -m models/wav2vec2-phoneme/ggml-model-f16.bin -f samples/audio.wav

# With timing information
./bin/wav2vec2-cli -m models/wav2vec2-phoneme/ggml-model-f16.bin -f samples/audio.wav --print-timestamps
```

### Quantize

```bash
# Quantize to Q6_K (recommended, ~4x smaller with <5% accuracy loss)
./bin/quantize-wav2vec2 models/wav2vec2-phoneme/ggml-model-f16.bin models/wav2vec2-phoneme/ggml-model-q6_k.bin q6_k
```

## Project Structure

```
wav2vec2.cpp/
├── src/                    # Core library
│   ├── wav2vec2.cpp       # Main implementation
│   ├── wav2vec2-arch.h    # Architecture definitions
│   └── CMakeLists.txt
├── include/
│   └── wav2vec2.h         # Public C API
├── examples/
│   ├── wav2vec2/          # CLI tools
│   │   ├── wav2vec2-cli.cpp
│   │   └── quantize-wav2vec2.cpp
│   ├── common.cpp/h       # Shared utilities
│   └── common-ggml.cpp/h  # GGML utilities
├── models/
│   └── convert-wav2vec2-to-ggml.py
├── ggml/                   # GGML tensor library
└── cmake/
```

## API Usage

```c
#include "wav2vec2.h"

// Initialize
struct wav2vec2_context_params cparams = wav2vec2_context_default_params();
cparams.use_gpu = true;

struct wav2vec2_context * ctx = wav2vec2_init_from_file("model.bin", cparams);

// Run inference
struct wav2vec2_full_params params = wav2vec2_full_default_params();
wav2vec2_full(ctx, params, samples, n_samples);

// Get results
int n_phonemes = wav2vec2_full_n_phonemes(ctx);
for (int i = 0; i < n_phonemes; i++) {
    const char * phoneme = wav2vec2_full_get_phoneme_text(ctx, i);
    int64_t t0 = wav2vec2_full_get_phoneme_t0(ctx, i);
    int64_t t1 = wav2vec2_full_get_phoneme_t1(ctx, i);
    printf("[%lld - %lld] %s\n", t0, t1, phoneme);
}

// Cleanup
wav2vec2_free(ctx);
```

## Evaluation

Tested on [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) accented English speech samples, comparing C++ output against the HuggingFace Python reference implementation.

### Accuracy vs Reference

| Model | PER vs Python | Notes |
|-------|---------------|-------|
| F16 | 1.0% | Near-exact parity with reference |
| Q6_K | 1.4% | +0.4% degradation, 2.2x smaller |
| Q4_K | 1.7% | +0.7% degradation, 3x smaller |

PER = Phoneme Error Rate (edit distance / reference length)

### Model Size

| Quantization | Size | Compression |
|--------------|------|-------------|
| F16 | ~600 MB | 1x |
| Q6_K | ~270 MB | 2.2x |
| Q4_K | ~200 MB | 3x |

Q4_K is recommended for mobile deployment - significant size reduction with minimal accuracy loss.

## References

```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael},
  journal={arXiv preprint arXiv:2006.11477},
  year={2020}
}
```

- [wav2vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
- [Facebook fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)
- [HuggingFace Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)

## Acknowledgments

This project draws heavily from [whisper.cpp](https://github.com/ggerganov/whisper.cpp) by Georgi Gerganov and contributors. The architecture, build system, and many implementation patterns are adapted from that excellent project.

## License

MIT
