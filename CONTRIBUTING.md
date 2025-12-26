# Contributing to wav2vec2.cpp

Thank you for your interest in contributing! This document provides guidelines for contributing to wav2vec2.cpp.

## Code of Conduct

Be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/wav2vec2.cpp`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Test thoroughly
6. Submit a pull request

## Development Setup

```bash
# Build
mkdir build && cd build
cmake -DGGML_METAL=ON ..
make -j

# Test
./bin/wav2vec2-cli -m models/wav2vec2-phoneme/ggml-model-f16.bin -f samples/jfk.wav
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code compiles without warnings
- [ ] Existing tests pass
- [ ] New functionality includes tests
- [ ] Accuracy metrics unchanged (run `scripts/eval_pytorch.py`)
- [ ] Code follows project style (see [CONVENTIONS.md](docs/CONVENTIONS.md))

### PR Format

**Title:** `<scope>: <description>`

Examples:
- `wav2vec2: fix attention mask handling`
- `examples: add streaming inference`
- `docs: update build instructions`

**Description:**
- What does this PR do?
- Why is this change needed?
- How was it tested?

### One PR Per Change

Submit separate PRs for unrelated changes. Don't mix bug fixes with new features.

## Code Style

- 4-space indentation
- 120 character line limit
- `snake_case` for functions and variables
- `wav2vec2_` prefix for public API functions
- See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) for details

## AI-Generated Code

If you use AI tools (GitHub Copilot, ChatGPT, Claude, etc.) to generate code, please disclose this in your PR description. This project was itself "vibe coded" with AI assistance, so AI-assisted contributions are welcome with transparency.

## Testing

### Unit Tests

```bash
cd build && ctest
```

### Accuracy Validation

Compare against PyTorch reference:

```bash
python scripts/eval_pytorch.py \
    --audio samples/jfk.wav \
    --model models/wav2vec2-phoneme/ggml-model-f16.bin
```

Expected: PER < 5% vs PyTorch, or improvement over current baseline.

## Reporting Issues

### Bug Reports

Include:
- Platform (macOS/Linux/Windows, CPU/GPU)
- Model used
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs

### Feature Requests

Describe:
- The problem you're trying to solve
- Your proposed solution
- Alternatives considered

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
