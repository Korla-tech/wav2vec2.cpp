# Sample Audio Files

Audio files for testing wav2vec2.cpp inference.

## Files

| File | Duration | Description |
|------|----------|-------------|
| `jfk.wav` | ~11s | JFK "Ask not what your country can do for you" speech excerpt |
| `test_tone.wav` | 2s | Generated test tone (440Hz + 880Hz) |

## Requirements

- 16-bit PCM WAV format
- 16kHz sample rate
- Mono channel

## Download More Samples

```bash
./scripts/download_samples.sh
```

## Source

Sample audio from [whisper.cpp](https://github.com/ggerganov/whisper.cpp) test samples.
