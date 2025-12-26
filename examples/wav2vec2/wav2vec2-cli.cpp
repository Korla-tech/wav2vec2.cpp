// wav2vec2-cli.cpp - Command-line interface for wav2vec2 phoneme recognition
//
// Usage: wav2vec2-cli -m model.bin audio.wav

#include "common.h"
#include "wav2vec2.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Command-line parameters
struct wav2vec2_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    bool use_gpu         = true;
    bool print_timestamps = true;
    bool print_probs     = false;
    bool merge_repeated  = true;
    bool suppress_blank  = true;
    bool test_streaming  = false;  // Test streaming parity against batch

    // Streaming test parameters (growing window mode)
    int32_t stream_max_window_ms = 10000;  // Max window size before sliding (10s)

    std::string model    = "models/wav2vec2-phoneme/ggml-model-q6_k.bin";
    std::vector<std::string> fname_inp = {};
};

static void wav2vec2_print_usage(int argc, char ** argv, const wav2vec2_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0.wav [file1.wav ...]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help             [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N        [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME      [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -ng,      --no-gpu           [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -nt,      --no-timestamps    [%-7s] disable timestamps\n", params.print_timestamps ? "false" : "true");
    fprintf(stderr, "  -pp,      --print-probs      [%-7s] print token probabilities\n", params.print_probs ? "true" : "false");
    fprintf(stderr, "  -nm,      --no-merge         [%-7s] don't merge repeated phonemes\n", !params.merge_repeated ? "true" : "false");
    fprintf(stderr, "  -nb,      --no-blank         [%-7s] don't suppress blank tokens\n", !params.suppress_blank ? "true" : "false");
    fprintf(stderr, "  -ts,      --test-streaming   [%-7s] test streaming parity against batch mode\n", params.test_streaming ? "true" : "false");
    fprintf(stderr, "  -sw N,    --stream-window N  [%-7d] streaming max window size in ms\n", params.stream_max_window_ms);
    fprintf(stderr, "\n");
}

static bool wav2vec2_params_parse(int argc, char ** argv, wav2vec2_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            wav2vec2_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"  || arg == "--threads")        { params.n_threads        = std::stoi(argv[++i]); }
        else if (arg == "-m"  || arg == "--model")          { params.model            = argv[++i]; }
        else if (arg == "-ng" || arg == "--no-gpu")         { params.use_gpu          = false; }
        else if (arg == "-nt" || arg == "--no-timestamps")  { params.print_timestamps = false; }
        else if (arg == "-pp" || arg == "--print-probs")    { params.print_probs      = true; }
        else if (arg == "-nm" || arg == "--no-merge")       { params.merge_repeated   = false; }
        else if (arg == "-nb" || arg == "--no-blank")       { params.suppress_blank   = false; }
        else if (arg == "-ts" || arg == "--test-streaming") { params.test_streaming   = true; }
        else if (arg == "-sw" || arg == "--stream-window")  { params.stream_max_window_ms = std::stoi(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            wav2vec2_print_usage(argc, argv, params);
            exit(1);
        }
    }

    return true;
}

// Simple WAV file reader (16-bit mono only)
static bool read_wav(const std::string & fname, std::vector<float> & pcmf32, int & sample_rate) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        fprintf(stderr, "error: failed to open '%s'\n", fname.c_str());
        return false;
    }

    // Read WAV header
    char riff[4];
    file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "error: '%s' is not a valid WAV file (no RIFF header)\n", fname.c_str());
        return false;
    }

    uint32_t chunk_size;
    file.read((char *) &chunk_size, 4);

    char wave[4];
    file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "error: '%s' is not a valid WAV file (no WAVE header)\n", fname.c_str());
        return false;
    }

    // Find fmt chunk
    while (file) {
        char chunk_id[4];
        uint32_t chunk_size;
        file.read(chunk_id, 4);
        file.read((char *) &chunk_size, 4);

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format;
            uint16_t num_channels;
            uint32_t sr;
            uint32_t byte_rate;
            uint16_t block_align;
            uint16_t bits_per_sample;

            file.read((char *) &audio_format, 2);
            file.read((char *) &num_channels, 2);
            file.read((char *) &sr, 4);
            file.read((char *) &byte_rate, 4);
            file.read((char *) &block_align, 2);
            file.read((char *) &bits_per_sample, 2);

            sample_rate = sr;

            if (audio_format != 1) {
                fprintf(stderr, "error: '%s' is not PCM format\n", fname.c_str());
                return false;
            }

            if (num_channels != 1) {
                fprintf(stderr, "error: '%s' is not mono (has %d channels)\n", fname.c_str(), num_channels);
                return false;
            }

            if (bits_per_sample != 16) {
                fprintf(stderr, "error: '%s' is not 16-bit (has %d bits)\n", fname.c_str(), bits_per_sample);
                return false;
            }

            // Skip any extra bytes in fmt chunk
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            // Read audio data
            std::vector<int16_t> pcm16(chunk_size / 2);
            file.read((char *) pcm16.data(), chunk_size);

            // Convert to float
            pcmf32.resize(pcm16.size());
            for (size_t i = 0; i < pcm16.size(); i++) {
                pcmf32[i] = (float) pcm16[i] / 32768.0f;
            }

            return true;
        }
        else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    fprintf(stderr, "error: '%s' has no data chunk\n", fname.c_str());
    return false;
}

// Structure to hold phoneme result for comparison
struct phoneme_result {
    wav2vec2_token id;
    int64_t t0;
    int64_t t1;
    std::string text;
};

// Resample audio to 16kHz if needed
static void resample_16khz(std::vector<float> & pcmf32, int sample_rate) {
    if (sample_rate == WAV2VEC2_SAMPLE_RATE) {
        return;  // Already 16kHz
    }

    // Simple linear interpolation resampling
    double ratio = (double) WAV2VEC2_SAMPLE_RATE / sample_rate;
    size_t new_size = (size_t) (pcmf32.size() * ratio);

    std::vector<float> resampled(new_size);

    for (size_t i = 0; i < new_size; i++) {
        double src_idx = i / ratio;
        size_t idx0 = (size_t) src_idx;
        size_t idx1 = std::min(idx0 + 1, pcmf32.size() - 1);
        double frac = src_idx - idx0;

        resampled[i] = (float) ((1.0 - frac) * pcmf32[idx0] + frac * pcmf32[idx1]);
    }

    pcmf32 = std::move(resampled);
}

// Test streaming parity against batch mode
static bool test_streaming_parity(
        struct wav2vec2_context * ctx,
        const std::vector<float> & pcmf32,
        const wav2vec2_params & params) {

    fprintf(stderr, "\n=== Streaming Parity Test ===\n");

    // First, get batch results
    fprintf(stderr, "Running batch mode...\n");

    struct wav2vec2_full_params wparams = wav2vec2_full_default_params();
    wparams.n_threads = params.n_threads;
    wparams.merge_repeated = params.merge_repeated;
    wparams.blank_suppress = params.suppress_blank;
    wparams.token_timestamps = true;

    if (wav2vec2_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stderr, "error: batch mode failed\n");
        return false;
    }

    // Collect batch results
    std::vector<phoneme_result> batch_results;
    int n_batch = wav2vec2_full_n_phonemes(ctx);
    for (int i = 0; i < n_batch; i++) {
        wav2vec2_phoneme_data data = wav2vec2_full_get_phoneme_data(ctx, i);
        const char * text = wav2vec2_full_get_phoneme_text(ctx, i);
        batch_results.push_back({data.id, data.t0, data.t1, text ? text : ""});
    }

    fprintf(stderr, "  Batch: %d phonemes\n", n_batch);

    // Now run streaming mode with growing window
    fprintf(stderr, "Running streaming mode (max_window=%dms)...\n",
            params.stream_max_window_ms);

    struct wav2vec2_stream_params sparams = wav2vec2_stream_default_params();
    sparams.n_threads = params.n_threads;
    sparams.max_window_samples = (params.stream_max_window_ms * WAV2VEC2_SAMPLE_RATE) / 1000;
    sparams.merge_repeated = params.merge_repeated;
    sparams.blank_suppress = params.suppress_blank;

    struct wav2vec2_stream_state * stream = wav2vec2_stream_init(ctx, sparams);
    if (!stream) {
        fprintf(stderr, "error: failed to init streaming state\n");
        return false;
    }

    // Simulate streaming by feeding audio in small chunks (e.g., 500ms at a time)
    int feed_chunk_samples = WAV2VEC2_SAMPLE_RATE / 2;  // 500ms
    int offset = 0;
    int total_new = 0;

    while (offset < (int) pcmf32.size()) {
        int n_to_feed = std::min(feed_chunk_samples, (int) pcmf32.size() - offset);

        int n_new = wav2vec2_stream_process(ctx, stream, pcmf32.data() + offset, n_to_feed);
        if (n_new < 0) {
            fprintf(stderr, "error: streaming process failed\n");
            wav2vec2_stream_free(stream);
            return false;
        }
        total_new += n_new;

        offset += n_to_feed;
    }

    // Finalize streaming
    int n_final = wav2vec2_stream_finalize(ctx, stream);
    if (n_final < 0) {
        fprintf(stderr, "error: streaming finalize failed\n");
        wav2vec2_stream_free(stream);
        return false;
    }
    total_new += n_final;

    // Get all accumulated phonemes for comparison
    std::vector<phoneme_result> stream_results;
    int n_all = wav2vec2_stream_n_all_phonemes(stream);
    for (int i = 0; i < n_all; i++) {
        wav2vec2_phoneme_data data = wav2vec2_stream_get_all_phoneme_data(stream, i);
        const char * text = wav2vec2_stream_get_all_phoneme_text(ctx, stream, i);
        stream_results.push_back({data.id, data.t0, data.t1, text ? text : ""});
    }

    wav2vec2_stream_free(stream);

    fprintf(stderr, "  Stream: %d phonemes (new phonemes emitted: %d)\n",
            (int) stream_results.size(), total_new);

    // Compare results using sequence alignment to handle insertions/deletions at chunk boundaries
    fprintf(stderr, "\nComparing results...\n");

    // Build full phoneme sequences
    std::string batch_seq, stream_seq;
    for (const auto & p : batch_results) batch_seq += p.text;
    for (const auto & p : stream_results) stream_seq += p.text;

    // Compute edit distance (simple character-level)
    int edit_distance = 0;
    {
        int m = (int) batch_seq.size();
        int n = (int) stream_seq.size();
        std::vector<int> prev(n + 1), curr(n + 1);
        for (int j = 0; j <= n; j++) prev[j] = j;
        for (int i = 1; i <= m; i++) {
            curr[0] = i;
            for (int j = 1; j <= n; j++) {
                if (batch_seq[i-1] == stream_seq[j-1]) {
                    curr[j] = prev[j-1];
                } else {
                    curr[j] = 1 + std::min({prev[j-1], prev[j], curr[j-1]});
                }
            }
            std::swap(prev, curr);
        }
        edit_distance = prev[n];
    }

    // Calculate similarity as percentage
    int max_len = std::max(batch_seq.size(), stream_seq.size());
    float similarity = max_len > 0 ? 100.0f * (1.0f - (float)edit_distance / max_len) : 100.0f;

    // Count exact phoneme matches
    int n_compare = std::min(batch_results.size(), stream_results.size());
    int mismatches = 0;

    for (int i = 0; i < n_compare; i++) {
        if (batch_results[i].id != stream_results[i].id) {
            if (mismatches < 10) {
                fprintf(stderr, "  MISMATCH[%d]: batch='%s'(%d) vs stream='%s'(%d)\n",
                        i, batch_results[i].text.c_str(), batch_results[i].id,
                        stream_results[i].text.c_str(), stream_results[i].id);
            }
            mismatches++;
        }
    }

    if (mismatches > 10) {
        fprintf(stderr, "  ... and %d more mismatches\n", mismatches - 10);
    }

    // Report results
    fprintf(stderr, "\n  Phoneme count: batch=%zu, stream=%zu\n",
            batch_results.size(), stream_results.size());
    fprintf(stderr, "  Sequence similarity: %.1f%% (edit distance: %d chars)\n",
            similarity, edit_distance);
    fprintf(stderr, "  Exact position matches: %d/%d (%.1f%%)\n",
            n_compare - mismatches, n_compare, 100.0f * (n_compare - mismatches) / n_compare);

    // Pass if similarity is >= 90% (allows for chunk boundary effects)
    bool passed = (similarity >= 90.0f);

    // Print summary
    fprintf(stderr, "\n=== Streaming Parity Test: %s ===\n", passed ? "PASSED" : "FAILED");

    if (passed) {
        fprintf(stderr, "  Streaming matches batch with %.1f%% sequence similarity\n", similarity);
        fprintf(stderr, "  (Minor differences expected at chunk boundaries due to transformer context)\n");
    }

    // Print sample output comparison
    fprintf(stderr, "\nSample output (first 10 phonemes):\n");
    fprintf(stderr, "  Batch:  ");
    for (int i = 0; i < std::min(10, (int) batch_results.size()); i++) {
        fprintf(stderr, "%s", batch_results[i].text.c_str());
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "  Stream: ");
    for (int i = 0; i < std::min(10, (int) stream_results.size()); i++) {
        fprintf(stderr, "%s", stream_results[i].text.c_str());
    }
    fprintf(stderr, "\n");

    // Print full sequences for comparison
    fprintf(stderr, "\nFull sequences:\n");
    fprintf(stderr, "  Batch (%zu): ", batch_results.size());
    for (size_t i = 0; i < batch_results.size(); i++) {
        fprintf(stderr, "%s", batch_results[i].text.c_str());
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "  Stream(%zu): ", stream_results.size());
    for (size_t i = 0; i < stream_results.size(); i++) {
        fprintf(stderr, "%s", stream_results[i].text.c_str());
    }
    fprintf(stderr, "\n");

    return passed;
}

int main(int argc, char ** argv) {
    wav2vec2_params params;

    if (!wav2vec2_params_parse(argc, argv, params)) {
        return 1;
    }

    if (params.fname_inp.empty()) {
        fprintf(stderr, "error: no input files specified\n");
        wav2vec2_print_usage(argc, argv, params);
        return 1;
    }

    // Initialize model
    struct wav2vec2_context_params cparams = wav2vec2_context_default_params();
    cparams.use_gpu = params.use_gpu;

    fprintf(stderr, "wav2vec2_init_from_file: loading model from '%s'\n", params.model.c_str());

    struct wav2vec2_context * ctx = wav2vec2_init_from_file(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to load model\n");
        return 1;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads, std::thread::hardware_concurrency(), wav2vec2_print_system_info());
    fprintf(stderr, "\n");

    // Process each input file
    for (const auto & fname : params.fname_inp) {
        fprintf(stderr, "\nprocessing '%s' ...\n", fname.c_str());

        // Read audio file
        std::vector<float> pcmf32;
        int sample_rate = 0;

        if (!read_wav(fname, pcmf32, sample_rate)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname.c_str());
            continue;
        }

        fprintf(stderr, "  audio: %.2f sec, %d Hz, %d samples\n",
                (float) pcmf32.size() / sample_rate, sample_rate, (int) pcmf32.size());

        // Resample to 16kHz if needed
        if (sample_rate != WAV2VEC2_SAMPLE_RATE) {
            fprintf(stderr, "  resampling to %d Hz...\n", WAV2VEC2_SAMPLE_RATE);
            resample_16khz(pcmf32, sample_rate);
            fprintf(stderr, "  after resampling: %d samples\n", (int) pcmf32.size());
        }

        // Normalize audio to mean=0, std=1 (matching HuggingFace Wav2Vec2FeatureExtractor)
        {
            double sum = 0.0;
            for (size_t i = 0; i < pcmf32.size(); i++) {
                sum += pcmf32[i];
            }
            double mean = sum / pcmf32.size();

            double sq_sum = 0.0;
            for (size_t i = 0; i < pcmf32.size(); i++) {
                double diff = pcmf32[i] - mean;
                sq_sum += diff * diff;
            }
            double std = std::sqrt(sq_sum / pcmf32.size());

            // Avoid division by zero
            if (std < 1e-7) std = 1e-7;

            for (size_t i = 0; i < pcmf32.size(); i++) {
                pcmf32[i] = (pcmf32[i] - mean) / std;
            }

            fprintf(stderr, "  normalized: mean=%.6f, std=%.6f\n", mean, std);
        }

        // If testing streaming parity, run the test and skip normal output
        if (params.test_streaming) {
            bool test_passed = test_streaming_parity(ctx, pcmf32, params);
            if (!test_passed) {
                fprintf(stderr, "\nStreaming parity test FAILED for '%s'\n", fname.c_str());
            }
            continue;  // Skip normal output when testing
        }

        // Run inference
        struct wav2vec2_full_params wparams = wav2vec2_full_default_params();
        wparams.n_threads = params.n_threads;
        wparams.merge_repeated = params.merge_repeated;
        wparams.blank_suppress = params.suppress_blank;
        wparams.token_timestamps = params.print_timestamps;

        if (wav2vec2_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "error: failed to process audio\n");
            continue;
        }

        // Print results
        fprintf(stderr, "\n");
        printf("\n");

        const int n_phonemes = wav2vec2_full_n_phonemes(ctx);

        if (n_phonemes == 0) {
            printf("(no phonemes detected)\n");
        } else {
            if (params.print_timestamps) {
                // Print with timestamps
                for (int i = 0; i < n_phonemes; i++) {
                    wav2vec2_phoneme_data data = wav2vec2_full_get_phoneme_data(ctx, i);
                    const char * text = wav2vec2_full_get_phoneme_text(ctx, i);

                    int64_t t0_ms = wav2vec2_samples_to_ms(data.t0);
                    int64_t t1_ms = wav2vec2_samples_to_ms(data.t1);

                    if (params.print_probs) {
                        printf("[%5" PRId64 " - %5" PRId64 "] %s (%.2f)\n", t0_ms, t1_ms, text, data.p);
                    } else {
                        printf("[%5" PRId64 " - %5" PRId64 "] %s\n", t0_ms, t1_ms, text);
                    }
                }
            } else {
                // Print as continuous string
                char * all_phonemes = wav2vec2_full_get_all_phonemes(ctx);
                printf("IPA: %s\n", all_phonemes);
                free(all_phonemes);
            }
        }

        printf("\n");
    }

    // Cleanup
    wav2vec2_free(ctx);

    return 0;
}
