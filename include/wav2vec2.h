#ifndef WAV2VEC2_H
#define WAV2VEC2_H

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef WAV2VEC2_SHARED
#    ifdef _WIN32
#        ifdef WAV2VEC2_BUILD
#            define WAV2VEC2_API __declspec(dllexport)
#        else
#            define WAV2VEC2_API __declspec(dllimport)
#        endif
#    else
#        define WAV2VEC2_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WAV2VEC2_API
#endif

#define WAV2VEC2_SAMPLE_RATE 16000

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // Basic usage:
    //
    //     #include "wav2vec2.h"
    //
    //     struct wav2vec2_context_params params = wav2vec2_context_default_params();
    //     struct wav2vec2_context * ctx = wav2vec2_init_from_file("/path/to/model.bin", params);
    //
    //     struct wav2vec2_full_params wparams = wav2vec2_full_default_params();
    //     if (wav2vec2_full(ctx, wparams, pcmf32, n_samples) != 0) {
    //         fprintf(stderr, "failed to process audio\n");
    //         return 1;
    //     }
    //
    //     const int n_phonemes = wav2vec2_full_n_phonemes(ctx);
    //     for (int i = 0; i < n_phonemes; ++i) {
    //         const char * phoneme = wav2vec2_full_get_phoneme_text(ctx, i);
    //         printf("%s", phoneme);
    //     }
    //
    //     wav2vec2_free(ctx);
    //

    struct wav2vec2_context;
    struct wav2vec2_state;

    typedef int32_t wav2vec2_token;

    // Phoneme token data with timing information
    typedef struct wav2vec2_phoneme_data {
        wav2vec2_token id;   // phoneme token id
        float p;             // probability of the phoneme
        int64_t t0;          // start time in samples
        int64_t t1;          // end time in samples
    } wav2vec2_phoneme_data;

    // Context parameters for initialization
    struct wav2vec2_context_params {
        bool use_gpu;
        int  gpu_device;
    };

    // Full processing parameters
    struct wav2vec2_full_params {
        int n_threads;

        // CTC decoding options
        bool blank_suppress;        // suppress blank tokens in output
        bool merge_repeated;        // merge repeated phonemes

        // Timing
        bool token_timestamps;      // compute token-level timestamps
    };

    // Return default context parameters
    WAV2VEC2_API struct wav2vec2_context_params wav2vec2_context_default_params(void);

    // Return default full params
    WAV2VEC2_API struct wav2vec2_full_params wav2vec2_full_default_params(void);

    // Initialize from GGML model file
    WAV2VEC2_API struct wav2vec2_context * wav2vec2_init_from_file(
        const char * path_model,
        struct wav2vec2_context_params params);

    // Initialize from buffer
    WAV2VEC2_API struct wav2vec2_context * wav2vec2_init_from_buffer(
        void * buffer,
        size_t buffer_size,
        struct wav2vec2_context_params params);

    // Free context
    WAV2VEC2_API void wav2vec2_free(struct wav2vec2_context * ctx);

    // Initialize state (for parallel processing)
    WAV2VEC2_API struct wav2vec2_state * wav2vec2_init_state(struct wav2vec2_context * ctx);

    // Free state
    WAV2VEC2_API void wav2vec2_free_state(struct wav2vec2_state * state);

    // Run the full phoneme recognition pipeline
    // Returns 0 on success
    WAV2VEC2_API int wav2vec2_full(
        struct wav2vec2_context * ctx,
        struct wav2vec2_full_params params,
        const float * samples,
        int n_samples);

    // Run with separate state (for parallel processing)
    WAV2VEC2_API int wav2vec2_full_with_state(
        struct wav2vec2_context * ctx,
        struct wav2vec2_state * state,
        struct wav2vec2_full_params params,
        const float * samples,
        int n_samples);

    // Get number of recognized phonemes from the last wav2vec2_full() call
    WAV2VEC2_API int wav2vec2_full_n_phonemes(struct wav2vec2_context * ctx);
    WAV2VEC2_API int wav2vec2_full_n_phonemes_from_state(struct wav2vec2_state * state);

    // Get phoneme data for index i
    WAV2VEC2_API wav2vec2_phoneme_data wav2vec2_full_get_phoneme_data(
        struct wav2vec2_context * ctx,
        int i);

    WAV2VEC2_API wav2vec2_phoneme_data wav2vec2_full_get_phoneme_data_from_state(
        struct wav2vec2_state * state,
        int i);

    // Get phoneme text (IPA symbol) for index i
    WAV2VEC2_API const char * wav2vec2_full_get_phoneme_text(
        struct wav2vec2_context * ctx,
        int i);

    WAV2VEC2_API const char * wav2vec2_full_get_phoneme_text_from_state(
        struct wav2vec2_state * state,
        int i);

    // Get the full IPA transcription as a string
    // The caller must free the returned string
    WAV2VEC2_API char * wav2vec2_full_get_all_phonemes(struct wav2vec2_context * ctx);
    WAV2VEC2_API char * wav2vec2_full_get_all_phonemes_from_state(struct wav2vec2_state * state);

    //
    // Streaming API for chunk-based processing
    //
    // Usage:
    //     struct wav2vec2_stream_params sparams = wav2vec2_stream_default_params();
    //     struct wav2vec2_stream_state * stream = wav2vec2_stream_init(ctx, sparams);
    //
    //     // Feed audio chunks as they arrive
    //     while (has_more_audio) {
    //         wav2vec2_stream_process(ctx, stream, chunk, chunk_size);
    //         // Get newly decoded phonemes
    //         int n = wav2vec2_stream_n_phonemes(stream);
    //         for (int i = 0; i < n; i++) { ... }
    //     }
    //
    //     // Finalize to flush remaining audio
    //     wav2vec2_stream_finalize(ctx, stream);
    //     wav2vec2_stream_free(stream);
    //

    struct wav2vec2_stream_state;

    // Streaming parameters
    struct wav2vec2_stream_params {
        int n_threads;

        // Growing window parameters (for real-time with early results)
        int min_samples;        // minimum samples before first processing (default: 16000 = 1s)
        int max_window_samples; // maximum window before sliding (default: 160000 = 10s)
        int step_samples;       // step size when sliding (default: 16000 = 1s)

        bool merge_repeated;    // merge repeated phonemes
        bool blank_suppress;    // suppress blank tokens
    };

    // Return default streaming parameters
    WAV2VEC2_API struct wav2vec2_stream_params wav2vec2_stream_default_params(void);

    // Initialize streaming state
    WAV2VEC2_API struct wav2vec2_stream_state * wav2vec2_stream_init(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_params params);

    // Process a chunk of audio samples
    // Returns number of new phonemes decoded, or -1 on error
    WAV2VEC2_API int wav2vec2_stream_process(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream,
        const float * samples,
        int n_samples);

    // Finalize streaming - process any remaining buffered audio
    // Returns number of new phonemes decoded, or -1 on error
    WAV2VEC2_API int wav2vec2_stream_finalize(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream);

    // Get number of NEW phonemes from last process/finalize call
    WAV2VEC2_API int wav2vec2_stream_n_phonemes(struct wav2vec2_stream_state * stream);

    // Get phoneme data for index i from last process/finalize call (new phonemes only)
    WAV2VEC2_API wav2vec2_phoneme_data wav2vec2_stream_get_phoneme_data(
        struct wav2vec2_stream_state * stream,
        int i);

    // Get phoneme text for index i from last process/finalize call (new phonemes only)
    WAV2VEC2_API const char * wav2vec2_stream_get_phoneme_text(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream,
        int i);

    // Get total number of phonemes accumulated across all process calls
    WAV2VEC2_API int wav2vec2_stream_n_all_phonemes(struct wav2vec2_stream_state * stream);

    // Get phoneme data for index i from all accumulated phonemes
    WAV2VEC2_API wav2vec2_phoneme_data wav2vec2_stream_get_all_phoneme_data(
        struct wav2vec2_stream_state * stream,
        int i);

    // Get phoneme text for index i from all accumulated phonemes
    WAV2VEC2_API const char * wav2vec2_stream_get_all_phoneme_text(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream,
        int i);

    // Reset streaming state for new audio (clears buffers and phonemes)
    WAV2VEC2_API void wav2vec2_stream_reset(struct wav2vec2_stream_state * stream);

    // Free streaming state
    WAV2VEC2_API void wav2vec2_stream_free(struct wav2vec2_stream_state * stream);

    //
    // Low-level API for more control
    //

    // Run CNN feature extractor
    WAV2VEC2_API int wav2vec2_encode(
        struct wav2vec2_context * ctx,
        struct wav2vec2_state * state,
        const float * samples,
        int n_samples,
        int n_threads);

    // Get encoder output length (number of frames)
    WAV2VEC2_API int wav2vec2_n_len(struct wav2vec2_context * ctx);
    WAV2VEC2_API int wav2vec2_n_len_from_state(struct wav2vec2_state * state);

    //
    // Model info
    //

    // Get vocabulary size (number of phoneme tokens)
    WAV2VEC2_API int wav2vec2_n_vocab(struct wav2vec2_context * ctx);

    // Get number of encoder layers
    WAV2VEC2_API int wav2vec2_n_layers(struct wav2vec2_context * ctx);

    // Get hidden size
    WAV2VEC2_API int wav2vec2_n_hidden(struct wav2vec2_context * ctx);

    // Get phoneme string for a token id
    WAV2VEC2_API const char * wav2vec2_token_to_str(struct wav2vec2_context * ctx, wav2vec2_token token);

    // Get token id for a phoneme string, returns -1 if not found
    WAV2VEC2_API wav2vec2_token wav2vec2_str_to_token(struct wav2vec2_context * ctx, const char * phoneme);

    // Special tokens
    WAV2VEC2_API wav2vec2_token wav2vec2_token_blank(struct wav2vec2_context * ctx);  // CTC blank token
    WAV2VEC2_API wav2vec2_token wav2vec2_token_pad(struct wav2vec2_context * ctx);    // padding token
    WAV2VEC2_API wav2vec2_token wav2vec2_token_unk(struct wav2vec2_context * ctx);    // unknown token

    //
    // Timing helpers
    //

    // Convert sample index to milliseconds (assuming 16kHz sample rate)
    WAV2VEC2_API int64_t wav2vec2_samples_to_ms(int64_t samples);

    // Convert milliseconds to sample index
    WAV2VEC2_API int64_t wav2vec2_ms_to_samples(int64_t ms);

    //
    // System info
    //

    WAV2VEC2_API const char * wav2vec2_print_system_info(void);

#ifdef __cplusplus
}
#endif

#endif // WAV2VEC2_H
