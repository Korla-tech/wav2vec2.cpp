// wav2vec2.cpp - Wav2Vec2 phoneme recognition implementation for whisper.cpp
//
// This implements the Wav2Vec2 architecture for phoneme recognition using CTC decoding.
// Based on the HuggingFace wav2vec2-xlsr-53-espeak-cv-ft model.

#include "wav2vec2.h"
#include "wav2vec2-arch.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#ifdef __GNUC__
#ifdef __MINGW32__
#define W2V_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define W2V_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define W2V_ATTRIBUTE_FORMAT(...)
#endif

//
// Logging
//

W2V_ATTRIBUTE_FORMAT(2, 3)
static void wav2vec2_log_internal(ggml_log_level level, const char * format, ...);
static void wav2vec2_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define W2V_LOG_ERROR(...) wav2vec2_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define W2V_LOG_WARN(...)  wav2vec2_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define W2V_LOG_INFO(...)  wav2vec2_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

//#define W2V_DEBUG
#if defined(W2V_DEBUG)
#define W2V_LOG_DEBUG(...) wav2vec2_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define W2V_LOG_DEBUG(...)
#endif

#define W2V_ASSERT(x) \
    do { \
        if (!(x)) { \
            W2V_LOG_ERROR("W2V_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#define W2V_MAX_NODES 8192
#define W2V_MAX_CONV_LAYERS 7

// SeamlessM4T feature extractor constants for Wav2Vec2Bert input_features
static constexpr int W2V_FBANK_SR          = 16000;
static constexpr int W2V_FBANK_FFT         = 512;
static constexpr int W2V_FBANK_FRAME       = 400;
static constexpr int W2V_FBANK_HOP         = 160;
static constexpr int W2V_FBANK_MELS        = 80;
static constexpr int W2V_FBANK_STRIDE      = 2;
static constexpr float W2V_FBANK_PREEMPH   = 0.97f;
static constexpr float W2V_FBANK_MEL_FLOOR = 1.192092955078125e-07f;

//
// Hyperparameters
//

struct wav2vec2_hparams {
    int32_t n_hidden       = 1024;  // hidden_size
    int32_t n_layers       = 24;    // num_hidden_layers
    int32_t n_heads        = 16;    // num_attention_heads
    int32_t n_intermediate = 4096;  // intermediate_size
    int32_t n_vocab        = 392;   // vocab_size (phonemes)
    int32_t n_conv_layers  = 31;    // conv_depthwise_kernel_size in wav2vec2-bert config

    // Legacy CNN config (unused by wav2vec2-bert)
    int32_t conv_dim[W2V_MAX_CONV_LAYERS]    = {512, 512, 512, 512, 512, 512, 512};
    int32_t conv_kernel[W2V_MAX_CONV_LAYERS] = {10, 3, 3, 3, 3, 2, 2};
    int32_t conv_stride[W2V_MAX_CONV_LAYERS] = {5, 2, 2, 2, 2, 2, 2};

    // Legacy positional conv config (unused by wav2vec2-bert)
    int32_t num_conv_pos_embeddings = 128;
    int32_t num_conv_pos_embedding_groups = 16;

    // Wav2Vec2Bert frontend config from preprocessor_config.json
    int32_t feature_size = 80;
    int32_t feature_stride = 2;

    // relative_key attention defaults
    int32_t left_max_position_embeddings = 64;
    int32_t right_max_position_embeddings = 8;

    // adapter defaults
    int32_t adapter_stride = 2;

    int32_t ftype = 1; // 0 = f32, 1 = f16

    float eps = 1e-5f;  // layer norm epsilon
};

//
// Vocabulary
//

struct wav2vec2_vocab {
    using id = int32_t;
    using token = std::string;

    int n_vocab = 392;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    // Special token IDs (typical for wav2vec2 phoneme models)
    id token_pad   = 0;   // <pad>
    id token_unk   = 1;   // <unk>
    id token_blank = 0;   // CTC blank (usually same as pad)
    id token_space = 0;   // word delimiter token (typically "|")
};

//
// CNN Feature Extractor layer
//

struct wav2vec2_conv_layer {
    struct ggml_tensor * conv_w;     // [out_ch, in_ch, kernel]
    struct ggml_tensor * conv_b;     // [out_ch] - may be null for first layer
    struct ggml_tensor * ln_w;       // [out_ch] - layer norm weight
    struct ggml_tensor * ln_b;       // [out_ch] - layer norm bias
};

//
// Transformer encoder layer
//

struct wav2vec2_encoder_layer {
    // Self attention
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;
    struct ggml_tensor * attn_k_w;
    struct ggml_tensor * attn_k_b;
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;
    struct ggml_tensor * attn_out_w;
    struct ggml_tensor * attn_out_b;
    struct ggml_tensor * attn_distance_emb_w;

    // Attention layer norm
    struct ggml_tensor * attn_ln_w;
    struct ggml_tensor * attn_ln_b;

    // Conformer convolution module
    struct ggml_tensor * conv_ln_w;       // conv_module.layer_norm.weight
    struct ggml_tensor * conv_ln_b;       // conv_module.layer_norm.bias
    struct ggml_tensor * conv_pw1_w;      // conv_module.pointwise_conv1.weight
    struct ggml_tensor * conv_dw_w;       // conv_module.depthwise_conv.weight
    struct ggml_tensor * conv_dw_ln_w;    // conv_module.depthwise_layer_norm.weight
    struct ggml_tensor * conv_dw_ln_b;    // conv_module.depthwise_layer_norm.bias
    struct ggml_tensor * conv_pw2_w;      // conv_module.pointwise_conv2.weight

    // Feed forward 1
    struct ggml_tensor * ffn1_up_w;    // ffn1.intermediate_dense
    struct ggml_tensor * ffn1_up_b;
    struct ggml_tensor * ffn1_down_w;  // ffn1.output_dense
    struct ggml_tensor * ffn1_down_b;
    struct ggml_tensor * ffn1_ln_w;    // ffn1_layer_norm
    struct ggml_tensor * ffn1_ln_b;

    // Feed forward 2
    struct ggml_tensor * ffn2_up_w;    // ffn2.intermediate_dense
    struct ggml_tensor * ffn2_up_b;
    struct ggml_tensor * ffn2_down_w;  // ffn2.output_dense
    struct ggml_tensor * ffn2_down_b;
    struct ggml_tensor * ffn2_ln_w;    // ffn2_layer_norm
    struct ggml_tensor * ffn2_ln_b;

    // Final layer norm in each encoder layer
    struct ggml_tensor * final_ln_w;   // final_layer_norm
    struct ggml_tensor * final_ln_b;
};

struct wav2vec2_adapter_layer {
    struct ggml_tensor * residual_ln_w;
    struct ggml_tensor * residual_ln_b;
    struct ggml_tensor * residual_conv_w;
    struct ggml_tensor * residual_conv_b;

    struct ggml_tensor * self_attn_ln_w;
    struct ggml_tensor * self_attn_ln_b;
    struct ggml_tensor * self_attn_conv_w;
    struct ggml_tensor * self_attn_conv_b;

    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;
    struct ggml_tensor * attn_k_w;
    struct ggml_tensor * attn_k_b;
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;
    struct ggml_tensor * attn_out_w;
    struct ggml_tensor * attn_out_b;

    struct ggml_tensor * ffn_ln_w;
    struct ggml_tensor * ffn_ln_b;
    struct ggml_tensor * ffn_up_w;
    struct ggml_tensor * ffn_up_b;
    struct ggml_tensor * ffn_down_w;
    struct ggml_tensor * ffn_down_b;
};

//
// Model
//

struct wav2vec2_model {
    wav2vec2_hparams hparams;

    // CNN Feature Extractor (7 layers typically)
    std::vector<wav2vec2_conv_layer> conv_layers;

    // Feature projection (after CNN)
    struct ggml_tensor * feat_proj_w;
    struct ggml_tensor * feat_proj_b;
    struct ggml_tensor * feat_ln_w;
    struct ggml_tensor * feat_ln_b;

    // Positional conv embedding
    struct ggml_tensor * pos_conv_w;
    struct ggml_tensor * pos_conv_b;

    // Encoder layer norm (before transformer layers)
    struct ggml_tensor * enc_ln_w;
    struct ggml_tensor * enc_ln_b;

    // Transformer encoder layers
    std::vector<wav2vec2_encoder_layer> layers;

    // Optional adapter stack
    struct ggml_tensor * adapter_proj_w = nullptr;
    struct ggml_tensor * adapter_proj_b = nullptr;
    struct ggml_tensor * adapter_proj_ln_w = nullptr;
    struct ggml_tensor * adapter_proj_ln_b = nullptr;
    std::vector<wav2vec2_adapter_layer> adapter_layers;

    // CTC head (lm_head)
    struct ggml_tensor * ctc_w;
    struct ggml_tensor * ctc_b;

    // ggml context for meta info
    std::vector<struct ggml_context *> ctxs;

    // Backend buffers
    std::vector<ggml_backend_buffer_t> buffers;

    // All tensors by name
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

//
// Phoneme result
//

struct wav2vec2_phoneme {
    wav2vec2_token id;
    float prob;
    int64_t t0;  // start sample
    int64_t t1;  // end sample
};

//
// Backend scheduler wrapper
//

struct wav2vec2_sched {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
};

//
// State (for inference)
//

struct wav2vec2_state {
    int64_t t_load_us   = 0;
    int64_t t_conv_us   = 0;
    int64_t t_encode_us = 0;
    int64_t t_ctc_us    = 0;

    int32_t n_encode = 0;

    std::vector<ggml_backend_t> backends;

    wav2vec2_sched sched_conv;
    wav2vec2_sched sched_encode;
    wav2vec2_sched sched_ctc;

    // Intermediate tensors
    struct ggml_tensor * embd_conv = nullptr;  // After CNN
    struct ggml_tensor * embd_enc  = nullptr;  // After transformer
    struct ggml_tensor * logits    = nullptr;  // CTC logits

    int n_len = 0;  // number of output frames

    // CTC logits buffer
    std::vector<float> logits_buf;

    // Decoded phonemes
    std::vector<wav2vec2_phoneme> phonemes;

    // Work buffers
    std::vector<float> inp_audio;
    std::vector<int32_t> rel_pos_idx_buf;
};

//
// Context
//

struct wav2vec2_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = GGML_TYPE_F16;
    ggml_type itype = GGML_TYPE_F32;

    wav2vec2_context_params params;

    wav2vec2_model model;
    wav2vec2_vocab vocab;

    wav2vec2_state * state = nullptr;

    std::string path_model;

    // File position after vocabulary for tensor data reading
    std::streampos vocab_end_pos = 0;
};

//
// Global state
//

struct wav2vec2_global {
    ggml_log_callback log_callback = wav2vec2_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static wav2vec2_global g_state;

static wav2vec2_vocab::id wav2vec2_find_first_token_id(
        const wav2vec2_vocab & vocab,
        const std::initializer_list<const char *> & names,
        wav2vec2_vocab::id fallback) {
    for (const char * n : names) {
        auto it = vocab.token_to_id.find(n);
        if (it != vocab.token_to_id.end()) {
            return it->second;
        }
    }
    return fallback;
}

static bool wav2vec2_is_special_print_suppressed(const wav2vec2_context & ctx, wav2vec2_token id) {
    return id == ctx.vocab.token_pad || id == ctx.vocab.token_blank || id == ctx.vocab.token_unk;
}

static std::string wav2vec2_render_token(const wav2vec2_context & ctx, wav2vec2_token id) {
    if (wav2vec2_is_special_print_suppressed(ctx, id)) {
        return "";
    }

    if (id == ctx.vocab.token_space) {
        return " ";
    }

    auto it = ctx.vocab.id_to_token.find(id);
    if (it == ctx.vocab.id_to_token.end()) {
        return "";
    }
    return it->second;
}

//
// Logging implementation
//

static void wav2vec2_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

static void wav2vec2_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, sizeof(buffer), format, args);
    if (len < (int) sizeof(buffer)) {
        if (g_state.log_callback) {
            g_state.log_callback(level, buffer, g_state.log_callback_user_data);
        }
    } else {
        std::vector<char> buf2(len + 1);
        vsnprintf(buf2.data(), buf2.size(), format, args);
        if (g_state.log_callback) {
            g_state.log_callback(level, buf2.data(), g_state.log_callback_user_data);
        }
    }
    va_end(args);
}

//
// Helper functions
//

static bool ggml_graph_compute_helper(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        int n_threads) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        if (reg) {
            auto * fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn) {
                fn(backend, n_threads);
            }
        }
    }

    bool ok = (ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_reset(sched);
    return ok;
}

static size_t wav2vec2_sched_size(struct wav2vec2_sched & sched) {
    size_t size = sched.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched.sched, i);
        size += ggml_backend_sched_get_buffer_size(sched.sched, backend);
    }
    return size;
}

// The model uses LayerNorm over feature/hidden dimension for [C, T] tensors.
// In this ggml layout, transpose to [T, C], normalize on last dim, then transpose back.
static struct ggml_tensor * wav2vec2_layer_norm_features(
        struct ggml_context * ctx,
        struct ggml_tensor * x,
        float eps) {
    // Explicit LayerNorm over feature axis for [F, T]:
    // mean/var are reduced over ne[0] using sum_rows.
    const float inv_f = 1.0f / (float) x->ne[0];

    struct ggml_tensor * mean = ggml_scale(ctx, ggml_sum_rows(ctx, x), inv_f); // [1, T]
    struct ggml_tensor * mean_rep = ggml_repeat(ctx, mean, x);                  // [F, T]
    struct ggml_tensor * xc = ggml_sub(ctx, x, mean_rep);                       // [F, T]

    struct ggml_tensor * var = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, xc)), inv_f); // [1, T]
    struct ggml_tensor * var_rep = ggml_repeat(ctx, var, x);                    // [F, T]

    struct ggml_tensor * var_eps = ggml_scale_bias(ctx, var_rep, 1.0f, eps);
    struct ggml_tensor * denom = ggml_sqrt(ctx, var_eps);
    return ggml_div(ctx, xc, denom);
}

// Apply per-feature affine parameters to [C, T] tensor explicitly.
// We reshape [C] -> [C, 1] then repeat to [C, T] to avoid ambiguous broadcasting.
static struct ggml_tensor * wav2vec2_affine_features(
        struct ggml_context * ctx,
        struct ggml_tensor * x,
        struct ggml_tensor * w,
        struct ggml_tensor * b) {
    if (w == nullptr || b == nullptr) {
        return x;
    }

    const int64_t n_feat = ggml_nelements(w);

    // x is [F, T]. Apply y = x * w[:, None] + b[:, None] explicitly.
    struct ggml_tensor * w_ft = ggml_reshape_2d(ctx, w, n_feat, 1);
    struct ggml_tensor * b_ft = ggml_reshape_2d(ctx, b, n_feat, 1);

    struct ggml_tensor * wr = ggml_repeat(ctx, w_ft, x);
    struct ggml_tensor * br = ggml_repeat(ctx, b_ft, x);

    return ggml_add(ctx, ggml_mul(ctx, x, wr), br);
}

// PyTorch nn.GLU(dim=channel): split tensor in half on channel dim and apply
// y = a * sigmoid(b).
static struct ggml_tensor * wav2vec2_glu_sigmoid(
        struct ggml_context * ctx,
        struct ggml_tensor * x) {
    const int64_t c2 = x->ne[0];
    GGML_ASSERT(c2 % 2 == 0);
    const int64_t c = c2 / 2;

    struct ggml_tensor * a = ggml_view_2d(ctx, x, c, x->ne[1], x->nb[1], 0);
    struct ggml_tensor * b = ggml_view_2d(ctx, x, c, x->ne[1], x->nb[1], c * x->nb[0]);
    b = ggml_sigmoid(ctx, b);

    return ggml_mul(ctx, a, b);
}

static struct ggml_tensor * wav2vec2_layer_norm_affine(
        struct ggml_context * ctx,
        struct ggml_tensor * x,
        struct ggml_tensor * w,
        struct ggml_tensor * b,
        float eps) {
    x = wav2vec2_layer_norm_features(ctx, x, eps);
    return wav2vec2_affine_features(ctx, x, w, b);
}

static inline float hz_to_mel(float hz) {
    return 1127.0f * logf(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

static std::vector<float> build_povey_window() {
    std::vector<float> w(W2V_FBANK_FRAME);
    for (int i = 0; i < W2V_FBANK_FRAME; ++i) {
        const float phase = 2.0f * (float)M_PI * i / (W2V_FBANK_FRAME - 1);
        const float hann = 0.5f - 0.5f * cosf(phase);
        w[i] = powf(hann, 0.85f);
    }
    return w;
}

static std::vector<float> build_mel_filter_bank() {
    const int n_freq = W2V_FBANK_FFT / 2 + 1;
    std::vector<float> fb(W2V_FBANK_MELS * n_freq, 0.0f);

    const float fmin = 20.0f;
    const float fmax = (float)W2V_FBANK_SR / 2.0f;
    const float mel_min = hz_to_mel(fmin);
    const float mel_max = hz_to_mel(fmax);

    std::vector<float> mel_points(W2V_FBANK_MELS + 2);
    for (int i = 0; i < (int)mel_points.size(); ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (W2V_FBANK_MELS + 1);
    }

    for (int m = 0; m < W2V_FBANK_MELS; ++m) {
        const float l = mel_points[m + 0];
        const float c = mel_points[m + 1];
        const float r = mel_points[m + 2];

        for (int k = 0; k < n_freq; ++k) {
            const float hz = (float)W2V_FBANK_SR * k / W2V_FBANK_FFT;
            const float mel = hz_to_mel(hz);

            float v = 0.0f;
            if (mel >= l && mel <= c && c > l) {
                v = (mel - l) / (c - l);
            } else if (mel >= c && mel <= r && r > c) {
                v = (r - mel) / (r - c);
            }
            fb[m * n_freq + k] = std::max(0.0f, v);
        }
    }

    return fb;
}

static void seamless_m4t_extract_features(
        const float * samples,
        int n_samples,
        std::vector<float> & out_features,
        int & out_n_frames) {
    static const std::vector<float> window = build_povey_window();
    static const std::vector<float> mel_fb = build_mel_filter_bank();

    if (!samples || n_samples < W2V_FBANK_FRAME) {
        out_features.clear();
        out_n_frames = 0;
        return;
    }

    const int n_raw_frames = 1 + (n_samples - W2V_FBANK_FRAME) / W2V_FBANK_HOP;
    const int n_freq = W2V_FBANK_FFT / 2 + 1;

    std::vector<float> mel_raw(n_raw_frames * W2V_FBANK_MELS, 0.0f);
    std::vector<float> frame(W2V_FBANK_FRAME, 0.0f);

    for (int t = 0; t < n_raw_frames; ++t) {
        const int off = t * W2V_FBANK_HOP;

        float mean = 0.0f;
        for (int i = 0; i < W2V_FBANK_FRAME; ++i) {
            mean += samples[off + i];
        }
        mean /= W2V_FBANK_FRAME;

        float prev = 0.0f;
        for (int i = 0; i < W2V_FBANK_FRAME; ++i) {
            float x = samples[off + i] - mean;
            const float y = (x - W2V_FBANK_PREEMPH * prev) * window[i] * 32768.0f;
            prev = x;
            frame[i] = y;
        }

        std::vector<float> power(n_freq, 0.0f);
        for (int k = 0; k < n_freq; ++k) {
            double re = 0.0;
            double im = 0.0;
            for (int n = 0; n < W2V_FBANK_FRAME; ++n) {
                const double a = -2.0 * M_PI * k * n / W2V_FBANK_FFT;
                re += frame[n] * cos(a);
                im += frame[n] * sin(a);
            }
            power[k] = (float)(re * re + im * im);
        }

        for (int m = 0; m < W2V_FBANK_MELS; ++m) {
            float e = 0.0f;
            const float * w = &mel_fb[m * n_freq];
            for (int k = 0; k < n_freq; ++k) {
                e += w[k] * power[k];
            }
            mel_raw[t * W2V_FBANK_MELS + m] = logf(std::max(e, W2V_FBANK_MEL_FLOOR));
        }
    }

    for (int m = 0; m < W2V_FBANK_MELS; ++m) {
        double sum = 0.0;
        for (int t = 0; t < n_raw_frames; ++t) sum += mel_raw[t * W2V_FBANK_MELS + m];
        const double mean = sum / n_raw_frames;

        double var = 0.0;
        for (int t = 0; t < n_raw_frames; ++t) {
            const double d = mel_raw[t * W2V_FBANK_MELS + m] - mean;
            var += d * d;
        }
        const double denom = std::max(1, n_raw_frames - 1);
        const float inv_std = (float)(1.0 / sqrt(var / denom + 1e-7));

        for (int t = 0; t < n_raw_frames; ++t) {
            mel_raw[t * W2V_FBANK_MELS + m] = (mel_raw[t * W2V_FBANK_MELS + m] - (float)mean) * inv_std;
        }
    }

    const int rem = n_raw_frames % W2V_FBANK_STRIDE;
    const int n_used = n_raw_frames - rem;
    out_n_frames = n_used / W2V_FBANK_STRIDE;
    out_features.assign(out_n_frames * (W2V_FBANK_MELS * W2V_FBANK_STRIDE), 0.0f);

    for (int t = 0; t < out_n_frames; ++t) {
        const int t0 = t * W2V_FBANK_STRIDE + 0;
        const int t1 = t * W2V_FBANK_STRIDE + 1;
        for (int m = 0; m < W2V_FBANK_MELS; ++m) {
            out_features[t * 160 + m] = mel_raw[t0 * W2V_FBANK_MELS + m];
            out_features[t * 160 + (W2V_FBANK_MELS + m)] = mel_raw[t1 * W2V_FBANK_MELS + m];
        }
    }
}

//
// Model loading
//

static bool wav2vec2_model_load(const char * fname, wav2vec2_context & wctx) {
    W2V_LOG_INFO("%s: loading model from '%s'\n", __func__, fname);

    const int64_t t_start_us = ggml_time_us();

    std::ifstream fin(fname, std::ios::binary);
    if (!fin) {
        W2V_LOG_ERROR("%s: failed to open '%s'\n", __func__, fname);
        return false;
    }

    // Read magic
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != 0x77766232) {  // "wv2b"
        W2V_LOG_ERROR("%s: invalid magic number: 0x%08x (expected 0x77766232)\n", __func__, magic);
        return false;
    }

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;
    auto & hparams = model.hparams;

    // Read hyperparameters
    fin.read((char *) &hparams.n_hidden,       sizeof(hparams.n_hidden));
    fin.read((char *) &hparams.n_layers,       sizeof(hparams.n_layers));
    fin.read((char *) &hparams.n_heads,        sizeof(hparams.n_heads));
    fin.read((char *) &hparams.n_intermediate, sizeof(hparams.n_intermediate));
    fin.read((char *) &hparams.n_vocab,        sizeof(hparams.n_vocab));
    fin.read((char *) &hparams.n_conv_layers,  sizeof(hparams.n_conv_layers));

    // wav2vec2-bert converter writes only these 6 hparams.
    hparams.feature_size = W2V_FBANK_MELS;
    hparams.feature_stride = W2V_FBANK_STRIDE;

    W2V_LOG_INFO("%s: n_hidden       = %d\n", __func__, hparams.n_hidden);
    W2V_LOG_INFO("%s: n_layers       = %d\n", __func__, hparams.n_layers);
    W2V_LOG_INFO("%s: n_heads        = %d\n", __func__, hparams.n_heads);
    W2V_LOG_INFO("%s: n_intermediate = %d\n", __func__, hparams.n_intermediate);
    W2V_LOG_INFO("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
    W2V_LOG_INFO("%s: conv_dw_kernel = %d\n", __func__, hparams.n_conv_layers);

    wctx.wtype = GGML_TYPE_F16;
    wctx.itype = GGML_TYPE_F32;

    // Read vocabulary
    {
        int32_t n_vocab;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        vocab.n_vocab = n_vocab;

        for (int i = 0; i < n_vocab; ++i) {
            int32_t len;
            fin.read((char *) &len, sizeof(len));

            std::string word(len, '\0');
            fin.read(&word[0], len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        W2V_LOG_INFO("%s: vocab size = %d\n", __func__, (int) vocab.n_vocab);

        // Remember position after vocabulary for second pass
        wctx.vocab_end_pos = fin.tellg();

        // Set special tokens with broad compatibility across converters/vocabs.
        vocab.token_pad = wav2vec2_find_first_token_id(vocab, {"<pad>", "[PAD]", "<PAD>", "pad", "PAD"}, vocab.token_pad);
        vocab.token_blank = vocab.token_pad;
        vocab.token_unk = wav2vec2_find_first_token_id(vocab, {"<unk>", "[UNK]", "<UNK>", "unk", "UNK"}, vocab.token_unk);
        vocab.token_space = wav2vec2_find_first_token_id(vocab, {"|", "<space>", "[SPACE]", " "}, vocab.token_space);

        W2V_LOG_INFO("%s: token ids: pad=%d blank=%d unk=%d space=%d\n",
                     __func__, vocab.token_pad, vocab.token_blank, vocab.token_unk, vocab.token_space);
    }

    // Calculate buffer sizes
    size_t ctx_size = 0;
    {
        const int n_layer = hparams.n_layers;
        const int n_hidden = hparams.n_hidden;
        const int n_inter = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_hidden * 160 * ggml_type_size(wctx.wtype);
        ctx_size += n_layer * (4 * n_hidden * n_hidden + 2 * n_hidden * n_inter + n_hidden * 8) * ggml_type_size(wctx.wtype);
        ctx_size += n_vocab * n_hidden * ggml_type_size(wctx.wtype);
        ctx_size += ctx_size;  // Double the size for safety

        W2V_LOG_INFO("%s: estimated ctx size = %.2f MB\n", __func__, ctx_size / 1024.0 / 1024.0);
    }

    W2V_LOG_INFO("%s: creating ggml context...\n", __func__);

    // Create ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        W2V_LOG_ERROR("%s: failed to create ggml context\n", __func__);
        return false;
    }
    model.ctxs.push_back(ctx);

    W2V_LOG_INFO("%s: ggml context created\n", __func__);

    // Initialize layer structures
    model.layers.resize(hparams.n_layers);

    W2V_LOG_INFO("%s: reading tensors...\n", __func__);

    // Read tensors
    model.n_loaded = 0;
    int tensor_idx = 0;

    while (true) {
        int32_t n_dims;
        int32_t name_len;
        int32_t ftype;

        fin.read((char *) &n_dims,   sizeof(n_dims));
        if (fin.eof()) {
            break;
        }
        if (!fin.good()) {
            W2V_LOG_ERROR("%s: error reading tensor %d\n", __func__, tensor_idx);
            break;
        }

        fin.read((char *) &name_len, sizeof(name_len));
        fin.read((char *) &ftype,    sizeof(ftype));

        if (n_dims < 0 || n_dims > 4 || name_len <= 0 || name_len > 256) {
            W2V_LOG_ERROR("%s: invalid tensor header: n_dims=%d, name_len=%d\n", __func__, n_dims, name_len);
            break;
        }

        tensor_idx++;

        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
            int32_t dim;
            fin.read((char *) &dim, sizeof(dim));
            ne[i] = dim;
        }

        std::string name(name_len, '\0');
        fin.read(&name[0], name_len);

        // Map ftype to ggml_type
        ggml_type type;
        switch (ftype) {
            case 0:  type = GGML_TYPE_F32;  break;
            case 1:  type = GGML_TYPE_F16;  break;
            case 2:  type = GGML_TYPE_Q4_0; break;
            case 3:  type = GGML_TYPE_Q4_1; break;
            case 6:  type = GGML_TYPE_Q5_0; break;
            case 7:  type = GGML_TYPE_Q8_0; break;
            case 10: type = GGML_TYPE_Q2_K; break;
            case 11: type = GGML_TYPE_Q3_K; break;
            case 12: type = GGML_TYPE_Q4_K; break;
            case 13: type = GGML_TYPE_Q5_K; break;
            case 14: type = GGML_TYPE_Q6_K; break;
            case 15: type = GGML_TYPE_Q8_K; break;
            default:
                W2V_LOG_ERROR("%s: unsupported ftype = %d for tensor '%s'\n", __func__, ftype, name.c_str());
                return false;
        }

        struct ggml_tensor * tensor = nullptr;

        if (n_dims == 1) {
            tensor = ggml_new_tensor_1d(ctx, type, ne[0]);
        } else if (n_dims == 2) {
            tensor = ggml_new_tensor_2d(ctx, type, ne[0], ne[1]);
        } else if (n_dims == 3) {
            tensor = ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
        } else if (n_dims == 4) {
            tensor = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        } else {
            W2V_LOG_ERROR("%s: unsupported n_dims = %d for tensor '%s'\n", __func__, n_dims, name.c_str());
            return false;
        }

        if (!tensor) {
            W2V_LOG_ERROR("%s: failed to create tensor '%s'\n", __func__, name.c_str());
            return false;
        }

        ggml_set_name(tensor, name.c_str());
        model.tensors[name] = tensor;
        model.n_loaded++;

        // Skip tensor data (we'll read it in the second pass)
        size_t nbytes = ggml_nbytes(tensor);
        fin.seekg(nbytes, std::ios::cur);

    }

    W2V_LOG_INFO("%s: loaded %d tensors\n", __func__, model.n_loaded);

    W2V_LOG_INFO("%s: initializing backend...\n", __func__);

    // Allocate backend buffer
    ggml_backend_t backend = nullptr;

    // Try to find GPU backend
    if (wctx.params.use_gpu) {
        int cnt = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);

            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                if (cnt == wctx.params.gpu_device) {
                    const char * dev_name = ggml_backend_dev_name(dev);
                    W2V_LOG_INFO("%s: using %s backend\n", __func__, dev_name);
                    backend = ggml_backend_dev_init(dev, nullptr);
                    break;
                }
                cnt++;
            }
        }
    }

    // Fall back to CPU
    if (!backend) {
        W2V_LOG_INFO("%s: using CPU backend\n", __func__);
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (!backend) {
        W2V_LOG_ERROR("%s: failed to initialize backend\n", __func__);
        return false;
    }

    // Allocate tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        W2V_LOG_ERROR("%s: failed to allocate tensor buffer\n", __func__);
        ggml_backend_free(backend);
        return false;
    }
    model.buffers.push_back(buffer);

    W2V_LOG_INFO("%s: buffer size = %.2f MB\n", __func__,
                 ggml_backend_buffer_get_size(buffer) / 1024.0 / 1024.0);

    // Read tensor data - seek to position after vocabulary
    fin.clear();  // Clear EOF flag
    fin.seekg(wctx.vocab_end_pos, std::ios::beg);
    W2V_LOG_INFO("%s: seeking to tensor data at position %lld\n",
                 __func__, (long long)wctx.vocab_end_pos);

    // Read tensor data
    int tensors_read = 0;
    while (tensors_read < model.n_loaded) {
        int32_t n_dims;
        int32_t name_len;
        int32_t ftype;

        std::streampos pos_before = fin.tellg();
        fin.read((char *) &n_dims,   sizeof(n_dims));
        if (fin.eof() || !fin.good()) {
            W2V_LOG_ERROR("%s: failed to read tensor %d header at pos %lld, eof=%d good=%d\n",
                          __func__, tensors_read, (long long)pos_before, (int)fin.eof(), (int)fin.good());
            break;
        }

        fin.read((char *) &name_len, sizeof(name_len));
        fin.read((char *) &ftype,    sizeof(ftype));

        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
            int32_t dim;
            fin.read((char *) &dim, sizeof(dim));
            ne[i] = dim;
        }

        std::string name(name_len, '\0');
        fin.read(&name[0], name_len);

        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            W2V_LOG_ERROR("%s: tensor '%s' not found\n", __func__, name.c_str());
            return false;
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        std::vector<char> buf(nbytes);
        fin.read(buf.data(), nbytes);

        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
        tensors_read++;
    }

    W2V_LOG_INFO("%s: read data for %d tensors\n", __func__, tensors_read);

    // Map tensors to model structure
    auto get_tensor = [&model](const std::string & name) -> ggml_tensor * {
        auto it = model.tensors.find(name);
        return it != model.tensors.end() ? it->second : nullptr;
    };

    // Feature projection
    model.feat_proj_w = get_tensor("wav2vec2_bert.feature_projection.projection.weight");
    model.feat_proj_b = get_tensor("wav2vec2_bert.feature_projection.projection.bias");
    model.feat_ln_w = get_tensor("wav2vec2_bert.feature_projection.layer_norm.weight");
    model.feat_ln_b = get_tensor("wav2vec2_bert.feature_projection.layer_norm.bias");

    // Not used by wav2vec2-bert
    model.pos_conv_w = nullptr;
    model.pos_conv_b = nullptr;
    model.enc_ln_w = nullptr;
    model.enc_ln_b = nullptr;

    // Map transformer layers
    for (int i = 0; i < hparams.n_layers; ++i) {
        auto & layer = model.layers[i];
        char buf[256];

        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_q.weight", i);
        layer.attn_q_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_q.bias", i);
        layer.attn_q_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_k.weight", i);
        layer.attn_k_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_k.bias", i);
        layer.attn_k_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_v.weight", i);
        layer.attn_v_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_v.bias", i);
        layer.attn_v_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_out.weight", i);
        layer.attn_out_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.linear_out.bias", i);
        layer.attn_out_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn.distance_embedding.weight", i);
        layer.attn_distance_emb_w = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn_layer_norm.weight", i);
        layer.attn_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.self_attn_layer_norm.bias", i);
        layer.attn_ln_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.layer_norm.weight", i);
        layer.conv_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.layer_norm.bias", i);
        layer.conv_ln_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.pointwise_conv1.weight", i);
        layer.conv_pw1_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.depthwise_conv.weight", i);
        layer.conv_dw_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.depthwise_layer_norm.weight", i);
        layer.conv_dw_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.depthwise_layer_norm.bias", i);
        layer.conv_dw_ln_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.conv_module.pointwise_conv2.weight", i);
        layer.conv_pw2_w = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn1.intermediate_dense.weight", i);
        layer.ffn1_up_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn1.intermediate_dense.bias", i);
        layer.ffn1_up_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn1.output_dense.weight", i);
        layer.ffn1_down_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn1.output_dense.bias", i);
        layer.ffn1_down_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn1_layer_norm.weight", i);
        layer.ffn1_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn1_layer_norm.bias", i);
        layer.ffn1_ln_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn2.intermediate_dense.weight", i);
        layer.ffn2_up_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn2.intermediate_dense.bias", i);
        layer.ffn2_up_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn2.output_dense.weight", i);
        layer.ffn2_down_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn2.output_dense.bias", i);
        layer.ffn2_down_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn2_layer_norm.weight", i);
        layer.ffn2_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.ffn2_layer_norm.bias", i);
        layer.ffn2_ln_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.final_layer_norm.weight", i);
        layer.final_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.encoder.layers.%d.final_layer_norm.bias", i);
        layer.final_ln_b = get_tensor(buf);
    }

    // Infer relative_key position bounds from embedding size if present.
    for (int i = 0; i < hparams.n_layers; ++i) {
        const auto * t = model.layers[i].attn_distance_emb_w;
        if (t && t->ne[1] > 1) {
            const int32_t n_pos = (int32_t)t->ne[1];
            hparams.right_max_position_embeddings = std::min<int32_t>(8, n_pos - 1);
            hparams.left_max_position_embeddings = n_pos - 1 - hparams.right_max_position_embeddings;
            break;
        }
    }

    // Optional adapter projection and adapter layers
    model.adapter_proj_w = get_tensor("wav2vec2_bert.adapter.proj.weight");
    model.adapter_proj_b = get_tensor("wav2vec2_bert.adapter.proj.bias");
    model.adapter_proj_ln_w = get_tensor("wav2vec2_bert.adapter.proj_layer_norm.weight");
    model.adapter_proj_ln_b = get_tensor("wav2vec2_bert.adapter.proj_layer_norm.bias");

    for (int i = 0;; ++i) {
        char buf[256];
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.residual_layer_norm.weight", i);
        if (!get_tensor(buf)) {
            break;
        }

        wav2vec2_adapter_layer al{};
        al.residual_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.residual_layer_norm.bias", i);
        al.residual_ln_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.residual_conv.weight", i);
        al.residual_conv_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.residual_conv.bias", i);
        al.residual_conv_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn_layer_norm.weight", i);
        al.self_attn_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn_layer_norm.bias", i);
        al.self_attn_ln_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn_conv.weight", i);
        al.self_attn_conv_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn_conv.bias", i);
        al.self_attn_conv_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_q.weight", i);
        al.attn_q_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_q.bias", i);
        al.attn_q_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_k.weight", i);
        al.attn_k_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_k.bias", i);
        al.attn_k_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_v.weight", i);
        al.attn_v_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_v.bias", i);
        al.attn_v_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_out.weight", i);
        al.attn_out_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.self_attn.linear_out.bias", i);
        al.attn_out_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.ffn_layer_norm.weight", i);
        al.ffn_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.ffn_layer_norm.bias", i);
        al.ffn_ln_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.ffn.intermediate_dense.weight", i);
        al.ffn_up_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.ffn.intermediate_dense.bias", i);
        al.ffn_up_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.ffn.output_dense.weight", i);
        al.ffn_down_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2_bert.adapter.layers.%d.ffn.output_dense.bias", i);
        al.ffn_down_b = get_tensor(buf);

        model.adapter_layers.push_back(al);
    }

    // CTC head
    model.ctc_w = get_tensor("lm_head.weight");
    model.ctc_b = get_tensor("lm_head.bias");

    // Initialize state
    wctx.state = new wav2vec2_state();

    // Add GPU backend if available
    if (backend && ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        wctx.state->backends.push_back(backend);
    }

    // Add CPU backend (must be last for scheduler)
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_backend) {
        wctx.state->backends.push_back(cpu_backend);
    }

    // If no GPU backend was added, backend is CPU and already added
    if (backend && ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        // Already added above
    }

    // Initialize scheduler meta buffers
    wctx.state->sched_conv.meta.resize(ggml_tensor_overhead() * W2V_MAX_NODES + ggml_graph_overhead());
    wctx.state->sched_encode.meta.resize(ggml_tensor_overhead() * W2V_MAX_NODES + ggml_graph_overhead());
    wctx.state->sched_ctc.meta.resize(ggml_tensor_overhead() * W2V_MAX_NODES + ggml_graph_overhead());

    wctx.t_load_us = ggml_time_us() - t_start_us;

    W2V_LOG_INFO("%s: model loaded in %.2f s\n", __func__, wctx.t_load_us / 1000000.0);

    return true;
}

//
// Build CNN feature extractor graph
//

static struct ggml_cgraph * wav2vec2_build_graph_conv(
        wav2vec2_context & wctx,
        wav2vec2_state & wstate,
        int n_frames) {

    const auto & hparams = wctx.model.hparams;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_conv.meta.size(),
        /*.mem_buffer =*/ wstate.sched_conv.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, W2V_MAX_NODES, false);

    // Input features: [160, n_frames]
    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.feature_size * hparams.feature_stride, n_frames);
    ggml_set_name(inp, "features_input");
    ggml_set_input(inp);

    struct ggml_tensor * cur = ggml_cont(ctx0, inp);

    ggml_set_name(cur, "conv_features");
    ggml_set_output(cur);
    wstate.embd_conv = cur;

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

//
// Build transformer encoder graph
//

static struct ggml_cgraph * wav2vec2_build_graph_encoder(
        wav2vec2_context & wctx,
        wav2vec2_state & wstate,
        int n_ctx) {

    const auto & model = wctx.model;
    const auto & hparams = model.hparams;

    const int n_hidden = hparams.n_hidden;
    const int n_heads = hparams.n_heads;
    const int n_layers = hparams.n_layers;
    const int n_head_dim = n_hidden / n_heads;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_encode.meta.size(),
        /*.mem_buffer =*/ wstate.sched_encode.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, W2V_MAX_NODES, false);

    // Input features from SeamlessM4T extractor: [160, n_ctx]
    const int n_cnn_out = hparams.feature_size * hparams.feature_stride;
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_cnn_out, n_ctx);
    ggml_set_name(cur, "encoder_input");
    ggml_set_input(cur);

    struct ggml_tensor * rel_pos_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx * n_ctx);
    ggml_set_name(rel_pos_idx, "relative_pos_indices");
    ggml_set_input(rel_pos_idx);

    // Feature layer norm (applied to 512-dim CNN output BEFORE projection)
    // cur has shape [512, n_ctx] - ggml_norm normalizes along ne[0]=512 which is features
    if (model.feat_ln_w && model.feat_ln_b) {
        // HF LayerNorm over feature dimension.
        cur = wav2vec2_layer_norm_features(ctx0, cur, hparams.eps);

        cur = wav2vec2_affine_features(ctx0, cur, model.feat_ln_w, model.feat_ln_b);
        cur = ggml_cont(ctx0, cur);
    }

    // Feature projection: project from 512 to n_hidden (1024)
    if (model.feat_proj_w) {
        cur = ggml_mul_mat(ctx0, model.feat_proj_w, cur);

        if (model.feat_proj_b) {
            cur = ggml_add(ctx0, cur, model.feat_proj_b);
        }
    }

    // Positional conv embedding (grouped convolution, groups=16)
    // Weight: [128, 64, 1024] - kernel_size=128, in_ch/groups=64, out_ch=1024
    // groups=16, so each group handles 64 channels
    // This is implemented as 16 separate conv1d operations concatenated
    if (model.pos_conv_w && model.pos_conv_b) {

        // cur is [n_hidden=1024, n_ctx]
        // Save for residual connection
        struct ggml_tensor * residual = cur;

        // Transpose to [n_ctx, 1024] for conv1d (which expects [time, channels, batch])
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // [n_ctx, 1024]
        cur = ggml_reshape_3d(ctx0, cur, n_ctx, n_hidden, 1);  // [n_ctx, 1024, 1]
        const int n_groups = hparams.num_conv_pos_embedding_groups;  // 16
        const int kernel_size = hparams.num_conv_pos_embeddings;     // 128
        const int ch_per_group = n_hidden / n_groups;                // 64
        const int padding = kernel_size / 2;                         // 64 (same padding)

        // Build grouped conv output by concatenating results from each group
        struct ggml_tensor * pos_out = nullptr;

        for (int g = 0; g < n_groups; g++) {
            // Extract input channels for this group: [n_ctx, 64, 1]
            // Using ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
            struct ggml_tensor * inp_g = ggml_view_3d(ctx0, cur,
                n_ctx, ch_per_group, 1,
                cur->nb[1], cur->nb[2],
                g * ch_per_group * cur->nb[1]);

            // Extract weight for this group: [128, 64, 64]
            // Weight is [128, 64, 1024], we want out_channels [g*64, (g+1)*64]
            struct ggml_tensor * w_g = ggml_view_3d(ctx0, model.pos_conv_w,
                kernel_size, ch_per_group, ch_per_group,
                model.pos_conv_w->nb[1], model.pos_conv_w->nb[2],
                g * ch_per_group * model.pos_conv_w->nb[2]);

            // Conv1d with same padding: stride=1, padding=kernel/2, dilation=1
            struct ggml_tensor * out_g = ggml_conv_1d(ctx0, w_g, inp_g, 1, padding, 1);

            // Concatenate along channel dimension (dim=1)
            if (pos_out == nullptr) {
                pos_out = out_g;
            } else {
                pos_out = ggml_concat(ctx0, pos_out, out_g, 1);
            }
        }

        // SamePadLayer: if output is longer than input, trim the last sample
        // This matches HuggingFace's Wav2Vec2SamePadLayer behavior
        int64_t out_len = pos_out->ne[0];
        if (out_len > n_ctx) {
            // Use view to extract first n_ctx samples
            pos_out = ggml_view_3d(ctx0, pos_out,
                n_ctx, n_hidden, 1,
                pos_out->nb[1], pos_out->nb[2], 0);
            pos_out = ggml_cont(ctx0, pos_out);
        }

        // Add bias: reshape bias [1024] to [1, 1024, 1] for broadcasting
        struct ggml_tensor * bias = ggml_reshape_3d(ctx0, model.pos_conv_b, 1, n_hidden, 1);
        pos_out = ggml_add(ctx0, pos_out, bias);

        // GELU activation
        pos_out = ggml_gelu(ctx0, pos_out);

        // pos_out is [n_ctx, 1024, 1], reshape to [n_ctx, 1024]
        pos_out = ggml_reshape_2d(ctx0, pos_out, n_ctx, n_hidden);

        // Transpose back to [1024, n_ctx]
        pos_out = ggml_cont(ctx0, ggml_transpose(ctx0, pos_out));
        ggml_set_output(pos_out);

        // Add to residual (positional embedding is additive)
        cur = ggml_add(ctx0, residual, pos_out);
        ggml_set_output(cur);

    }

    // NOTE: For Wav2Vec2EncoderStableLayerNorm, the encoder layer norm is applied
    // AFTER all transformer layers, not before. This is handled at the end of the loop.

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf((float) n_head_dim);

    // Transformer layers
    for (int il = 0; il < n_layers; ++il) {
        const auto & layer = model.layers[il];

        // Conformer FFN1 branch (residual + 0.5 * FFN1)
        struct ggml_tensor * res_ffn1 = inpL;
        struct ggml_tensor * ffn1_in = inpL;
        if (layer.ffn1_ln_w && layer.ffn1_ln_b) {
            ffn1_in = wav2vec2_layer_norm_affine(ctx0, ffn1_in, layer.ffn1_ln_w, layer.ffn1_ln_b, hparams.eps);
            ggml_set_output(ffn1_in);
        }

        if (layer.ffn1_up_w && layer.ffn1_down_w) {
            struct ggml_tensor * ffn1 = ggml_mul_mat(ctx0, layer.ffn1_up_w, ffn1_in);
            if (layer.ffn1_up_b) {
                ffn1 = ggml_add(ctx0, ffn1, layer.ffn1_up_b);
            }
            // config.hidden_act = swish for this model
            ffn1 = ggml_silu(ctx0, ffn1);
            ffn1 = ggml_mul_mat(ctx0, layer.ffn1_down_w, ffn1);
            if (layer.ffn1_down_b) {
                ffn1 = ggml_add(ctx0, ffn1, layer.ffn1_down_b);
            }

            ffn1 = ggml_scale(ctx0, ffn1, 0.5f);
            inpL = ggml_add(ctx0, res_ffn1, ffn1);
        }

        // Pre-attention layer norm
        if (layer.attn_ln_w && layer.attn_ln_b) {
            cur = wav2vec2_layer_norm_affine(ctx0, inpL, layer.attn_ln_w, layer.attn_ln_b, hparams.eps);
            ggml_set_output(cur);
        } else {
            cur = inpL;
        }

        // Self-attention
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
        if (layer.attn_q_b) Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
        if (layer.attn_k_b) Kcur = ggml_add(ctx0, Kcur, layer.attn_k_b);

        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
        if (layer.attn_v_b) Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);

        // Reshape and permute for multi-head attention
        // Following whisper.cpp pattern:
        // Q, K: [n_head_dim, n_heads, n_ctx] -> permute(0, 2, 1, 3) -> [n_head_dim, n_ctx, n_heads]
        // This allows mul_mat to compute attention scores per head
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_head_dim, n_heads, n_ctx);
        Qcur = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);  // [n_head_dim, n_ctx, n_heads]
        Qcur = ggml_cont(ctx0, Qcur);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_head_dim, n_heads, n_ctx);
        Kcur = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);  // [n_head_dim, n_ctx, n_heads]
        Kcur = ggml_cont(ctx0, Kcur);

        Vcur = ggml_reshape_3d(ctx0, Vcur, n_head_dim, n_heads, n_ctx);
        // V gets different permutation: [n_head_dim, n_heads, n_ctx] -> permute(1, 2, 0, 3) -> [n_heads, n_ctx, n_head_dim]
        Vcur = ggml_permute(ctx0, Vcur, 1, 2, 0, 3);  // [n_heads, n_ctx, n_head_dim]
        Vcur = ggml_cont(ctx0, Vcur);

        // K * Q -> attention scores [n_ctx, n_ctx, n_heads]
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, Kcur, Qcur);
        KQ = ggml_scale(ctx0, KQ, KQscale);

        // relative_key attention bias: einsum("bhld,lrd->bhlr", query, distance_embedding)
        if (layer.attn_distance_emb_w) {
            // [d, n_ctx*n_ctx]
            struct ggml_tensor * rel_pos = ggml_get_rows(ctx0, layer.attn_distance_emb_w, rel_pos_idx);
            // [d, r, l]
            rel_pos = ggml_reshape_3d(ctx0, rel_pos, n_head_dim, n_ctx, n_ctx);

            // Qcur [d, l, h] -> [d, 1, l, h]
            struct ggml_tensor * Q4 = ggml_reshape_4d(ctx0, Qcur, n_head_dim, 1, n_ctx, n_heads);
            // rel_pos [d, r, l] behaves as [d, r, l, 1] for batched mul_mat
            // result: [r, 1, l, h]
            struct ggml_tensor * rel_bias = ggml_mul_mat(ctx0, rel_pos, Q4);
            // [r, l, h]
            rel_bias = ggml_reshape_3d(ctx0, rel_bias, n_ctx, n_ctx, n_heads);
            // same scale as HF
            rel_bias = ggml_scale(ctx0, rel_bias, KQscale);

            KQ = ggml_add(ctx0, KQ, rel_bias);
        }

        KQ = ggml_soft_max(ctx0, KQ);

        // KQ * V -> [n_head_dim, n_ctx, n_heads]
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, Vcur, KQ);
        // Permute back: [n_head_dim, n_ctx, n_heads] -> permute(0, 2, 1, 3) -> [n_head_dim, n_heads, n_ctx]
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_2d(ctx0, KQV, n_hidden, n_ctx);

        // Output projection
        cur = ggml_mul_mat(ctx0, layer.attn_out_w, KQV);
        if (layer.attn_out_b) cur = ggml_add(ctx0, cur, layer.attn_out_b);

        // Residual
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * attn_out = cur;

        // 3) Conformer convolution module: LN -> PW1 -> GLU -> causal DW -> LN -> activation -> PW2 -> residual
        struct ggml_tensor * conv_res = attn_out;
        if (layer.conv_ln_w && layer.conv_ln_b && layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w &&
            layer.conv_dw_ln_w && layer.conv_dw_ln_b) {
            struct ggml_tensor * conv = attn_out;

            // LayerNorm over hidden dim.
                conv = wav2vec2_layer_norm_affine(ctx0, conv, layer.conv_ln_w, layer.conv_ln_b, hparams.eps);

            // [hidden, time] -> [time, hidden, 1]
            conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));
            conv = ggml_reshape_3d(ctx0, conv, n_ctx, n_hidden, 1);

            // PW1 (1x1 conv): hidden -> 2*hidden
            conv = ggml_conv_1d(ctx0, layer.conv_pw1_w, conv, 1, 0, 1);

            // GLU over channel dimension.
            const int64_t t_pw1 = conv->ne[0];
            const int64_t c_pw1 = conv->ne[1];
            const int64_t c_half = c_pw1 / 2;
            struct ggml_tensor * a = ggml_view_3d(ctx0, conv, t_pw1, c_half, 1, conv->nb[1], conv->nb[2], 0);
            struct ggml_tensor * b = ggml_view_3d(ctx0, conv, t_pw1, c_half, 1, conv->nb[1], conv->nb[2], c_half * conv->nb[1]);
            b = ggml_sigmoid(ctx0, b);
            conv = ggml_mul(ctx0, a, b);

            // Causal depthwise conv via symmetric padding + trim first n_ctx frames.
            const int64_t k_dw = layer.conv_dw_w->ne[0];
            const int pad_dw = (int) std::max<int64_t>(0, k_dw - 1);
            conv = ggml_conv_1d_dw(ctx0, layer.conv_dw_w, conv, 1, pad_dw, 1);
            conv = ggml_view_3d(ctx0, conv, n_ctx, n_hidden, 1, conv->nb[1], conv->nb[2], 0);
            conv = ggml_cont(ctx0, conv);

            // Depthwise LN expects [batch,time,hidden], so normalize on [hidden,time].
            conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));
            conv = ggml_cont_2d(ctx0, conv, n_hidden, n_ctx);
                conv = wav2vec2_layer_norm_affine(ctx0, conv, layer.conv_dw_ln_w, layer.conv_dw_ln_b, hparams.eps);

            // Activation (config.hidden_act is usually swish/silu for this model).
            conv = ggml_silu(ctx0, conv);

            // Back to [time, hidden, 1] for PW2.
            conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));
            conv = ggml_reshape_3d(ctx0, conv, n_ctx, n_hidden, 1);

            // PW2 (1x1 conv): hidden -> hidden
            conv = ggml_conv_1d(ctx0, layer.conv_pw2_w, conv, 1, 0, 1);

            // Back to [hidden, time].
            conv = ggml_cont(ctx0, ggml_transpose(ctx0, conv));
            conv = ggml_cont_2d(ctx0, conv, n_hidden, n_ctx);

            conv_res = ggml_add(ctx0, conv_res, conv);
        }

        // Conformer FFN2 branch (residual + 0.5 * FFN2)
        struct ggml_tensor * res_ffn2 = conv_res;
        struct ggml_tensor * ffn2_in = conv_res;
        if (layer.ffn2_ln_w && layer.ffn2_ln_b) {
            ffn2_in = wav2vec2_layer_norm_affine(ctx0, ffn2_in, layer.ffn2_ln_w, layer.ffn2_ln_b, hparams.eps);
            ggml_set_output(ffn2_in);
        }

        if (layer.ffn2_up_w && layer.ffn2_down_w) {
            cur = ggml_mul_mat(ctx0, layer.ffn2_up_w, ffn2_in);
            if (layer.ffn2_up_b) {
                cur = ggml_add(ctx0, cur, layer.ffn2_up_b);
            }
            // config.hidden_act = swish for encoder FFN
            cur = ggml_silu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, layer.ffn2_down_w, cur);
            if (layer.ffn2_down_b) {
                cur = ggml_add(ctx0, cur, layer.ffn2_down_b);
            }

            cur = ggml_scale(ctx0, cur, 0.5f);
            inpL = ggml_add(ctx0, res_ffn2, cur);
        } else {
            inpL = res_ffn2;
        }

        // Final per-layer norm
        if (layer.final_ln_w && layer.final_ln_b) {
            inpL = wav2vec2_layer_norm_affine(ctx0, inpL, layer.final_ln_w, layer.final_ln_b, hparams.eps);
            ggml_set_output(inpL);
        }

    }

    cur = inpL;

    // Encoder layer norm (StableLayerNorm: applied AFTER all transformer layers)
    if (model.enc_ln_w && model.enc_ln_b) {
        cur = wav2vec2_layer_norm_affine(ctx0, cur, model.enc_ln_w, model.enc_ln_b, hparams.eps);
        ggml_set_output(cur);
    }

    // Optional adapter stack (for add_adapter=True models)
    if (!model.adapter_layers.empty()) {
        // Optional projection to output_hidden_size
        if (model.adapter_proj_w) {
            cur = ggml_mul_mat(ctx0, model.adapter_proj_w, cur);
            if (model.adapter_proj_b) {
                cur = ggml_add(ctx0, cur, model.adapter_proj_b);
            }
            if (model.adapter_proj_ln_w && model.adapter_proj_ln_b) {
                cur = wav2vec2_layer_norm_affine(ctx0, cur, model.adapter_proj_ln_w, model.adapter_proj_ln_b, hparams.eps);
            }
        }

        for (size_t ial = 0; ial < model.adapter_layers.size(); ++ial) {
            const auto & al = model.adapter_layers[ial];

            const int64_t n_embed = cur->ne[0];
            const int64_t n_seq = cur->ne[1];

            // Residual branch: LN -> Conv(stride) -> GLU
            struct ggml_tensor * residual = cur;
            if (al.residual_ln_w && al.residual_ln_b) {
                residual = wav2vec2_layer_norm_affine(ctx0, residual, al.residual_ln_w, al.residual_ln_b, hparams.eps);
            }

            residual = ggml_cont(ctx0, ggml_transpose(ctx0, residual));
            residual = ggml_reshape_3d(ctx0, residual, n_seq, n_embed, 1);
            residual = ggml_conv_1d(ctx0, al.residual_conv_w, residual, hparams.adapter_stride, hparams.adapter_stride / 2, 1);
            if (al.residual_conv_b) {
                struct ggml_tensor * rb = ggml_reshape_3d(ctx0, al.residual_conv_b, 1, al.residual_conv_b->ne[0], 1);
                residual = ggml_add(ctx0, residual, rb);
            }
            residual = ggml_cont(ctx0, ggml_transpose(ctx0, residual));
            residual = ggml_cont_2d(ctx0, residual, residual->ne[0], residual->ne[1]);
            residual = wav2vec2_glu_sigmoid(ctx0, residual);

            // Self-attention branch pre-pool
            struct ggml_tensor * hsa = cur;
            if (al.self_attn_ln_w && al.self_attn_ln_b) {
                hsa = wav2vec2_layer_norm_affine(ctx0, hsa, al.self_attn_ln_w, al.self_attn_ln_b, hparams.eps);
            }

            hsa = ggml_cont(ctx0, ggml_transpose(ctx0, hsa));
            hsa = ggml_reshape_3d(ctx0, hsa, n_seq, n_embed, 1);
            hsa = ggml_conv_1d(ctx0, al.self_attn_conv_w, hsa, hparams.adapter_stride, hparams.adapter_stride / 2, 1);
            if (al.self_attn_conv_b) {
                struct ggml_tensor * sb = ggml_reshape_3d(ctx0, al.self_attn_conv_b, 1, al.self_attn_conv_b->ne[0], 1);
                hsa = ggml_add(ctx0, hsa, sb);
            }
            hsa = ggml_cont(ctx0, ggml_transpose(ctx0, hsa));
            hsa = ggml_cont_2d(ctx0, hsa, hsa->ne[0], hsa->ne[1]);
            hsa = wav2vec2_glu_sigmoid(ctx0, hsa);

            // Adapter self-attention (no relative positions)
            const int64_t a_embed = hsa->ne[0];
            const int64_t a_ctx = hsa->ne[1];
            const int64_t a_head_dim = a_embed / n_heads;

            struct ggml_tensor * Q = ggml_mul_mat(ctx0, al.attn_q_w, hsa);
            if (al.attn_q_b) Q = ggml_add(ctx0, Q, al.attn_q_b);
            struct ggml_tensor * K = ggml_mul_mat(ctx0, al.attn_k_w, hsa);
            if (al.attn_k_b) K = ggml_add(ctx0, K, al.attn_k_b);
            struct ggml_tensor * V = ggml_mul_mat(ctx0, al.attn_v_w, hsa);
            if (al.attn_v_b) V = ggml_add(ctx0, V, al.attn_v_b);

            Q = ggml_reshape_3d(ctx0, Q, a_head_dim, n_heads, a_ctx);
            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            Q = ggml_cont(ctx0, Q);
            K = ggml_reshape_3d(ctx0, K, a_head_dim, n_heads, a_ctx);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            K = ggml_cont(ctx0, K);
            V = ggml_reshape_3d(ctx0, V, a_head_dim, n_heads, a_ctx);
            V = ggml_permute(ctx0, V, 1, 2, 0, 3);
            V = ggml_cont(ctx0, V);

            struct ggml_tensor * AKQ = ggml_mul_mat(ctx0, K, Q);
            AKQ = ggml_scale(ctx0, AKQ, 1.0f / sqrtf((float)a_head_dim));
            AKQ = ggml_soft_max(ctx0, AKQ);

            struct ggml_tensor * AKQV = ggml_mul_mat(ctx0, V, AKQ);
            AKQV = ggml_permute(ctx0, AKQV, 0, 2, 1, 3);
            AKQV = ggml_cont_2d(ctx0, AKQV, a_embed, a_ctx);

            struct ggml_tensor * AOUT = ggml_mul_mat(ctx0, al.attn_out_w, AKQV);
            if (al.attn_out_b) AOUT = ggml_add(ctx0, AOUT, al.attn_out_b);

            cur = ggml_add(ctx0, AOUT, residual);

            // Adapter FFN
            struct ggml_tensor * affn = cur;
            if (al.ffn_ln_w && al.ffn_ln_b) {
                affn = wav2vec2_layer_norm_affine(ctx0, affn, al.ffn_ln_w, al.ffn_ln_b, hparams.eps);
            }

            affn = ggml_mul_mat(ctx0, al.ffn_up_w, affn);
            if (al.ffn_up_b) affn = ggml_add(ctx0, affn, al.ffn_up_b);
            // config.adapter_act = relu
            affn = ggml_relu(ctx0, affn);
            affn = ggml_mul_mat(ctx0, al.ffn_down_w, affn);
            if (al.ffn_down_b) affn = ggml_add(ctx0, affn, al.ffn_down_b);

            cur = ggml_add(ctx0, affn, cur);
        }

        ggml_set_output(cur);
    }

    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    wstate.embd_enc = cur;

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

//
// Build CTC head graph
//

static struct ggml_cgraph * wav2vec2_build_graph_ctc(
        wav2vec2_context & wctx,
        wav2vec2_state & wstate,
        int n_ctx) {

    const auto & model = wctx.model;
    const auto & hparams = model.hparams;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_ctc.meta.size(),
        /*.mem_buffer =*/ wstate.sched_ctc.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Input: encoder output [n_hidden, n_ctx]
    const int64_t ctc_in_dim = (wctx.model.adapter_proj_w != nullptr) ? wctx.model.adapter_proj_w->ne[1] : hparams.n_hidden;
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ctc_in_dim, n_ctx);
    ggml_set_name(cur, "ctc_input");
    ggml_set_input(cur);

    // CTC linear projection to vocab
    if (model.ctc_w) {
        cur = ggml_mul_mat(ctx0, model.ctc_w, cur);
        if (model.ctc_b) {
            cur = ggml_add(ctx0, cur, model.ctc_b);
        }
    }

    // Transpose from [n_vocab, n_ctx] to [n_ctx, n_vocab] for CTC decoder
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    ggml_set_name(cur, "ctc_logits");
    ggml_set_output(cur);
    wstate.logits = cur;

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

//
// CTC Decoding
//

static std::vector<wav2vec2_phoneme> ctc_decode_greedy(
        const float * logits,
        int n_frames,
        int n_vocab,
        int blank_id,
        bool merge_repeated,
        int stride_samples) {

    std::vector<wav2vec2_phoneme> result;

    int prev_id = -1;
    int64_t start_frame = 0;

    for (int t = 0; t < n_frames; ++t) {
        // Find argmax
        // GGML uses column-major: logits[t, v] is at position t + v * n_frames
        int max_id = 0;
        float max_logit = logits[t + 0 * n_frames];
        for (int v = 1; v < n_vocab; ++v) {
            float l = logits[t + v * n_frames];
            if (l > max_logit) {
                max_logit = l;
                max_id = v;
            }
        }

        // Compute softmax probability for this token
        float sum_exp = 0.0f;
        for (int v = 0; v < n_vocab; ++v) {
            sum_exp += expf(logits[t + v * n_frames] - max_logit);
        }
        float prob = 1.0f / sum_exp;

        // Skip blank tokens
        if (max_id == blank_id) {
            if (prev_id != -1 && prev_id != blank_id) {
                wav2vec2_phoneme ph;
                ph.id = prev_id;
                ph.prob = prob;
                ph.t0 = start_frame * stride_samples;
                ph.t1 = t * stride_samples;
                result.push_back(ph);
            }
            prev_id = blank_id;
            continue;
        }

        // Merge repeated tokens
        if (merge_repeated && max_id == prev_id) {
            continue;
        }

        // New token
        if (prev_id != max_id && prev_id != -1 && prev_id != blank_id) {
            wav2vec2_phoneme ph;
            ph.id = prev_id;
            ph.prob = prob;
            ph.t0 = start_frame * stride_samples;
            ph.t1 = t * stride_samples;
            result.push_back(ph);
        }

        start_frame = t;
        prev_id = max_id;
    }

    // Handle last token
    if (prev_id != -1 && prev_id != blank_id) {
        wav2vec2_phoneme ph;
        ph.id = prev_id;
        ph.prob = 1.0f;
        ph.t0 = start_frame * stride_samples;
        ph.t1 = n_frames * stride_samples;
        result.push_back(ph);
    }

    return result;
}

//
// Public API Implementation
//

struct wav2vec2_context_params wav2vec2_context_default_params(void) {
    struct wav2vec2_context_params params = {
        /*.use_gpu    =*/ true,
        /*.gpu_device =*/ 0,
    };
    return params;
}

struct wav2vec2_full_params wav2vec2_full_default_params(void) {
    struct wav2vec2_full_params params = {
        /*.n_threads        =*/ 4,
        /*.blank_suppress   =*/ true,
        /*.merge_repeated   =*/ true,
        /*.token_timestamps =*/ true,
    };
    return params;
}

struct wav2vec2_context * wav2vec2_init_from_file(
        const char * path_model,
        struct wav2vec2_context_params params) {

    wav2vec2_context * ctx = new wav2vec2_context();
    ctx->params = params;
    ctx->path_model = path_model;

    if (!wav2vec2_model_load(path_model, *ctx)) {
        W2V_LOG_ERROR("%s: failed to load model from '%s'\n", __func__, path_model);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

struct wav2vec2_context * wav2vec2_init_from_buffer(
        void * buffer,
        size_t buffer_size,
        struct wav2vec2_context_params params) {
    (void) buffer;
    (void) buffer_size;
    (void) params;
    W2V_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

void wav2vec2_free(struct wav2vec2_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->state) {
        wav2vec2_free_state(ctx->state);
    }

    for (auto & buffer : ctx->model.buffers) {
        ggml_backend_buffer_free(buffer);
    }

    for (auto & ctx_ggml : ctx->model.ctxs) {
        ggml_free(ctx_ggml);
    }

    delete ctx;
}

struct wav2vec2_state * wav2vec2_init_state(struct wav2vec2_context * ctx) {
    if (!ctx) return nullptr;

    wav2vec2_state * state = new wav2vec2_state();

    if (ctx->state) {
        state->backends = ctx->state->backends;
    }

    state->sched_conv.meta.resize(ggml_tensor_overhead() * W2V_MAX_NODES + ggml_graph_overhead());
    state->sched_encode.meta.resize(ggml_tensor_overhead() * W2V_MAX_NODES + ggml_graph_overhead());
    state->sched_ctc.meta.resize(ggml_tensor_overhead() * W2V_MAX_NODES + ggml_graph_overhead());

    return state;
}

void wav2vec2_free_state(struct wav2vec2_state * state) {
    if (state == nullptr) {
        return;
    }

    if (state->sched_conv.sched) {
        ggml_backend_sched_free(state->sched_conv.sched);
        state->sched_conv.sched = nullptr;
    }
    if (state->sched_encode.sched) {
        ggml_backend_sched_free(state->sched_encode.sched);
        state->sched_encode.sched = nullptr;
    }
    if (state->sched_ctc.sched) {
        ggml_backend_sched_free(state->sched_ctc.sched);
        state->sched_ctc.sched = nullptr;
    }

    delete state;
}

int wav2vec2_full(
        struct wav2vec2_context * ctx,
        struct wav2vec2_full_params params,
        const float * samples,
        int n_samples) {

    if (!ctx || !ctx->state) {
        return -1;
    }

    return wav2vec2_full_with_state(ctx, ctx->state, params, samples, n_samples);
}

int wav2vec2_full_with_state(
        struct wav2vec2_context * ctx,
        struct wav2vec2_state * state,
        struct wav2vec2_full_params params,
        const float * samples,
        int n_samples) {

    if (!ctx || !state || !samples || n_samples <= 0) {
        return -1;
    }

    const auto & hparams = ctx->model.hparams;

    // Clear previous results
    state->phonemes.clear();

    W2V_LOG_INFO("%s: processing %d samples (%.2f s)\n", __func__,
                 n_samples, (float) n_samples / WAV2VEC2_SAMPLE_RATE);

    // SeamlessM4T frontend: waveform -> 80-bin log-mel -> stride-2 concat to 160 features
    std::vector<float> input_features;
    int n_ctx = 0;
    seamless_m4t_extract_features(samples, n_samples, input_features, n_ctx);
    if (n_ctx < 1) n_ctx = 1;
    state->n_len = n_ctx;

    const int downsample_factor = W2V_FBANK_HOP * W2V_FBANK_STRIDE;
    W2V_LOG_INFO("%s: frontend frames = %d, downsample factor = %d\n", __func__, n_ctx, downsample_factor);

    // Initialize schedulers if needed
    if (!state->sched_conv.sched) {
        state->sched_conv.sched = ggml_backend_sched_new(
            state->backends.data(), nullptr, state->backends.size(), W2V_MAX_NODES, false, true);
    }
    if (!state->sched_encode.sched) {
        state->sched_encode.sched = ggml_backend_sched_new(
            state->backends.data(), nullptr, state->backends.size(), W2V_MAX_NODES, false, true);
    }
    if (!state->sched_ctc.sched) {
        state->sched_ctc.sched = ggml_backend_sched_new(
            state->backends.data(), nullptr, state->backends.size(), W2V_MAX_NODES, false, true);
    }

    const int64_t t_start_us = ggml_time_us();

    // Build and run CNN feature extraction
    {
        struct ggml_cgraph * gf_conv = wav2vec2_build_graph_conv(*ctx, *state, n_ctx);
        if (!ggml_backend_sched_alloc_graph(state->sched_conv.sched, gf_conv)) {
            W2V_LOG_ERROR("%s: failed to allocate conv graph\n", __func__);
            return -1;
        }

        // Set input
        struct ggml_tensor * features_input = ggml_graph_get_tensor(gf_conv, "features_input");
        if (!features_input) {
            W2V_LOG_ERROR("%s: features_input tensor not found\n", __func__);
            return -1;
        }
        ggml_backend_tensor_set(features_input, input_features.data(), 0, input_features.size() * sizeof(float));

        // Compute
        if (!ggml_graph_compute_helper(state->sched_conv.sched, gf_conv, params.n_threads)) {
            W2V_LOG_ERROR("%s: failed to compute conv graph\n", __func__);
            return -1;
        }

        state->t_conv_us = ggml_time_us() - t_start_us;
    }

    // Build and run transformer encoder
    {
        const int64_t t_enc_start = ggml_time_us();

        struct ggml_cgraph * gf_encode = wav2vec2_build_graph_encoder(*ctx, *state, n_ctx);
        if (!ggml_backend_sched_alloc_graph(state->sched_encode.sched, gf_encode)) {
            W2V_LOG_ERROR("%s: failed to allocate encoder graph\n", __func__);
            return -1;
        }

        struct ggml_tensor * encoder_input = ggml_graph_get_tensor(gf_encode, "encoder_input");
        struct ggml_tensor * rel_pos_indices = ggml_graph_get_tensor(gf_encode, "relative_pos_indices");

        if (!encoder_input) {
            W2V_LOG_ERROR("%s: encoder_input tensor not found\n", __func__);
            return -1;
        }

        // Copy conv output to encoder input
        if (state->embd_conv) {
            ggml_backend_tensor_copy(state->embd_conv, encoder_input);
        }

        // relative_key attention indices: idx[l * n_ctx + r] = clamp((r - l), -left, right) + left
        if (rel_pos_indices) {
            const int32_t left = hparams.left_max_position_embeddings;
            const int32_t right = hparams.right_max_position_embeddings;
            state->rel_pos_idx_buf.resize((size_t) n_ctx * (size_t) n_ctx);
            std::vector<int32_t> & idx = state->rel_pos_idx_buf;
            for (int l = 0; l < n_ctx; ++l) {
                for (int r = 0; r < n_ctx; ++r) {
                    int d = r - l;
                    d = std::max(-left, std::min(right, d));
                    idx[l * n_ctx + r] = d + left;
                }
            }
            ggml_backend_tensor_set(rel_pos_indices, idx.data(), 0, idx.size() * sizeof(int32_t));
        }

        if (!ggml_graph_compute_helper(state->sched_encode.sched, gf_encode, params.n_threads)) {
            W2V_LOG_ERROR("%s: failed to compute encoder graph\n", __func__);
            return -1;
        }

        state->t_encode_us = ggml_time_us() - t_enc_start;
    }

    // Build and run CTC head
    {
        const int64_t t_ctc_start = ggml_time_us();
        const int n_ctx_enc = state->embd_enc ? (int) state->embd_enc->ne[1] : n_ctx;
        state->n_len = n_ctx_enc;

        struct ggml_cgraph * gf_ctc = wav2vec2_build_graph_ctc(*ctx, *state, n_ctx_enc);
        if (!ggml_backend_sched_alloc_graph(state->sched_ctc.sched, gf_ctc)) {
            W2V_LOG_ERROR("%s: failed to allocate CTC graph\n", __func__);
            return -1;
        }

        struct ggml_tensor * ctc_input = ggml_graph_get_tensor(gf_ctc, "ctc_input");
        if (!ctc_input) {
            W2V_LOG_ERROR("%s: ctc_input tensor not found\n", __func__);
            return -1;
        }

        // Copy encoder output to CTC input
        if (state->embd_enc) {
            ggml_backend_tensor_copy(state->embd_enc, ctc_input);
        }

        if (!ggml_graph_compute_helper(state->sched_ctc.sched, gf_ctc, params.n_threads)) {
            W2V_LOG_ERROR("%s: failed to compute CTC graph\n", __func__);
            return -1;
        }

        // Get logits
        state->logits_buf.resize(n_ctx_enc * hparams.n_vocab);
        ggml_backend_tensor_get(state->logits, state->logits_buf.data(), 0,
                                 state->logits_buf.size() * sizeof(float));

        state->t_ctc_us = ggml_time_us() - t_ctc_start;
    }

    // CTC decoding
    state->phonemes = ctc_decode_greedy(
        state->logits_buf.data(),
        state->n_len,
        hparams.n_vocab,
        ctx->vocab.token_blank,
        params.merge_repeated,
        downsample_factor
    );

    W2V_LOG_INFO("%s: decoded %d phonemes\n", __func__, (int) state->phonemes.size());
    W2V_LOG_INFO("%s: conv = %.2f ms, encode = %.2f ms, ctc = %.2f ms\n",
                 __func__,
                 state->t_conv_us / 1000.0,
                 state->t_encode_us / 1000.0,
                 state->t_ctc_us / 1000.0);

    return 0;
}

int wav2vec2_full_n_phonemes(struct wav2vec2_context * ctx) {
    if (!ctx || !ctx->state) return 0;
    return (int) ctx->state->phonemes.size();
}

int wav2vec2_full_n_phonemes_from_state(struct wav2vec2_state * state) {
    if (!state) return 0;
    return (int) state->phonemes.size();
}

wav2vec2_phoneme_data wav2vec2_full_get_phoneme_data(struct wav2vec2_context * ctx, int i) {
    wav2vec2_phoneme_data data = {0, 0.0f, 0, 0};
    if (!ctx || !ctx->state || i < 0 || i >= (int) ctx->state->phonemes.size()) {
        return data;
    }
    const auto & ph = ctx->state->phonemes[i];
    data.id = ph.id;
    data.p = ph.prob;
    data.t0 = ph.t0;
    data.t1 = ph.t1;
    return data;
}

wav2vec2_phoneme_data wav2vec2_full_get_phoneme_data_from_state(struct wav2vec2_state * state, int i) {
    wav2vec2_phoneme_data data = {0, 0.0f, 0, 0};
    if (!state || i < 0 || i >= (int) state->phonemes.size()) {
        return data;
    }
    const auto & ph = state->phonemes[i];
    data.id = ph.id;
    data.p = ph.prob;
    data.t0 = ph.t0;
    data.t1 = ph.t1;
    return data;
}

const char * wav2vec2_full_get_phoneme_text(struct wav2vec2_context * ctx, int i) {
    if (!ctx || !ctx->state || i < 0 || i >= (int) ctx->state->phonemes.size()) {
        return "";
    }
    static thread_local std::string rendered;
    wav2vec2_token id = ctx->state->phonemes[i].id;
    rendered = wav2vec2_render_token(*ctx, id);
    return rendered.c_str();
}

const char * wav2vec2_full_get_phoneme_text_from_state(struct wav2vec2_state * state, int i) {
    (void) state;
    (void) i;
    return "";
}

char * wav2vec2_full_get_all_phonemes(struct wav2vec2_context * ctx) {
    if (!ctx || !ctx->state) {
        return strdup("");
    }

    std::string result;
    for (size_t i = 0; i < ctx->state->phonemes.size(); ++i) {
        const wav2vec2_token id = ctx->state->phonemes[i].id;
        result += wav2vec2_render_token(*ctx, id);
    }

    return strdup(result.c_str());
}

char * wav2vec2_full_get_all_phonemes_from_state(struct wav2vec2_state * state) {
    (void) state;
    return strdup("");
}

int wav2vec2_encode(
        struct wav2vec2_context * ctx,
        struct wav2vec2_state * state,
        const float * samples,
        int n_samples,
        int n_threads) {
    (void) ctx;
    (void) state;
    (void) samples;
    (void) n_samples;
    (void) n_threads;
    return 0;
}

int wav2vec2_n_len(struct wav2vec2_context * ctx) {
    if (!ctx || !ctx->state) return 0;
    return ctx->state->n_len;
}

int wav2vec2_n_len_from_state(struct wav2vec2_state * state) {
    if (!state) return 0;
    return state->n_len;
}

int wav2vec2_n_vocab(struct wav2vec2_context * ctx) {
    if (!ctx) return 0;
    return ctx->vocab.n_vocab;
}

int wav2vec2_n_layers(struct wav2vec2_context * ctx) {
    if (!ctx) return 0;
    return ctx->model.hparams.n_layers;
}

int wav2vec2_n_hidden(struct wav2vec2_context * ctx) {
    if (!ctx) return 0;
    return ctx->model.hparams.n_hidden;
}

const char * wav2vec2_token_to_str(struct wav2vec2_context * ctx, wav2vec2_token token) {
    if (!ctx) return "";
    auto it = ctx->vocab.id_to_token.find(token);
    if (it == ctx->vocab.id_to_token.end()) {
        return "";
    }
    return it->second.c_str();
}

wav2vec2_token wav2vec2_str_to_token(struct wav2vec2_context * ctx, const char * phoneme) {
    if (!ctx || !phoneme) return -1;
    auto it = ctx->vocab.token_to_id.find(phoneme);
    if (it == ctx->vocab.token_to_id.end()) {
        return -1;
    }
    return it->second;
}

wav2vec2_token wav2vec2_token_blank(struct wav2vec2_context * ctx) {
    if (!ctx) return 0;
    return ctx->vocab.token_blank;
}

wav2vec2_token wav2vec2_token_pad(struct wav2vec2_context * ctx) {
    if (!ctx) return 0;
    return ctx->vocab.token_pad;
}

wav2vec2_token wav2vec2_token_unk(struct wav2vec2_context * ctx) {
    if (!ctx) return 1;
    return ctx->vocab.token_unk;
}

int64_t wav2vec2_samples_to_ms(int64_t samples) {
    return samples * 1000 / WAV2VEC2_SAMPLE_RATE;
}

int64_t wav2vec2_ms_to_samples(int64_t ms) {
    return ms * WAV2VEC2_SAMPLE_RATE / 1000;
}

const char * wav2vec2_print_system_info(void) {
    static std::string s;
    s = "";

    s += "backends: ";
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (i > 0) s += ", ";
        s += name;
    }

    return s.c_str();
}

//
// Streaming API Implementation
//

struct wav2vec2_stream_state {
    wav2vec2_state * state;              // Underlying inference state
    wav2vec2_stream_params params;

    // Audio buffering (growing window)
    std::vector<float> audio_buffer;     // Current window of audio samples
    int64_t total_samples_received;      // Total samples fed to stream
    int64_t window_start_offset;         // Sample offset of window start (for sliding)

    // All phonemes emitted so far (accumulated across all process calls)
    std::vector<wav2vec2_phoneme> all_phonemes;

    // New phonemes from last process/finalize call only
    std::vector<wav2vec2_phoneme> new_phonemes;

    // Previous inference result (to diff against)
    int prev_phoneme_count;              // Number of phonemes from previous run

    // Processing state
    int downsample_factor;               // Cached downsample factor
};

struct wav2vec2_stream_params wav2vec2_stream_default_params(void) {
    struct wav2vec2_stream_params params = {
        /*.n_threads          =*/ 4,
        /*.min_samples        =*/ 16000,   // 1s at 16kHz - start processing early
        /*.max_window_samples =*/ 160000,  // 10s at 16kHz - max window before sliding
        /*.step_samples       =*/ 16000,   // 1s step when sliding
        /*.merge_repeated     =*/ true,
        /*.blank_suppress     =*/ true,
    };
    return params;
}

struct wav2vec2_stream_state * wav2vec2_stream_init(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_params params) {

    if (!ctx) {
        W2V_LOG_ERROR("%s: context is null\n", __func__);
        return nullptr;
    }

    wav2vec2_stream_state * stream = new wav2vec2_stream_state();

    // Create a new state for streaming (separate from ctx->state)
    stream->state = wav2vec2_init_state(ctx);
    if (!stream->state) {
        W2V_LOG_ERROR("%s: failed to create inference state\n", __func__);
        delete stream;
        return nullptr;
    }

    // Copy backends from context state if available
    if (ctx->state) {
        stream->state->backends = ctx->state->backends;
    }

    stream->params = params;
    stream->total_samples_received = 0;
    stream->window_start_offset = 0;
    stream->prev_phoneme_count = 0;

    // SeamlessM4T frontend (hop=160, stride=2) maps one frame to 320 samples.
    stream->downsample_factor = W2V_FBANK_HOP * W2V_FBANK_STRIDE;

    W2V_LOG_INFO("%s: initialized streaming with min=%d, max_window=%d, step=%d, downsample=%d\n",
                 __func__, params.min_samples, params.max_window_samples, params.step_samples,
                 stream->downsample_factor);

    return stream;
}

void wav2vec2_stream_free(struct wav2vec2_stream_state * stream) {
    if (!stream) return;

    if (stream->state) {
        wav2vec2_free_state(stream->state);
    }

    delete stream;
}

void wav2vec2_stream_reset(struct wav2vec2_stream_state * stream) {
    if (!stream) return;

    stream->audio_buffer.clear();
    stream->all_phonemes.clear();
    stream->new_phonemes.clear();
    stream->total_samples_received = 0;
    stream->window_start_offset = 0;
    stream->prev_phoneme_count = 0;
}

// Internal: Run inference on current window and extract new phonemes
static int wav2vec2_stream_run_inference(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream) {

    // Create full params from stream params
    struct wav2vec2_full_params fparams = {
        /*.n_threads        =*/ stream->params.n_threads,
        /*.blank_suppress   =*/ stream->params.blank_suppress,
        /*.merge_repeated   =*/ stream->params.merge_repeated,
        /*.token_timestamps =*/ true,
    };

    // Run inference on current window
    int ret = wav2vec2_full_with_state(ctx, stream->state, fparams,
                                        stream->audio_buffer.data(),
                                        (int)stream->audio_buffer.size());
    if (ret != 0) {
        W2V_LOG_ERROR("%s: inference failed\n", __func__);
        return -1;
    }

    // Get all phonemes from this run
    int n_phonemes_now = wav2vec2_full_n_phonemes_from_state(stream->state);

    // Clear new phonemes
    stream->new_phonemes.clear();

    // Update all_phonemes to reflect current inference (with offset adjustment)
    // Note: transformer context means earlier phonemes may change as we add audio
    stream->all_phonemes.clear();
    for (int i = 0; i < n_phonemes_now; ++i) {
        const auto & ph = stream->state->phonemes[i];

        // Adjust timestamps by window start offset
        wav2vec2_phoneme new_ph;
        new_ph.id = ph.id;
        new_ph.prob = ph.prob;
        new_ph.t0 = ph.t0 + stream->window_start_offset;
        new_ph.t1 = ph.t1 + stream->window_start_offset;

        stream->all_phonemes.push_back(new_ph);

        // New phonemes are those beyond what we had in previous run
        if (i >= stream->prev_phoneme_count) {
            stream->new_phonemes.push_back(new_ph);
        }
    }

    // Update previous count for next run
    stream->prev_phoneme_count = n_phonemes_now;

    return (int)stream->new_phonemes.size();
}

int wav2vec2_stream_process(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream,
        const float * samples,
        int n_samples) {

    if (!ctx || !stream || !samples || n_samples <= 0) {
        return -1;
    }

    // Append new samples to buffer
    stream->audio_buffer.insert(stream->audio_buffer.end(), samples, samples + n_samples);
    stream->total_samples_received += n_samples;

    // Clear new phonemes for this call
    stream->new_phonemes.clear();

    const int min_samples = stream->params.min_samples;
    const int max_window = stream->params.max_window_samples;
    const int step = stream->params.step_samples;

    // Not enough samples yet - wait for more
    if ((int)stream->audio_buffer.size() < min_samples) {
        return 0;
    }

    // Check if we need to slide the window
    if ((int)stream->audio_buffer.size() > max_window) {
        // How much to slide
        int excess = (int)stream->audio_buffer.size() - max_window;
        int slide_amount = ((excess + step - 1) / step) * step;  // Round up to step boundary

        // Remove old samples from beginning
        stream->audio_buffer.erase(
            stream->audio_buffer.begin(),
            stream->audio_buffer.begin() + slide_amount
        );

        // Update window offset
        stream->window_start_offset += slide_amount;

        // When sliding, we lose early audio so we reset prev_phoneme_count
        // The all_phonemes will be replaced with current window's phonemes
        stream->prev_phoneme_count = 0;
    }

    // Growing window mode - just run inference and find new phonemes at the end
    return wav2vec2_stream_run_inference(ctx, stream);
}

int wav2vec2_stream_finalize(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream) {

    if (!ctx || !stream) {
        return -1;
    }

    // Process any remaining audio in the buffer
    if (stream->audio_buffer.empty()) {
        stream->new_phonemes.clear();
        return 0;
    }

    int n_remaining = (int) stream->audio_buffer.size();

    // Minimum samples needed for inference (at least 1 output frame)
    int min_samples_needed = stream->downsample_factor;
    if (n_remaining < min_samples_needed) {
        // Pad with zeros to get at least one frame
        stream->audio_buffer.resize(min_samples_needed, 0.0f);
    }

    // Run final inference to get any remaining phonemes
    int n_new = wav2vec2_stream_run_inference(ctx, stream);

    // Clear buffer
    stream->audio_buffer.clear();

    return n_new;
}

int wav2vec2_stream_n_phonemes(struct wav2vec2_stream_state * stream) {
    if (!stream) return 0;
    return (int) stream->new_phonemes.size();
}

wav2vec2_phoneme_data wav2vec2_stream_get_phoneme_data(
        struct wav2vec2_stream_state * stream,
        int i) {

    wav2vec2_phoneme_data data = {0, 0.0f, 0, 0};
    if (!stream || i < 0 || i >= (int) stream->new_phonemes.size()) {
        return data;
    }

    const auto & ph = stream->new_phonemes[i];
    data.id = ph.id;
    data.p = ph.prob;
    data.t0 = ph.t0;
    data.t1 = ph.t1;
    return data;
}

const char * wav2vec2_stream_get_phoneme_text(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream,
        int i) {

    if (!ctx || !stream || i < 0 || i >= (int) stream->new_phonemes.size()) {
        return "";
    }

    wav2vec2_token id = stream->new_phonemes[i].id;
    return wav2vec2_token_to_str(ctx, id);
}

int wav2vec2_stream_n_all_phonemes(struct wav2vec2_stream_state * stream) {
    if (!stream) return 0;
    return (int) stream->all_phonemes.size();
}

wav2vec2_phoneme_data wav2vec2_stream_get_all_phoneme_data(
        struct wav2vec2_stream_state * stream,
        int i) {

    wav2vec2_phoneme_data data = {0, 0.0f, 0, 0};
    if (!stream || i < 0 || i >= (int) stream->all_phonemes.size()) {
        return data;
    }

    const auto & ph = stream->all_phonemes[i];
    data.id = ph.id;
    data.p = ph.prob;
    data.t0 = ph.t0;
    data.t1 = ph.t1;
    return data;
}

const char * wav2vec2_stream_get_all_phoneme_text(
        struct wav2vec2_context * ctx,
        struct wav2vec2_stream_state * stream,
        int i) {

    if (!ctx || !stream || i < 0 || i >= (int) stream->all_phonemes.size()) {
        return "";
    }

    wav2vec2_token id = stream->all_phonemes[i].id;
    return wav2vec2_token_to_str(ctx, id);
}
