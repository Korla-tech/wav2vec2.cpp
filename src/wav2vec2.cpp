// wav2vec2.cpp - Wav2Vec2 phoneme recognition implementation for whisper.cpp
//
// This implements the Wav2Vec2 architecture for phoneme recognition using CTC decoding.
// Based on the HuggingFace wav2vec2-xlsr-53-espeak-cv-ft model.

#include "wav2vec2.h"
#include "wav2vec2-arch.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
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

//
// Hyperparameters
//

struct wav2vec2_hparams {
    int32_t n_hidden       = 1024;  // hidden_size
    int32_t n_layers       = 24;    // num_hidden_layers
    int32_t n_heads        = 16;    // num_attention_heads
    int32_t n_intermediate = 4096;  // intermediate_size
    int32_t n_vocab        = 392;   // vocab_size (phonemes)
    int32_t n_conv_layers  = 7;     // number of CNN feature extractor layers

    // CNN config
    int32_t conv_dim[W2V_MAX_CONV_LAYERS]    = {512, 512, 512, 512, 512, 512, 512};
    int32_t conv_kernel[W2V_MAX_CONV_LAYERS] = {10, 3, 3, 3, 3, 2, 2};
    int32_t conv_stride[W2V_MAX_CONV_LAYERS] = {5, 2, 2, 2, 2, 2, 2};

    // Positional conv embedding config
    int32_t num_conv_pos_embeddings = 128;
    int32_t num_conv_pos_embedding_groups = 16;

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

    // Attention layer norm
    struct ggml_tensor * attn_ln_w;
    struct ggml_tensor * attn_ln_b;

    // Feed forward
    struct ggml_tensor * ffn_up_w;    // intermediate_dense
    struct ggml_tensor * ffn_up_b;
    struct ggml_tensor * ffn_down_w;  // output_dense
    struct ggml_tensor * ffn_down_b;

    // FFN layer norm (final_layer_norm)
    struct ggml_tensor * ffn_ln_w;
    struct ggml_tensor * ffn_ln_b;
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
    if (magic != 0x77766332) {  // "wv2c"
        W2V_LOG_ERROR("%s: invalid magic number: 0x%08x (expected 0x77766332)\n", __func__, magic);
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

    // Read CNN config
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        fin.read((char *) &hparams.conv_dim[i], sizeof(hparams.conv_dim[i]));
    }
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        fin.read((char *) &hparams.conv_kernel[i], sizeof(hparams.conv_kernel[i]));
    }
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        fin.read((char *) &hparams.conv_stride[i], sizeof(hparams.conv_stride[i]));
    }

    // Additional config
    fin.read((char *) &hparams.num_conv_pos_embeddings, sizeof(hparams.num_conv_pos_embeddings));
    fin.read((char *) &hparams.num_conv_pos_embedding_groups, sizeof(hparams.num_conv_pos_embedding_groups));
    fin.read((char *) &hparams.ftype, sizeof(hparams.ftype));

    W2V_LOG_INFO("%s: n_hidden       = %d\n", __func__, hparams.n_hidden);
    W2V_LOG_INFO("%s: n_layers       = %d\n", __func__, hparams.n_layers);
    W2V_LOG_INFO("%s: n_heads        = %d\n", __func__, hparams.n_heads);
    W2V_LOG_INFO("%s: n_intermediate = %d\n", __func__, hparams.n_intermediate);
    W2V_LOG_INFO("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
    W2V_LOG_INFO("%s: n_conv_layers  = %d\n", __func__, hparams.n_conv_layers);
    W2V_LOG_INFO("%s: ftype          = %d\n", __func__, hparams.ftype);

    wctx.wtype = hparams.ftype == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32;
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

        // Set special tokens
        if (vocab.token_to_id.count("<pad>")) {
            vocab.token_pad = vocab.token_to_id["<pad>"];
            vocab.token_blank = vocab.token_pad;
        }
        if (vocab.token_to_id.count("<unk>")) {
            vocab.token_unk = vocab.token_to_id["<unk>"];
        }
    }

    // Calculate buffer sizes
    size_t ctx_size = 0;
    {
        const int n_conv = hparams.n_conv_layers;
        const int n_layer = hparams.n_layers;
        const int n_hidden = hparams.n_hidden;
        const int n_inter = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_conv * (512 * 512 * 10 + 512 * 4) * ggml_type_size(wctx.wtype);
        ctx_size += n_hidden * 512 * ggml_type_size(wctx.wtype);
        ctx_size += n_hidden * 128 * 16 * ggml_type_size(wctx.wtype);
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
    model.conv_layers.resize(hparams.n_conv_layers);
    model.layers.resize(hparams.n_layers);

    W2V_LOG_INFO("%s: reading tensors...\n", __func__);

    // Read tensors
    model.n_loaded = 0;
    int tensor_idx = 0;

    while (true) {
        int32_t n_dims;
        int32_t name_len;
        int32_t ftype;

        W2V_LOG_INFO("%s: reading tensor header %d at pos %lld...\n", __func__, tensor_idx, (long long)fin.tellg());

        fin.read((char *) &n_dims,   sizeof(n_dims));
        if (fin.eof()) {
            W2V_LOG_INFO("%s: reached EOF at tensor %d\n", __func__, tensor_idx);
            break;
        }
        if (!fin.good()) {
            W2V_LOG_ERROR("%s: error reading tensor %d\n", __func__, tensor_idx);
            break;
        }

        fin.read((char *) &name_len, sizeof(name_len));
        fin.read((char *) &ftype,    sizeof(ftype));

        W2V_LOG_INFO("%s: tensor %d: n_dims=%d, name_len=%d, ftype=%d\n",
                     __func__, tensor_idx, n_dims, name_len, ftype);

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

        W2V_LOG_INFO("%s: loaded tensor '%s' [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] (%zu bytes)\n",
                      __func__, name.c_str(), (int64_t)ne[0], (int64_t)ne[1], (int64_t)ne[2], (int64_t)ne[3], nbytes);
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
    W2V_LOG_INFO("%s: seeking to tensor data at position %lld (stream good=%d)\n",
                 __func__, (long long)wctx.vocab_end_pos, (int)fin.good());

    // Read tensor data
    int tensors_read = 0;
    W2V_LOG_INFO("%s: starting tensor data read loop, n_loaded=%d\n", __func__, model.n_loaded);

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

        W2V_LOG_INFO("%s: reading tensor %d data at pos %lld: n_dims=%d name_len=%d ftype=%d\n",
                     __func__, tensors_read, (long long)pos_before, n_dims, name_len, ftype);

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

    // Map CNN layers
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        auto & layer = model.conv_layers[i];
        char buf[256];

        snprintf(buf, sizeof(buf), "wav2vec2.feature_extractor.conv_layers.%d.conv.weight", i);
        layer.conv_w = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2.feature_extractor.conv_layers.%d.conv.bias", i);
        layer.conv_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2.feature_extractor.conv_layers.%d.layer_norm.weight", i);
        layer.ln_w = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2.feature_extractor.conv_layers.%d.layer_norm.bias", i);
        layer.ln_b = get_tensor(buf);
    }

    // Feature projection
    model.feat_proj_w = get_tensor("wav2vec2.feature_projection.projection.weight");
    model.feat_proj_b = get_tensor("wav2vec2.feature_projection.projection.bias");
    model.feat_ln_w = get_tensor("wav2vec2.feature_projection.layer_norm.weight");
    model.feat_ln_b = get_tensor("wav2vec2.feature_projection.layer_norm.bias");

    // Debug: print shapes of feature projection/layer norm tensors
    auto debug_tensor_info = [](const char * name, struct ggml_tensor * t) {
        if (t) {
            W2V_LOG_INFO("  %s: shape [%lld, %lld, %lld, %lld], type %s\n",
                         name, (long long)t->ne[0], (long long)t->ne[1],
                         (long long)t->ne[2], (long long)t->ne[3],
                         ggml_type_name(t->type));
            // Print first few values if it's small (for debugging)
            if (ggml_nelements(t) <= 512 && t->type == GGML_TYPE_F32) {
                std::vector<float> data(ggml_nelements(t));
                ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
                float mn = data[0], mx = data[0], sum = 0;
                for (size_t i = 0; i < data.size(); i++) {
                    if (data[i] < mn) mn = data[i];
                    if (data[i] > mx) mx = data[i];
                    sum += data[i];
                }
                W2V_LOG_INFO("    values: min=%.6f, max=%.6f, mean=%.6f\n", mn, mx, sum / data.size());
            }
        }
    };
    W2V_LOG_INFO("%s: Feature projection tensor shapes:\n", __func__);
    debug_tensor_info("feat_proj_w", model.feat_proj_w);
    debug_tensor_info("feat_proj_b", model.feat_proj_b);
    debug_tensor_info("feat_ln_w", model.feat_ln_w);
    debug_tensor_info("feat_ln_b", model.feat_ln_b);

    // Positional conv
    model.pos_conv_w = get_tensor("wav2vec2.encoder.pos_conv_embed.conv.weight");
    model.pos_conv_b = get_tensor("wav2vec2.encoder.pos_conv_embed.conv.bias");

    // Encoder layer norm
    model.enc_ln_w = get_tensor("wav2vec2.encoder.layer_norm.weight");
    model.enc_ln_b = get_tensor("wav2vec2.encoder.layer_norm.bias");
    W2V_LOG_INFO("%s: Encoder layer norm tensor shapes:\n", __func__);
    debug_tensor_info("enc_ln_w", model.enc_ln_w);
    debug_tensor_info("enc_ln_b", model.enc_ln_b);

    // Map transformer layers
    for (int i = 0; i < hparams.n_layers; ++i) {
        auto & layer = model.layers[i];
        char buf[256];

        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.q_proj.weight", i);
        layer.attn_q_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.q_proj.bias", i);
        layer.attn_q_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.k_proj.weight", i);
        layer.attn_k_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.k_proj.bias", i);
        layer.attn_k_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.v_proj.weight", i);
        layer.attn_v_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.v_proj.bias", i);
        layer.attn_v_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.out_proj.weight", i);
        layer.attn_out_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.attention.out_proj.bias", i);
        layer.attn_out_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.layer_norm.weight", i);
        layer.attn_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.layer_norm.bias", i);
        layer.attn_ln_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.feed_forward.intermediate_dense.weight", i);
        layer.ffn_up_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.feed_forward.intermediate_dense.bias", i);
        layer.ffn_up_b = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.feed_forward.output_dense.weight", i);
        layer.ffn_down_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.feed_forward.output_dense.bias", i);
        layer.ffn_down_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.final_layer_norm.weight", i);
        layer.ffn_ln_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "wav2vec2.encoder.layers.%d.final_layer_norm.bias", i);
        layer.ffn_ln_b = get_tensor(buf);
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
        int n_samples) {

    const auto & model = wctx.model;
    const auto & hparams = model.hparams;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_conv.meta.size(),
        /*.mem_buffer =*/ wstate.sched_conv.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, W2V_MAX_NODES, false);

    // Input audio: [1, n_samples]
    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(inp, "audio_input");
    ggml_set_input(inp);

    struct ggml_tensor * cur = inp;

    // Reshape to [n_samples, 1, 1] for conv1d
    cur = ggml_reshape_3d(ctx0, cur, n_samples, 1, 1);

    // CNN Feature Extractor: 7 conv layers
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        const auto & layer = model.conv_layers[i];
        const int kernel = hparams.conv_kernel[i];
        const int stride = hparams.conv_stride[i];

        if (!layer.conv_w) {
            W2V_LOG_WARN("%s: conv layer %d missing weights\n", __func__, i);
            continue;
        }

        // Conv1D with padding to maintain length / stride
        int padding = kernel / 2;
        cur = ggml_conv_1d(ctx0, layer.conv_w, cur, stride, padding, 1);

        if (layer.conv_b) {
            // Reshape bias from [OC] to [1, OC, 1] for broadcasting with conv output [OL, OC, 1]
            struct ggml_tensor * bias = ggml_reshape_3d(ctx0, layer.conv_b, 1, ggml_nelements(layer.conv_b), 1);
            cur = ggml_add(ctx0, cur, bias);
        }

        // Layer norm: HuggingFace transposes before/after layer norm
        // Conv output is [time, channels, 1], need to normalize along channels (not time)
        if (layer.ln_w && layer.ln_b) {
            // Transpose to [channels, time, 1] so ggml_norm normalizes along channels
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));  // [channels, time, 1]

            cur = ggml_norm(ctx0, cur, hparams.eps);  // normalize along ne[0] which is now channels
            // Reshape ln weights for broadcasting with [channels, time, 1]
            struct ggml_tensor * ln_w = ggml_reshape_3d(ctx0, layer.ln_w, ggml_nelements(layer.ln_w), 1, 1);
            struct ggml_tensor * ln_b = ggml_reshape_3d(ctx0, layer.ln_b, ggml_nelements(layer.ln_b), 1, 1);
            cur = ggml_mul(ctx0, cur, ln_w);
            cur = ggml_add(ctx0, cur, ln_b);

            // Transpose back to [time, channels, 1]
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        }

        // GELU activation
        cur = ggml_gelu(ctx0, cur);

        // Debug: name this output for tracing
        char layer_name[64];
        snprintf(layer_name, sizeof(layer_name), "conv_layer_%d_out", i);
        ggml_set_name(cur, layer_name);
    }

    // Reshape from [n_ctx, 512, 1] to [512, n_ctx] for encoder input
    // First reshape to 2D, then transpose to get [channels, time] layout
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1]);  // [n_ctx, 512]
    cur = ggml_transpose(ctx0, cur);  // [512, n_ctx]
    cur = ggml_cont(ctx0, cur);  // Make contiguous after transpose

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

    // Input: CNN features [512, n_ctx] (512 is the CNN output channels)
    const int n_cnn_out = 512;
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_cnn_out, n_ctx);
    ggml_set_name(cur, "encoder_input");
    ggml_set_input(cur);

    // Feature layer norm (applied to 512-dim CNN output BEFORE projection)
    // cur has shape [512, n_ctx] - ggml_norm normalizes along ne[0]=512 which is features
    if (model.feat_ln_w && model.feat_ln_b) {
        // Debug: print tensor info and first few values
        {
            W2V_LOG_INFO("%s: feat_ln_w shape [%lld, %lld, %lld, %lld], type %d, n_dims %d\n",
                         __func__,
                         (long long)model.feat_ln_w->ne[0], (long long)model.feat_ln_w->ne[1],
                         (long long)model.feat_ln_w->ne[2], (long long)model.feat_ln_w->ne[3],
                         (int)model.feat_ln_w->type, (int)ggml_n_dims(model.feat_ln_w));
            W2V_LOG_INFO("%s: feat_ln_w strides nb=[%zu, %zu, %zu, %zu]\n",
                         __func__,
                         model.feat_ln_w->nb[0], model.feat_ln_w->nb[1],
                         model.feat_ln_w->nb[2], model.feat_ln_w->nb[3]);
            std::vector<float> w_data(ggml_nelements(model.feat_ln_w));
            ggml_backend_tensor_get(model.feat_ln_w, w_data.data(), 0, w_data.size() * sizeof(float));
            W2V_LOG_INFO("%s: feat_ln_w first 5 values: %.6f, %.6f, %.6f, %.6f, %.6f\n",
                         __func__, w_data[0], w_data[1], w_data[2], w_data[3], w_data[4]);
        }

        cur = ggml_norm(ctx0, cur, hparams.eps);
        ggml_set_name(cur, "after_feat_norm");
        ggml_set_output(cur);  // Prevent buffer reuse

        // Apply layer norm: norm * weight + bias (like whisper.cpp)
        cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.feat_ln_w),
                model.feat_ln_b);
        ggml_set_name(cur, "after_feat_ln");
    }

    // Feature projection: project from 512 to n_hidden (1024)
    if (model.feat_proj_w) {
        cur = ggml_mul_mat(ctx0, model.feat_proj_w, cur);
        ggml_set_name(cur, "after_feat_proj_matmul");

        if (model.feat_proj_b) {
            cur = ggml_add(ctx0, cur, model.feat_proj_b);
            ggml_set_name(cur, "after_feat_proj_bias");
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
        ggml_set_name(cur, "pos_conv_input");

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

            if (g == 0) {
                ggml_set_name(out_g, "pos_conv_g0");
            }

            // Concatenate along channel dimension (dim=1)
            if (pos_out == nullptr) {
                pos_out = out_g;
            } else {
                pos_out = ggml_concat(ctx0, pos_out, out_g, 1);
            }
        }

        ggml_set_name(pos_out, "pos_conv_grouped");

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
        ggml_set_name(pos_out, "pos_conv_bias");

        // GELU activation
        pos_out = ggml_gelu(ctx0, pos_out);
        ggml_set_name(pos_out, "pos_conv_gelu");

        // pos_out is [n_ctx, 1024, 1], reshape to [n_ctx, 1024]
        pos_out = ggml_reshape_2d(ctx0, pos_out, n_ctx, n_hidden);

        // Transpose back to [1024, n_ctx]
        pos_out = ggml_cont(ctx0, ggml_transpose(ctx0, pos_out));
        ggml_set_name(pos_out, "pos_conv_out");
        ggml_set_output(pos_out);

        // Add to residual (positional embedding is additive)
        cur = ggml_add(ctx0, residual, pos_out);
        ggml_set_name(cur, "after_pos_conv");
        ggml_set_output(cur);

    }

    // NOTE: For Wav2Vec2EncoderStableLayerNorm, the encoder layer norm is applied
    // AFTER all transformer layers, not before. This is handled at the end of the loop.

    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf((float) n_head_dim);

    // Transformer layers
    for (int il = 0; il < n_layers; ++il) {
        const auto & layer = model.layers[il];

        // Pre-attention layer norm
        if (layer.attn_ln_w && layer.attn_ln_b) {
            cur = ggml_norm(ctx0, inpL, hparams.eps);
            ggml_set_output(cur);  // Prevent buffer reuse
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0, cur, layer.attn_ln_w),
                    layer.attn_ln_b);
            if (il == 0) {
                ggml_set_name(cur, "layer0_attn_ln");
                ggml_set_output(cur);
            }
        } else {
            cur = inpL;
        }

        // Self-attention
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
        if (layer.attn_q_b) Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);
        if (il == 0) {
            ggml_set_name(Qcur, "layer0_Q");
            ggml_set_output(Qcur);
        }

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
        KQ = ggml_soft_max(ctx0, KQ);
        if (il == 0) {
            ggml_set_name(KQ, "layer0_KQ_softmax");
            ggml_set_output(KQ);
        }

        // KQ * V -> [n_head_dim, n_ctx, n_heads]
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, Vcur, KQ);
        // Permute back: [n_head_dim, n_ctx, n_heads] -> permute(0, 2, 1, 3) -> [n_head_dim, n_heads, n_ctx]
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_2d(ctx0, KQV, n_hidden, n_ctx);
        if (il == 0) {
            ggml_set_name(KQV, "layer0_KQV");
            ggml_set_output(KQV);
        }

        // Output projection
        cur = ggml_mul_mat(ctx0, layer.attn_out_w, KQV);
        if (layer.attn_out_b) cur = ggml_add(ctx0, cur, layer.attn_out_b);
        if (il == 0) {
            ggml_set_name(cur, "layer0_attn_out");
            ggml_set_output(cur);
        }

        // Residual
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * attn_out = cur;
        if (il == 0) {
            ggml_set_name(attn_out, "layer0_attn_residual");
            ggml_set_output(attn_out);
        }

        // Pre-FFN layer norm
        if (layer.ffn_ln_w && layer.ffn_ln_b) {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            ggml_set_output(cur);  // Prevent buffer reuse
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0, cur, layer.ffn_ln_w),
                    layer.ffn_ln_b);
            if (il == 0) {
                ggml_set_name(cur, "layer0_ffn_ln");
                ggml_set_output(cur);
            }
        }

        // FFN
        cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur);
        if (layer.ffn_up_b) cur = ggml_add(ctx0, cur, layer.ffn_up_b);
        if (il == 0) {
            ggml_set_name(cur, "layer0_ffn_up");
            ggml_set_output(cur);
        }
        cur = ggml_gelu(ctx0, cur);
        if (il == 0) {
            ggml_set_name(cur, "layer0_ffn_gelu");
            ggml_set_output(cur);
        }
        cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur);
        if (layer.ffn_down_b) cur = ggml_add(ctx0, cur, layer.ffn_down_b);
        if (il == 0) {
            ggml_set_name(cur, "layer0_ffn_down");
            ggml_set_output(cur);
        }

        // Residual
        inpL = ggml_add(ctx0, cur, attn_out);
        if (il == 0) {
            ggml_set_name(inpL, "layer0_output");
            ggml_set_output(inpL);
        }
    }

    cur = inpL;

    // Encoder layer norm (StableLayerNorm: applied AFTER all transformer layers)
    if (model.enc_ln_w && model.enc_ln_b) {
        cur = ggml_norm(ctx0, cur, hparams.eps);
        ggml_set_output(cur);  // Prevent buffer reuse
        cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.enc_ln_w),
                model.enc_ln_b);
        ggml_set_name(cur, "after_final_enc_ln");
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
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_hidden, n_ctx);
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
    }
    if (state->sched_encode.sched) {
        ggml_backend_sched_free(state->sched_encode.sched);
    }
    if (state->sched_ctc.sched) {
        ggml_backend_sched_free(state->sched_ctc.sched);
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

    // Calculate output length after CNN downsampling
    int downsample_factor = 1;
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        downsample_factor *= hparams.conv_stride[i];
    }

    int n_ctx = n_samples / downsample_factor;
    if (n_ctx < 1) n_ctx = 1;
    state->n_len = n_ctx;

    W2V_LOG_INFO("%s: downsample factor = %d, n_ctx = %d\n", __func__, downsample_factor, n_ctx);

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
        auto * gf = wav2vec2_build_graph_conv(*ctx, *state, n_samples);

        if (!ggml_backend_sched_alloc_graph(state->sched_conv.sched, gf)) {
            W2V_LOG_ERROR("%s: failed to allocate conv graph\n", __func__);
            return -1;
        }

        // Set input
        struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "audio_input");
        ggml_backend_tensor_set(inp, samples, 0, n_samples * sizeof(float));

        // Compute
        if (!ggml_graph_compute_helper(state->sched_conv.sched, gf, params.n_threads)) {
            W2V_LOG_ERROR("%s: failed to compute conv graph\n", __func__);
            return -1;
        }

        // Debug: trace each CNN layer output
        auto debug_cnn_layer = [&](const char * name) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (t) {
                size_t n = ggml_nelements(t);
                std::vector<float> data(n);
                ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
                float mn = data[0], mx = data[0];
                for (size_t i = 0; i < n; i++) {
                    if (data[i] < mn) mn = data[i];
                    if (data[i] > mx) mx = data[i];
                }
                W2V_LOG_INFO("%s: %s shape [%lld, %lld, %lld], range [%.4f, %.4f]\n",
                             __func__, name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], mn, mx);
            }
        };
        for (int i = 0; i < hparams.n_conv_layers; i++) {
            char name[64];
            snprintf(name, sizeof(name), "conv_layer_%d_out", i);
            debug_cnn_layer(name);
        }

        state->t_conv_us = ggml_time_us() - t_start_us;
    }

    // Build and run transformer encoder
    {
        const int64_t t_enc_start = ggml_time_us();

        auto * gf = wav2vec2_build_graph_encoder(*ctx, *state, n_ctx);

        if (!ggml_backend_sched_alloc_graph(state->sched_encode.sched, gf)) {
            W2V_LOG_ERROR("%s: failed to allocate encoder graph\n", __func__);
            return -1;
        }

        // Copy conv output to encoder input
        struct ggml_tensor * enc_inp = ggml_graph_get_tensor(gf, "encoder_input");
        if (enc_inp && state->embd_conv) {
            // Need to get conv output data and set it as encoder input
            size_t nbytes = ggml_nbytes(enc_inp);
            std::vector<float> conv_out(nbytes / sizeof(float));
            ggml_backend_tensor_get(state->embd_conv, conv_out.data(), 0, nbytes);

            // Debug: check CNN output statistics
            float conv_min = conv_out[0], conv_max = conv_out[0], conv_sum = 0;
            for (size_t i = 0; i < conv_out.size(); i++) {
                conv_sum += conv_out[i];
                if (conv_out[i] < conv_min) conv_min = conv_out[i];
                if (conv_out[i] > conv_max) conv_max = conv_out[i];
            }
            W2V_LOG_INFO("%s: CNN output range [%.4f, %.4f], mean %.4f, size %zu\n",
                         __func__, conv_min, conv_max, conv_sum / conv_out.size(), conv_out.size());

            ggml_backend_tensor_set(enc_inp, conv_out.data(), 0, nbytes);
        }

        // Debug: check feat_ln_w right before graph execution
        {
            std::vector<float> w_data(ggml_nelements(ctx->model.feat_ln_w));
            ggml_backend_tensor_get(ctx->model.feat_ln_w, w_data.data(), 0, w_data.size() * sizeof(float));
            float sum = 0;
            for (size_t i = 0; i < w_data.size(); i++) sum += w_data[i];
            W2V_LOG_INFO("%s: feat_ln_w before compute: first 5 values = %.6f, %.6f, %.6f, %.6f, %.6f\n",
                         __func__, w_data[0], w_data[1], w_data[2], w_data[3], w_data[4]);
            W2V_LOG_INFO("%s: feat_ln_w sum = %.6f, buffer=%p, data=%p\n", __func__, sum,
                         (void*)ctx->model.feat_ln_w->buffer, (void*)ctx->model.feat_ln_w->data);
        }

        if (!ggml_graph_compute_helper(state->sched_encode.sched, gf, params.n_threads)) {
            W2V_LOG_ERROR("%s: failed to compute encoder graph\n", __func__);
            return -1;
        }

        // Debug: check intermediate tensors at each step
        auto debug_tensor = [&](const char * name, bool print_samples = false) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (t) {
                size_t n = ggml_nelements(t);
                std::vector<float> data(n);
                ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
                float mn = data[0], mx = data[0], sum = 0;
                int nan_count = 0, inf_count = 0;
                for (size_t i = 0; i < n; i++) {
                    if (std::isnan(data[i])) { nan_count++; continue; }
                    if (std::isinf(data[i])) { inf_count++; continue; }
                    if (data[i] < mn) mn = data[i];
                    if (data[i] > mx) mx = data[i];
                    sum += data[i];
                }
                W2V_LOG_INFO("%s: %s: range [%.4f, %.4f], mean %.4f, nan=%d, inf=%d, shape [%lld, %lld]\n",
                             __func__, name, mn, mx, sum / n, nan_count, inf_count,
                             (long long)t->ne[0], (long long)t->ne[1]);
                if (print_samples) {
                    W2V_LOG_INFO("%s:   first 5: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                                 __func__, data[0], data[1], data[2], data[3], data[4]);
                    // Also print first element of each column (time step)
                    if (t->ne[1] > 1) {
                        W2V_LOG_INFO("%s:   [0,0]...[0,4]: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                                     __func__, data[0], data[t->ne[0]], data[2*t->ne[0]],
                                     data[3*t->ne[0]], data[4*t->ne[0]]);
                    }
                }
            }
        };

        debug_tensor("after_feat_norm", true);
        debug_tensor("after_feat_ln", true);
        debug_tensor("after_feat_proj_matmul");
        debug_tensor("after_feat_proj_bias");
        debug_tensor("after_enc_ln");
        debug_tensor("layer0_attn_ln");
        debug_tensor("layer0_Q");
        debug_tensor("layer0_KQ_softmax");
        debug_tensor("layer0_KQV");
        debug_tensor("layer0_attn_out");
        debug_tensor("layer0_attn_residual");
        debug_tensor("layer0_ffn_ln");
        debug_tensor("layer0_ffn_up");
        debug_tensor("layer0_ffn_gelu");
        debug_tensor("layer0_ffn_down");
        debug_tensor("layer0_output");
        debug_tensor("encoder_output");

        state->t_encode_us = ggml_time_us() - t_enc_start;
    }

    // Build and run CTC head
    {
        const int64_t t_ctc_start = ggml_time_us();

        auto * gf = wav2vec2_build_graph_ctc(*ctx, *state, n_ctx);

        if (!ggml_backend_sched_alloc_graph(state->sched_ctc.sched, gf)) {
            W2V_LOG_ERROR("%s: failed to allocate CTC graph\n", __func__);
            return -1;
        }

        // Copy encoder output to CTC input
        struct ggml_tensor * ctc_inp = ggml_graph_get_tensor(gf, "ctc_input");
        if (ctc_inp && state->embd_enc) {
            size_t nbytes = ggml_nbytes(ctc_inp);
            std::vector<float> enc_out(nbytes / sizeof(float));
            ggml_backend_tensor_get(state->embd_enc, enc_out.data(), 0, nbytes);

            // Debug: check encoder output statistics
            float enc_min = enc_out[0], enc_max = enc_out[0], enc_sum = 0;
            for (size_t i = 0; i < enc_out.size(); i++) {
                enc_sum += enc_out[i];
                if (enc_out[i] < enc_min) enc_min = enc_out[i];
                if (enc_out[i] > enc_max) enc_max = enc_out[i];
            }
            W2V_LOG_INFO("%s: encoder output range [%.4f, %.4f], mean %.4f\n",
                         __func__, enc_min, enc_max, enc_sum / enc_out.size());

            ggml_backend_tensor_set(ctc_inp, enc_out.data(), 0, nbytes);
        }

        if (!ggml_graph_compute_helper(state->sched_ctc.sched, gf, params.n_threads)) {
            W2V_LOG_ERROR("%s: failed to compute CTC graph\n", __func__);
            return -1;
        }

        // Get logits
        state->logits_buf.resize(n_ctx * hparams.n_vocab);
        ggml_backend_tensor_get(state->logits, state->logits_buf.data(), 0,
                                 state->logits_buf.size() * sizeof(float));

        state->t_ctc_us = ggml_time_us() - t_ctc_start;
    }

    // Debug: print logits statistics
    {
        float min_logit = state->logits_buf[0], max_logit = state->logits_buf[0];
        float sum = 0;
        std::vector<int> token_counts(hparams.n_vocab, 0);
        for (int t = 0; t < n_ctx; t++) {
            int argmax = 0;
            float max_val = state->logits_buf[t * hparams.n_vocab];
            for (int v = 0; v < hparams.n_vocab; v++) {
                float l = state->logits_buf[t * hparams.n_vocab + v];
                sum += l;
                if (l < min_logit) min_logit = l;
                if (l > max_logit) max_logit = l;
                if (l > max_val) { max_val = l; argmax = v; }
            }
            token_counts[argmax]++;
        }
        W2V_LOG_INFO("%s: logits range [%.2f, %.2f], mean %.4f\n",
                     __func__, min_logit, max_logit, sum / (n_ctx * hparams.n_vocab));

        // Print top 10 most common tokens
        std::vector<std::pair<int, int>> sorted_tokens;
        for (int v = 0; v < hparams.n_vocab; v++) {
            if (token_counts[v] > 0) sorted_tokens.push_back({token_counts[v], v});
        }
        std::sort(sorted_tokens.rbegin(), sorted_tokens.rend());
        W2V_LOG_INFO("%s: top tokens: ", __func__);
        for (int i = 0; i < std::min(10, (int)sorted_tokens.size()); i++) {
            fprintf(stderr, "%d(%d) ", sorted_tokens[i].second, sorted_tokens[i].first);
        }
        fprintf(stderr, "\n");
    }

    // CTC decoding
    state->phonemes = ctc_decode_greedy(
        state->logits_buf.data(),
        n_ctx,
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
    wav2vec2_token id = ctx->state->phonemes[i].id;
    return wav2vec2_token_to_str(ctx, id);
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
        result += wav2vec2_full_get_phoneme_text(ctx, i);
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

    // Calculate downsample factor
    stream->downsample_factor = 1;
    const auto & hparams = ctx->model.hparams;
    for (int i = 0; i < hparams.n_conv_layers; ++i) {
        stream->downsample_factor *= hparams.conv_stride[i];
    }

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
