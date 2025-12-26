#pragma once

#include "ggml.h"

#include <map>
#include <string>

// Wav2Vec2 tensor types
enum w2v_tensor {
    // CNN Feature Extractor
    W2V_TENSOR_CONV_WEIGHT,
    W2V_TENSOR_CONV_BIAS,
    W2V_TENSOR_CONV_LN_WEIGHT,
    W2V_TENSOR_CONV_LN_BIAS,

    // Feature projection
    W2V_TENSOR_FEAT_PROJ_WEIGHT,
    W2V_TENSOR_FEAT_PROJ_BIAS,
    W2V_TENSOR_FEAT_LN_WEIGHT,
    W2V_TENSOR_FEAT_LN_BIAS,

    // Positional conv embedding
    W2V_TENSOR_POS_CONV_WEIGHT,
    W2V_TENSOR_POS_CONV_BIAS,

    // Encoder layers
    W2V_TENSOR_ENC_LN_WEIGHT,
    W2V_TENSOR_ENC_LN_BIAS,
    W2V_TENSOR_ATTN_Q_WEIGHT,
    W2V_TENSOR_ATTN_Q_BIAS,
    W2V_TENSOR_ATTN_K_WEIGHT,
    W2V_TENSOR_ATTN_K_BIAS,
    W2V_TENSOR_ATTN_V_WEIGHT,
    W2V_TENSOR_ATTN_V_BIAS,
    W2V_TENSOR_ATTN_OUT_WEIGHT,
    W2V_TENSOR_ATTN_OUT_BIAS,
    W2V_TENSOR_ATTN_LN_WEIGHT,
    W2V_TENSOR_ATTN_LN_BIAS,
    W2V_TENSOR_FFN_UP_WEIGHT,
    W2V_TENSOR_FFN_UP_BIAS,
    W2V_TENSOR_FFN_DOWN_WEIGHT,
    W2V_TENSOR_FFN_DOWN_BIAS,
    W2V_TENSOR_FFN_LN_WEIGHT,
    W2V_TENSOR_FFN_LN_BIAS,

    // Final layer norm
    W2V_TENSOR_FINAL_LN_WEIGHT,
    W2V_TENSOR_FINAL_LN_BIAS,

    // CTC head (lm_head)
    W2V_TENSOR_CTC_WEIGHT,
    W2V_TENSOR_CTC_BIAS,
};

// Tensor operations mapping (for backend selection)
static const std::map<w2v_tensor, ggml_op> W2V_TENSOR_OPS = {
    // CNN (use im2col for conv1d)
    {W2V_TENSOR_CONV_WEIGHT,       GGML_OP_IM2COL},
    {W2V_TENSOR_CONV_BIAS,         GGML_OP_ADD},
    {W2V_TENSOR_CONV_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_TENSOR_CONV_LN_BIAS,      GGML_OP_ADD},

    // Feature projection
    {W2V_TENSOR_FEAT_PROJ_WEIGHT,  GGML_OP_MUL_MAT},
    {W2V_TENSOR_FEAT_PROJ_BIAS,    GGML_OP_ADD},
    {W2V_TENSOR_FEAT_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_TENSOR_FEAT_LN_BIAS,      GGML_OP_ADD},

    // Positional conv
    {W2V_TENSOR_POS_CONV_WEIGHT,   GGML_OP_IM2COL},
    {W2V_TENSOR_POS_CONV_BIAS,     GGML_OP_ADD},

    // Encoder layer norms
    {W2V_TENSOR_ENC_LN_WEIGHT,     GGML_OP_MUL},
    {W2V_TENSOR_ENC_LN_BIAS,       GGML_OP_ADD},

    // Attention
    {W2V_TENSOR_ATTN_Q_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_TENSOR_ATTN_Q_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_ATTN_K_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_TENSOR_ATTN_K_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_ATTN_V_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_TENSOR_ATTN_V_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_ATTN_OUT_WEIGHT,   GGML_OP_MUL_MAT},
    {W2V_TENSOR_ATTN_OUT_BIAS,     GGML_OP_ADD},
    {W2V_TENSOR_ATTN_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_TENSOR_ATTN_LN_BIAS,      GGML_OP_ADD},

    // FFN
    {W2V_TENSOR_FFN_UP_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_TENSOR_FFN_UP_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_FFN_DOWN_WEIGHT,   GGML_OP_MUL_MAT},
    {W2V_TENSOR_FFN_DOWN_BIAS,     GGML_OP_ADD},
    {W2V_TENSOR_FFN_LN_WEIGHT,     GGML_OP_MUL},
    {W2V_TENSOR_FFN_LN_BIAS,       GGML_OP_ADD},

    // Final layer norm
    {W2V_TENSOR_FINAL_LN_WEIGHT,   GGML_OP_MUL},
    {W2V_TENSOR_FINAL_LN_BIAS,     GGML_OP_ADD},

    // CTC head
    {W2V_TENSOR_CTC_WEIGHT,        GGML_OP_MUL_MAT},
    {W2V_TENSOR_CTC_BIAS,          GGML_OP_ADD},
};

// HuggingFace tensor name patterns
// Note: %d is replaced with layer index
static const std::map<w2v_tensor, const char *> W2V_TENSOR_NAMES = {
    // CNN Feature Extractor (wav2vec2.feature_extractor.conv_layers.X.conv.weight)
    {W2V_TENSOR_CONV_WEIGHT,       "wav2vec2.feature_extractor.conv_layers.%d.conv.weight"},
    {W2V_TENSOR_CONV_BIAS,         "wav2vec2.feature_extractor.conv_layers.%d.conv.bias"},
    {W2V_TENSOR_CONV_LN_WEIGHT,    "wav2vec2.feature_extractor.conv_layers.%d.layer_norm.weight"},
    {W2V_TENSOR_CONV_LN_BIAS,      "wav2vec2.feature_extractor.conv_layers.%d.layer_norm.bias"},

    // Feature projection
    {W2V_TENSOR_FEAT_PROJ_WEIGHT,  "wav2vec2.feature_projection.projection.weight"},
    {W2V_TENSOR_FEAT_PROJ_BIAS,    "wav2vec2.feature_projection.projection.bias"},
    {W2V_TENSOR_FEAT_LN_WEIGHT,    "wav2vec2.feature_projection.layer_norm.weight"},
    {W2V_TENSOR_FEAT_LN_BIAS,      "wav2vec2.feature_projection.layer_norm.bias"},

    // Positional conv embedding
    {W2V_TENSOR_POS_CONV_WEIGHT,   "wav2vec2.encoder.pos_conv_embed.conv.weight"},
    {W2V_TENSOR_POS_CONV_BIAS,     "wav2vec2.encoder.pos_conv_embed.conv.bias"},

    // Encoder layer norm (before transformer layers)
    {W2V_TENSOR_ENC_LN_WEIGHT,     "wav2vec2.encoder.layer_norm.weight"},
    {W2V_TENSOR_ENC_LN_BIAS,       "wav2vec2.encoder.layer_norm.bias"},

    // Encoder layers (wav2vec2.encoder.layers.X.*)
    {W2V_TENSOR_ATTN_Q_WEIGHT,     "wav2vec2.encoder.layers.%d.attention.q_proj.weight"},
    {W2V_TENSOR_ATTN_Q_BIAS,       "wav2vec2.encoder.layers.%d.attention.q_proj.bias"},
    {W2V_TENSOR_ATTN_K_WEIGHT,     "wav2vec2.encoder.layers.%d.attention.k_proj.weight"},
    {W2V_TENSOR_ATTN_K_BIAS,       "wav2vec2.encoder.layers.%d.attention.k_proj.bias"},
    {W2V_TENSOR_ATTN_V_WEIGHT,     "wav2vec2.encoder.layers.%d.attention.v_proj.weight"},
    {W2V_TENSOR_ATTN_V_BIAS,       "wav2vec2.encoder.layers.%d.attention.v_proj.bias"},
    {W2V_TENSOR_ATTN_OUT_WEIGHT,   "wav2vec2.encoder.layers.%d.attention.out_proj.weight"},
    {W2V_TENSOR_ATTN_OUT_BIAS,     "wav2vec2.encoder.layers.%d.attention.out_proj.bias"},
    {W2V_TENSOR_ATTN_LN_WEIGHT,    "wav2vec2.encoder.layers.%d.layer_norm.weight"},
    {W2V_TENSOR_ATTN_LN_BIAS,      "wav2vec2.encoder.layers.%d.layer_norm.bias"},
    {W2V_TENSOR_FFN_UP_WEIGHT,     "wav2vec2.encoder.layers.%d.feed_forward.intermediate_dense.weight"},
    {W2V_TENSOR_FFN_UP_BIAS,       "wav2vec2.encoder.layers.%d.feed_forward.intermediate_dense.bias"},
    {W2V_TENSOR_FFN_DOWN_WEIGHT,   "wav2vec2.encoder.layers.%d.feed_forward.output_dense.weight"},
    {W2V_TENSOR_FFN_DOWN_BIAS,     "wav2vec2.encoder.layers.%d.feed_forward.output_dense.bias"},
    {W2V_TENSOR_FFN_LN_WEIGHT,     "wav2vec2.encoder.layers.%d.final_layer_norm.weight"},
    {W2V_TENSOR_FFN_LN_BIAS,       "wav2vec2.encoder.layers.%d.final_layer_norm.bias"},

    // Final layer norm (after all transformer layers, not always present)
    {W2V_TENSOR_FINAL_LN_WEIGHT,   "wav2vec2.encoder.layer_norm.weight"},
    {W2V_TENSOR_FINAL_LN_BIAS,     "wav2vec2.encoder.layer_norm.bias"},

    // CTC head (lm_head)
    {W2V_TENSOR_CTC_WEIGHT,        "lm_head.weight"},
    {W2V_TENSOR_CTC_BIAS,          "lm_head.bias"},
};

// Helper function to get tensor name with layer index
inline std::string w2v_tensor_name(w2v_tensor tensor, int layer = -1) {
    auto it = W2V_TENSOR_NAMES.find(tensor);
    if (it == W2V_TENSOR_NAMES.end()) {
        return "";
    }

    std::string name = it->second;
    if (layer >= 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), name.c_str(), layer);
        return buf;
    }
    return name;
}
