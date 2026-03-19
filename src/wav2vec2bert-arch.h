#pragma once

#include "ggml.h"

#include <map>
#include <string>

// Wav2Vec2 tensor types
enum w2v_tensor {
    // Feature projection
    W2V_TENSOR_FEAT_PROJ_WEIGHT,
    W2V_TENSOR_FEAT_PROJ_BIAS,
    W2V_TENSOR_FEAT_LN_WEIGHT,
    W2V_TENSOR_FEAT_LN_BIAS,

    // Encoder layers
    // Attention
    W2V_TENSOR_ATTN_Q_WEIGHT,
    W2V_TENSOR_ATTN_Q_BIAS,
    W2V_TENSOR_ATTN_K_WEIGHT,
    W2V_TENSOR_ATTN_K_BIAS,
    W2V_TENSOR_ATTN_V_WEIGHT,
    W2V_TENSOR_ATTN_V_BIAS,
    W2V_TENSOR_ATTN_LN_WEIGHT,
    W2V_TENSOR_ATTN_LN_BIAS,
    W2V_TENSOR_ATTN_OUT_WEIGHT,
    W2V_TENSOR_ATTN_OUT_BIAS,
    W2V_TENSOR_ATTN_DISTANCE_EMBED_WEIGHT,
    // FFN
    W2V_TENSOR_FFN1_UP_WEIGHT,
    W2V_TENSOR_FFN1_UP_BIAS,
    W2V_TENSOR_FFN1_DOWN_WEIGHT,
    W2V_TENSOR_FFN1_DOWN_BIAS,
    W2V_TENSOR_FFN1_LN_WEIGHT,
    W2V_TENSOR_FFN1_LN_BIAS,
    W2V_TENSOR_FFN2_UP_WEIGHT,
    W2V_TENSOR_FFN2_UP_BIAS,
    W2V_TENSOR_FFN2_DOWN_WEIGHT,
    W2V_TENSOR_FFN2_DOWN_BIAS,
    W2V_TENSOR_FFN2_LN_WEIGHT,
    W2V_TENSOR_FFN2_LN_BIAS,
    W2V_TENSOR_FINAL_LN_WEIGHT,
    W2V_TENSOR_FINAL_LN_BIAS,
    // Conv Module
    W2V_TENSOR_DW_CONV_WEIGHT,
    W2V_TENSOR_DW_CONV_LN_BIAS,
    W2V_TENSOR_DW_CONV_LN_WEIGHT,
    W2V_TENSOR_CONV_LN_BIAS,
    W2V_TENSOR_CONV_LN_WEIGHT,
    W2V_TENSOR_PW_CONV1_WEIGHT,
    W2V_TENSOR_PW_CONV2_WEIGHT,

    // Adapter Layers
    W2V_ADAPTER_LN_WEIGHT,
    W2V_ADAPTER_LN_BIAS,
    W2V_ADAPTER_UP_WEIGHT,
    W2V_ADAPTER_UP_BIAS,
    W2V_ADAPTER_DOWN_WEIGHT,
    W2V_ADAPTER_DOWN_BIAS,
    W2V_ADAPTER_CONV_WEIGHT,
    W2V_ADAPTER_CONV_BIAS,
    W2V_ADAPTER_CONV_LN_WEIGHT,
    W2V_ADAPTER_CONV_LN_BIAS,
    W2V_ADAPTER_ATTN_Q_WEIGHT,
    W2V_ADAPTER_ATTN_Q_BIAS,
    W2V_ADAPTER_ATTN_K_WEIGHT,
    W2V_ADAPTER_ATTN_K_BIAS,
    W2V_ADAPTER_ATTN_V_WEIGHT,
    W2V_ADAPTER_ATTN_V_BIAS,
    W2V_ADAPTER_ATTN_CONV_WEIGHT,
    W2V_ADAPTER_ATTN_CONV_BIAS,
    W2V_ADAPTER_ATTN_LN_WEIGHT,
    W2V_ADAPTER_ATTN_LN_BIAS,

    // CTC head (lm_head)
    W2V_TENSOR_CTC_WEIGHT,
    W2V_TENSOR_CTC_BIAS,
};

// Tensor operations mapping (for backend selection)
static const std::map<w2v_tensor, ggml_op> W2V_TENSOR_OPS = {
    // Feature projection
    {W2V_TENSOR_FEAT_PROJ_WEIGHT,  GGML_OP_MUL_MAT},
    {W2V_TENSOR_FEAT_PROJ_BIAS,    GGML_OP_ADD},
    {W2V_TENSOR_FEAT_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_TENSOR_FEAT_LN_BIAS,      GGML_OP_ADD},

    // Encoder layers
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
    {W2V_TENSOR_ATTN_DISTANCE_EMBED_WEIGHT, GGML_OP_MUL_MAT},
    // FFN
    {W2V_TENSOR_FFN1_UP_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_TENSOR_FFN1_UP_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_FFN1_DOWN_WEIGHT,   GGML_OP_MUL_MAT},
    {W2V_TENSOR_FFN1_DOWN_BIAS,     GGML_OP_ADD},
    {W2V_TENSOR_FFN1_LN_WEIGHT,     GGML_OP_MUL},
    {W2V_TENSOR_FFN1_LN_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_FFN2_UP_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_TENSOR_FFN2_UP_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_FFN2_DOWN_WEIGHT,   GGML_OP_MUL_MAT},
    {W2V_TENSOR_FFN2_DOWN_BIAS,     GGML_OP_ADD},
    {W2V_TENSOR_FFN2_LN_WEIGHT,     GGML_OP_MUL},
    {W2V_TENSOR_FFN2_LN_BIAS,       GGML_OP_ADD},
    {W2V_TENSOR_FINAL_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_TENSOR_FINAL_LN_BIAS,      GGML_OP_ADD},
    // Conv Module
    {W2V_TENSOR_DW_CONV_WEIGHT,    GGML_OP_IM2COL},
    {W2V_TENSOR_DW_CONV_LN_BIAS,   GGML_OP_ADD},
    {W2V_TENSOR_DW_CONV_LN_WEIGHT, GGML_OP_MUL},
    {W2V_TENSOR_CONV_LN_BIAS,      GGML_OP_ADD},
    {W2V_TENSOR_CONV_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_TENSOR_PW_CONV1_WEIGHT,   GGML_OP_MUL_MAT},
    {W2V_TENSOR_PW_CONV2_WEIGHT,   GGML_OP_MUL_MAT},

    // Adapter Layers
    {W2V_ADAPTER_LN_WEIGHT,         GGML_OP_MUL},
    {W2V_ADAPTER_LN_BIAS,           GGML_OP_ADD},
    {W2V_ADAPTER_UP_WEIGHT,         GGML_OP_MUL_MAT},
    {W2V_ADAPTER_UP_BIAS,           GGML_OP_ADD},
    {W2V_ADAPTER_DOWN_WEIGHT,       GGML_OP_MUL_MAT},
    {W2V_ADAPTER_DOWN_BIAS,         GGML_OP_ADD},
    {W2V_ADAPTER_CONV_WEIGHT,       GGML_OP_IM2COL},
    {W2V_ADAPTER_CONV_BIAS,         GGML_OP_ADD},
    {W2V_ADAPTER_CONV_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_ADAPTER_CONV_LN_BIAS,      GGML_OP_ADD},
    {W2V_ADAPTER_ATTN_Q_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_ADAPTER_ATTN_Q_BIAS,       GGML_OP_ADD},
    {W2V_ADAPTER_ATTN_K_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_ADAPTER_ATTN_K_BIAS,       GGML_OP_ADD},
    {W2V_ADAPTER_ATTN_V_WEIGHT,     GGML_OP_MUL_MAT},
    {W2V_ADAPTER_ATTN_V_BIAS,       GGML_OP_ADD},
    {W2V_ADAPTER_ATTN_CONV_WEIGHT,  GGML_OP_IM2COL},
    {W2V_ADAPTER_ATTN_CONV_BIAS,    GGML_OP_ADD},
    {W2V_ADAPTER_ATTN_LN_WEIGHT,    GGML_OP_MUL},
    {W2V_ADAPTER_ATTN_LN_BIAS,      GGML_OP_ADD},

    // CTC head
    {W2V_TENSOR_CTC_WEIGHT,        GGML_OP_MUL_MAT},
    {W2V_TENSOR_CTC_BIAS,          GGML_OP_ADD},
};

// HuggingFace tensor name patterns
// Note: %d is replaced with layer index
static const std::map<w2v_tensor, const char *> W2V_TENSOR_NAMES = {
    // Feature projection
    {W2V_TENSOR_FEAT_PROJ_WEIGHT,  "wav2vec2_bert.feature_projection.projection.weight"},
    {W2V_TENSOR_FEAT_PROJ_BIAS,    "wav2vec2_bert.feature_projection.projection.bias"},
    {W2V_TENSOR_FEAT_LN_WEIGHT,    "wav2vec2_bert.feature_projection.layer_norm.weight"},
    {W2V_TENSOR_FEAT_LN_BIAS,      "wav2vec2_bert.feature_projection.layer_norm.bias"},

    // Encoder layers (wav2vec2_bert.encoder.layers.X.*)
    {W2V_TENSOR_ATTN_Q_WEIGHT,     "wav2vec2_bert.encoder.layers.%d.self_attn.linear_q.weight"},
    {W2V_TENSOR_ATTN_Q_BIAS,       "wav2vec2_bert.encoder.layers.%d.self_attn.linear_q.bias"},
    {W2V_TENSOR_ATTN_K_WEIGHT,     "wav2vec2_bert.encoder.layers.%d.self_attn.linear_k.weight"},
    {W2V_TENSOR_ATTN_K_BIAS,       "wav2vec2_bert.encoder.layers.%d.self_attn.linear_k.bias"},
    {W2V_TENSOR_ATTN_V_WEIGHT,     "wav2vec2_bert.encoder.layers.%d.self_attn.linear_v.weight"},
    {W2V_TENSOR_ATTN_V_BIAS,       "wav2vec2_bert.encoder.layers.%d.self_attn.linear_v.bias"},
    {W2V_TENSOR_ATTN_LN_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.self_attn_layer_norm.weight"},
    {W2V_TENSOR_ATTN_LN_BIAS,      "wav2vec2_bert.encoder.layers.%d.self_attn_layer_norm.bias"},
    {W2V_TENSOR_ATTN_OUT_WEIGHT,   "wav2vec2_bert.encoder.layers.%d.self_attn.linear_out.weight"},
    {W2V_TENSOR_ATTN_OUT_BIAS,     "wav2vec2_bert.encoder.layers.%d.self_attn.linear_out.bias"},
    {W2V_TENSOR_FFN1_UP_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.ffn1.intermediate_dense.weight"},
    {W2V_TENSOR_FFN1_UP_BIAS,      "wav2vec2_bert.encoder.layers.%d.ffn1.intermediate_dense.bias"},
    {W2V_TENSOR_FFN1_DOWN_WEIGHT,  "wav2vec2_bert.encoder.layers.%d.ffn1.output_dense.weight"},
    {W2V_TENSOR_FFN1_DOWN_BIAS,    "wav2vec2_bert.encoder.layers.%d.ffn1.output_dense.bias"},
    {W2V_TENSOR_FFN1_LN_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.ffn1_layer_norm.weight"},
    {W2V_TENSOR_FFN1_LN_BIAS,      "wav2vec2_bert.encoder.layers.%d.ffn1_layer_norm.bias"},
    {W2V_TENSOR_FFN2_UP_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.ffn2.intermediate_dense.weight"},
    {W2V_TENSOR_FFN2_UP_BIAS,      "wav2vec2_bert.encoder.layers.%d.ffn2.intermediate_dense.bias"},
    {W2V_TENSOR_FFN2_DOWN_WEIGHT,  "wav2vec2_bert.encoder.layers.%d.ffn2.output_dense.weight"},
    {W2V_TENSOR_FFN2_DOWN_BIAS,    "wav2vec2_bert.encoder.layers.%d.ffn2.output_dense.bias"},
    {W2V_TENSOR_FFN2_LN_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.ffn2_layer_norm.weight"},
    {W2V_TENSOR_FFN2_LN_BIAS,      "wav2vec2_bert.encoder.layers.%d.ffn2_layer_norm.bias"},
    {W2V_TENSOR_FINAL_LN_WEIGHT,   "wav2vec2_bert.encoder.layers.%d.final_layer_norm.weight"},
    {W2V_TENSOR_FINAL_LN_BIAS,     "wav2vec2_bert.encoder.layers.%d.final_layer_norm.bias"},
    {W2V_TENSOR_ATTN_DISTANCE_EMBED_WEIGHT,"wav2vec2_bert.encoder.layers.%d.self_attn.distance_embedding.weight"},
    // Conv Module
    {W2V_TENSOR_DW_CONV_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.conv_module.depthwise_conv.weight"},
    {W2V_TENSOR_DW_CONV_LN_BIAS,   "wav2vec2_bert.encoder.layers.%d.conv_module.depthwise_layer_norm.bias"},
    {W2V_TENSOR_DW_CONV_LN_WEIGHT, "wav2vec2_bert.encoder.layers.%d.conv_module.depthwise_layer_norm.weight"},
    {W2V_TENSOR_CONV_LN_BIAS,      "wav2vec2_bert.encoder.layers.%d.conv_module.layer_norm.bias"},
    {W2V_TENSOR_CONV_LN_WEIGHT,    "wav2vec2_bert.encoder.layers.%d.conv_module.layer_norm.weight"},
    {W2V_TENSOR_PW_CONV1_WEIGHT,   "wav2vec2_bert.encoder.layers.%d.conv_module.pointwise_conv1.weight"},
    {W2V_TENSOR_PW_CONV2_WEIGHT,   "wav2vec2_bert.encoder.layers.%d.conv_module.pointwise_conv2.weight"},
    
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
