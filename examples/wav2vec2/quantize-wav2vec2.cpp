// quantize-wav2vec2.cpp - Quantize wav2vec2 GGML models
//
// Usage: quantize-wav2vec2 model-f16.bin model-q6_k.bin q6_k

#include "ggml.h"
#include "ggml-backend.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

#define W2V_MAX_CONV_LAYERS 7

// Wav2Vec2 hyperparameters
struct wav2vec2_hparams {
    int32_t n_hidden       = 1024;
    int32_t n_layers       = 24;
    int32_t n_heads        = 16;
    int32_t n_intermediate = 4096;
    int32_t n_vocab        = 392;
    int32_t n_conv_layers  = 7;

    int32_t conv_dim[W2V_MAX_CONV_LAYERS];
    int32_t conv_kernel[W2V_MAX_CONV_LAYERS];
    int32_t conv_stride[W2V_MAX_CONV_LAYERS];

    int32_t num_conv_pos_embeddings = 128;
    int32_t num_conv_pos_embedding_groups = 16;
    int32_t ftype = 1;
};

// Quantize wav2vec2 model
static bool wav2vec2_model_quantize(
        const std::string & fname_inp,
        const std::string & fname_out,
        ggml_ftype ftype) {

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // Verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != 0x77766332) {  // "wv2c"
            fprintf(stderr, "%s: invalid model file '%s' (bad magic: 0x%08x, expected 0x77766332)\n",
                    __func__, fname_inp.c_str(), magic);
            return false;
        }
        fout.write((char *) &magic, sizeof(magic));
    }

    wav2vec2_hparams hparams;

    // Load hyperparameters
    {
        finp.read((char *) &hparams.n_hidden,       sizeof(hparams.n_hidden));
        finp.read((char *) &hparams.n_layers,       sizeof(hparams.n_layers));
        finp.read((char *) &hparams.n_heads,        sizeof(hparams.n_heads));
        finp.read((char *) &hparams.n_intermediate, sizeof(hparams.n_intermediate));
        finp.read((char *) &hparams.n_vocab,        sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_conv_layers,  sizeof(hparams.n_conv_layers));

        for (int i = 0; i < hparams.n_conv_layers; ++i) {
            finp.read((char *) &hparams.conv_dim[i], sizeof(hparams.conv_dim[i]));
        }
        for (int i = 0; i < hparams.n_conv_layers; ++i) {
            finp.read((char *) &hparams.conv_kernel[i], sizeof(hparams.conv_kernel[i]));
        }
        for (int i = 0; i < hparams.n_conv_layers; ++i) {
            finp.read((char *) &hparams.conv_stride[i], sizeof(hparams.conv_stride[i]));
        }

        finp.read((char *) &hparams.num_conv_pos_embeddings, sizeof(hparams.num_conv_pos_embeddings));
        finp.read((char *) &hparams.num_conv_pos_embedding_groups, sizeof(hparams.num_conv_pos_embedding_groups));
        finp.read((char *) &hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        fprintf(stderr, "%s: n_hidden       = %d\n", __func__, hparams.n_hidden);
        fprintf(stderr, "%s: n_layers       = %d\n", __func__, hparams.n_layers);
        fprintf(stderr, "%s: n_heads        = %d\n", __func__, hparams.n_heads);
        fprintf(stderr, "%s: n_intermediate = %d\n", __func__, hparams.n_intermediate);
        fprintf(stderr, "%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_conv_layers  = %d\n", __func__, hparams.n_conv_layers);
        fprintf(stderr, "%s: ftype (src)    = %d\n", __func__, hparams.ftype);
        fprintf(stderr, "%s: qntvr (src)    = %d\n", __func__, qntvr_src);
        fprintf(stderr, "%s: ftype (dst)    = %d\n", __func__, ftype_dst);
        fprintf(stderr, "%s: qntvr (dst)    = %d\n", __func__, GGML_QNT_VERSION);

        // Write hyperparameters to output
        fout.write((const char *) &hparams.n_hidden,       sizeof(hparams.n_hidden));
        fout.write((const char *) &hparams.n_layers,       sizeof(hparams.n_layers));
        fout.write((const char *) &hparams.n_heads,        sizeof(hparams.n_heads));
        fout.write((const char *) &hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fout.write((const char *) &hparams.n_vocab,        sizeof(hparams.n_vocab));
        fout.write((const char *) &hparams.n_conv_layers,  sizeof(hparams.n_conv_layers));

        for (int i = 0; i < hparams.n_conv_layers; ++i) {
            fout.write((const char *) &hparams.conv_dim[i], sizeof(hparams.conv_dim[i]));
        }
        for (int i = 0; i < hparams.n_conv_layers; ++i) {
            fout.write((const char *) &hparams.conv_kernel[i], sizeof(hparams.conv_kernel[i]));
        }
        for (int i = 0; i < hparams.n_conv_layers; ++i) {
            fout.write((const char *) &hparams.conv_stride[i], sizeof(hparams.conv_stride[i]));
        }

        fout.write((const char *) &hparams.num_conv_pos_embeddings, sizeof(hparams.num_conv_pos_embeddings));
        fout.write((const char *) &hparams.num_conv_pos_embedding_groups, sizeof(hparams.num_conv_pos_embedding_groups));
        fout.write((const char *) &ftype_dst, sizeof(ftype_dst));
    }

    // Load and copy vocabulary
    {
        int32_t n_vocab = 0;
        finp.read((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_vocab, sizeof(n_vocab));

        fprintf(stderr, "%s: vocab size = %d\n", __func__, n_vocab);

        for (int i = 0; i < n_vocab; i++) {
            int32_t len;
            finp.read((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            std::vector<char> word(len);
            finp.read(word.data(), len);
            fout.write(word.data(), len);
        }
    }

    // Tensors to NOT quantize (keep in FP16/FP32)
    // - CNN feature extractor layers (small and sensitive)
    // - Layer norms (always keep high precision)
    // - CTC head (output layer, keep for accuracy)
    // - Positional embeddings
    const std::vector<std::string> to_skip = {
        // CNN feature extractor
        ".*feature_extractor.*",
        // Layer norms
        ".*layer_norm.*",
        ".*LayerNorm.*",
        ".*ln.*weight",
        ".*ln.*bias",
        // Positional conv embedding
        ".*pos_conv.*",
        // CTC head (output layer)
        ".*lm_head.*",
        ".*ctc.*",
        // Biases (small tensors)
        ".*bias",
    };

    // Quantize all other tensors
    if (!ggml_common_quantize_0(finp, fout, ftype, { ".*" }, to_skip)) {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s model-f16.bin model-quant.bin type\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Quantization types:\n");
    ggml_print_ftypes(stderr);
    fprintf(stderr, "\n");
    fprintf(stderr, "Recommended types for wav2vec2:\n");
    fprintf(stderr, "  q6_k  - Best accuracy, ~50MB model size\n");
    fprintf(stderr, "  q5_k  - Good balance, ~45MB model size\n");
    fprintf(stderr, "  q4_k  - Smaller, ~40MB model size\n");
    fprintf(stderr, "  q8_0  - Baseline, ~100MB model size\n");
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    // Initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // Quantize
    {
        const int64_t t_start_us = ggml_time_us();

        if (!wav2vec2_model_quantize(fname_inp, fname_out, ggml_ftype(ftype))) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // Report timing and sizes
    {
        const int64_t t_main_end_us = ggml_time_us();

        // Get file sizes
        std::ifstream finp(fname_inp, std::ios::binary | std::ios::ate);
        std::ifstream fout(fname_out, std::ios::binary | std::ios::ate);

        size_t size_inp = finp.tellg();
        size_t size_out = fout.tellg();

        printf("\n");
        printf("%s: input  model size = %8.2f MB\n", __func__, size_inp / (1024.0 * 1024.0));
        printf("%s: output model size = %8.2f MB\n", __func__, size_out / (1024.0 * 1024.0));
        printf("%s: compression ratio = %8.2fx\n", __func__, (float) size_inp / size_out);
        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}
