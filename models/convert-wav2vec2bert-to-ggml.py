#!/usr/bin/env python3
# Convert Wav2Vec2 phoneme model from HuggingFace to GGML format
#
# Usage: python convert-wav2vec2-to-ggml.py [model_name_or_path] [output_dir] [use-f32]
#
# This script converts the wav2vec2-xlsr-53-espeak-cv-ft or similar phoneme models
# to GGML format for use with whisper.cpp's wav2vec2 implementation.
#
# The output is a single binary file containing:
#   - Magic number (0x77766332 = "wv2c")
#   - Hyperparameters
#   - Phoneme vocabulary (IPA symbols)
#   - CNN feature extractor weights
#   - Transformer encoder weights
#   - CTC head weights
#
# For each tensor, we write:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])

import os
import sys
import struct
import json
import argparse
import numpy as np

def convert_wav2vec2_to_ggml(model_name_or_path, output_dir, use_f16=True):
    """Convert HuggingFace Wav2Vec2 model to GGML format."""
    try:
        import torch
        from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
    except ImportError:
        print("Error: Please install transformers and torch:")
        print("  pip install transformers torch")
        sys.exit(1)

    print(f"Loading model: {model_name_or_path}")
    model = Wav2Vec2BertForCTC.from_pretrained(model_name_or_path)
    model.eval()

    # Get config
    config = model.config
    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  conv_depthwise_kernel_size: {config.conv_depthwise_kernel_size}")
    print(f"  conv_depthwise_kernel_size: {config.conv_depthwise_kernel_size}")

    # Load vocabulary
    vocab_list = None
    try:
        processor = Wav2Vec2BertProcessor.from_pretrained(model_name_or_path)
        vocab = processor.tokenizer.get_vocab()
        # Sort by id
        vocab_list = sorted(vocab.items(), key=lambda x: x[1])
        print(f"\nVocabulary size: {len(vocab_list)}")
        print(f"Sample tokens: {vocab_list[:10]}")
    except Exception as e:
        print(f"Warning: Could not load vocabulary via processor: {e}")
        # Try to load vocab.json directly from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            vocab_file = hf_hub_download(repo_id=model_name_or_path, filename="vocab.json")
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            vocab_list = sorted(vocab.items(), key=lambda x: x[1])
            print(f"\nLoaded vocabulary from vocab.json")
            print(f"Vocabulary size: {len(vocab_list)}")
            print(f"Sample tokens: {vocab_list[:10]}")
        except Exception as e2:
            print(f"Warning: Could not load vocab.json: {e2}")
            print("Using default IPA vocabulary from config")
            vocab_list = [(f"<unk_{i}>", i) for i in range(config.vocab_size)]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output filename
    if use_f16:
        fname_out = os.path.join(output_dir, "ggml-model-f16.bin")
    else:
        fname_out = os.path.join(output_dir, "ggml-model-f32.bin")

    print(f"\nWriting to: {fname_out}")

    # Get state dict
    state_dict = model.state_dict()

    with open(fname_out, "wb") as fout:
        # Write magic number: "wv2b" = 0x77766232
        fout.write(struct.pack("I", 0x77766232))

        # Write hyperparameters
        fout.write(struct.pack("i", config.hidden_size))        # n_hidden
        fout.write(struct.pack("i", config.num_hidden_layers))  # n_layers
        fout.write(struct.pack("i", config.num_attention_heads)) # n_heads
        fout.write(struct.pack("i", config.intermediate_size))  # n_intermediate
        fout.write(struct.pack("i", config.vocab_size))         # n_vocab
        fout.write(struct.pack("i", config.conv_depthwise_kernel_size))# conv_depthwise_kernel_size

        # Additional config
        # fout.write(struct.pack("i", config.num_conv_pos_embeddings))
        # fout.write(struct.pack("i", config.num_conv_pos_embedding_groups))
        # fout.write(struct.pack("i", 1 if use_f16 else 0))  # ftype

        # Write vocabulary
        fout.write(struct.pack("i", len(vocab_list)))
        for token, idx in vocab_list:
            token_bytes = token.encode('utf-8')
            fout.write(struct.pack("i", len(token_bytes)))
            fout.write(token_bytes)

        # Write tensors
        n_tensors = 0

        # Pre-process: combine parametrized pos_conv weight
        # Weight normalization stores g (gain) and v (direction): weight = g * (v / ||v||)
        # pos_conv_g_name = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        # pos_conv_v_name = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        # combined_tensors = {}

        # if pos_conv_g_name in state_dict and pos_conv_v_name in state_dict:
        #     g = state_dict[pos_conv_g_name].numpy()  # [128]
        #     v = state_dict[pos_conv_v_name].numpy()  # [1024, 64, 128]

        #     # Compute weight = g * v / ||v||
        #     # Normalize along dim 0 and 1 (the spatial and input channel dims)
        #     v_norm = np.linalg.norm(v, axis=(0, 1), keepdims=True)
        #     weight = g * (v / (v_norm + 1e-12))

        #     combined_tensors["wav2vec2.encoder.pos_conv_embed.conv.weight"] = weight
        #     print(f"Combined pos_conv weight: {weight.shape}")

        for name, tensor in state_dict.items():
            # Skip the original parametrized tensors (we've combined them)
            if "parametrizations.weight.original" in name:
                print(f"Skipping parametrized tensor: {name}")
                continue

            # Don't squeeze - preserve all dimensions (important for conv weights with IC=1)
            data = tensor.numpy()
            #print(f"Processing: {name} with shape {data.shape}")

            n_dims = len(data.shape)

            # Determine ftype for this tensor
            # ftype == 0 -> float32, ftype == 1 -> float16
            ftype = 1
            if use_f16:
                # Keep small tensors, biases, layer norms, and embeddings in FP32
                # for numerical stability
                if n_dims < 2 or \
                   "bias" in name or \
                   "layer_norm" in name or \
                   "LayerNorm" in name or \
                   "pos_conv" in name or \
                   data.size < 1024:
                    #print(f"  -> Converting to float32 (ftype=0)")
                    data = data.astype(np.float32)
                    ftype = 0
                else:
                    data = data.astype(np.float16)
            else:
                data = data.astype(np.float32)
                ftype = 0

            # Write tensor header
            name_bytes = name.encode('utf-8')
            fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))

            # Write dimensions (in reverse order for GGML)
            for i in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))

            # Write name
            fout.write(name_bytes)

            # Write data
            data.tofile(fout)
            n_tensors += 1

        # # Write combined tensors (pos_conv weight)
        # for name, data in combined_tensors.items():
        #     print(f"Writing combined tensor: {name} with shape {data.shape}")

        #     n_dims = len(data.shape)

        #     # Keep pos_conv in FP32 for stability
        #     data = data.astype(np.float32)
        #     ftype = 0

        #     name_bytes = name.encode('utf-8')
        #     fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))

        #     for i in range(n_dims):
        #         fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))

        #     fout.write(name_bytes)
        #     data.tofile(fout)
        #     n_tensors += 1

        print(f"\nDone! Wrote {n_tensors} tensors")

    # Get file size
    file_size = os.path.getsize(fname_out)
    print(f"Output file: {fname_out}")
    print(f"File size: {file_size / 1e6:.1f} MB")

    # Also save vocabulary as JSON for reference
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(dict(vocab_list), f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to: {vocab_path}")

    return fname_out


def main():
    parser = argparse.ArgumentParser(
        description="Convert Wav2Vec2 phoneme model to GGML format"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="HuggingFace model name or local path (default: facebook/wav2vec2-lv-60-espeak-cv-ft)"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="models/wav2vec2-phoneme",
        help="Output directory (default: models/wav2vec2-phoneme)"
    )
    parser.add_argument(
        "--f32",
        action="store_true",
        help="Use float32 instead of float16 (larger file, slightly more accurate)"
    )

    args = parser.parse_args()

    use_f16 = not args.f32
    convert_wav2vec2_to_ggml(args.model, args.output_dir, use_f16)


if __name__ == "__main__":
    main()
