#!/bin/bash
# Author: Seth Z. Zhao <sethzhao506@g.ucla.edu>

# Cached inference script that separates encoding and decoding to avoid OOM
# This version saves encoded codes to disk between stages

MODEL_DIR="./opencood/logs/your_model_directory"
CACHE_DIR="${MODEL_DIR}/encoded_cache"

# Check if cache directory exists and has files
if [ -d "${CACHE_DIR}" ] && [ "$(ls -A ${CACHE_DIR}/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "==== CACHE FOUND ===="
    echo "Found existing cache in: ${CACHE_DIR}"
    echo "Number of cached files: $(ls ${CACHE_DIR}/*.pkl | wc -l)"
    echo "Skipping encoding stage, proceeding directly to decoding..."
    echo ""
else
    echo "==== STAGE 1: ENCODING ===="
    echo "No cache found, starting encoding phase..."
    CUDA_VISIBLE_DEVICES=4 python ./opencood/tools/inference_mc_codebook_encdec_cached.py \
        --fusion_method intermediate \
        --model_dir ${MODEL_DIR} \
        --cache_dir ${CACHE_DIR} \
        --encode_only \
        --save_vis_interval 40

    echo ""
    echo "Encoding complete! Cached files: $(ls ${CACHE_DIR}/*.pkl 2>/dev/null | wc -l)"
    echo ""
fi

echo "==== STAGE 2: DECODING ===="
CUDA_VISIBLE_DEVICES=5 python ./opencood/tools/inference_mc_codebook_encdec_cached.py \
    --fusion_method intermediate \
    --model_dir ${MODEL_DIR} \
    --cache_dir ${CACHE_DIR} \
    --decode_only \
    --save_vis_interval 40

echo ""
echo "==== COMPLETE ===="
echo "Results saved to: ${MODEL_DIR}"
