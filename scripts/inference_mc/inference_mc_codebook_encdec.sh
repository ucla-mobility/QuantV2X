#!/bin/bash
# Author: Seth Z. Zhao <sethzhao506@g.ucla.edu>

# Inference script for codebook models using encode/decode approach
# This script simulates the actual compression/decompression pipeline

CUDA_VISIBLE_DEVICES=4 python ./opencood/tools/inference_mc_codebook_encdec.py \
    --fusion_method intermediate \
    --model_dir ./opencood/logs/your_model_directory \
