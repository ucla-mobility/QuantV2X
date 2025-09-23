# Tutorial on Post-Training Quantization (PTQ) Stage on V2X-Real dataset

Please refer to [Tutorial of Baseline Training and Inference on V2X-Real dataset](./Tutorial_V2X-Real_Baseline.md) and [Tutorial of Codebook Learning on V2X-Real dataset](./Tutorial_V2X-Real_Codebook.md) before reading this documentation. Post-Training quantization (PTQ) involves the curation of calibration dataset and the calibration process as described in the paper. 

### Post-Training Quantization (PTQ)

```python
python opencood/tools/inference_mc_quant.py ${CHECKPOINT_FOLDER} [--fusion_method intermediate] --num_cali_batches 16 --n_bits_w 8 --n_bits_a 8 --iters_w 5000
```

- `num_cali_batches` refers to the size of the calibration dataset.
- `n_bits_w` refers to the bitwidth for weight quantization.
- `n_bits_a` refers to the bitwidth for activation quantization.
- `iters_w` refers to the number of calibration steps.


### Notes:
- You could refer to `/scripts/inference_mc/inference_mc_quant.sh` for example running scripts. `mc` stands for `multi-class`, which differentiates itself from `single-class` training and inference. `quant` stands for the PTQ process that is different from `fp` which stands for full-precision inference.
