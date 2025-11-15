# Tutorial on Training and Testing on other datasets

The main model architecture difference with **V2X-Real** and other datasets such as **OPV2V** and **DAIR-V2X** is the usage of `multi-class classification` in **V2X-Real** dataset compared with the `single-class classification` (only vehicle detection) used in other datasets. Please be aware that we arrange separate python files for running on different datasets. Please refer to [Tutorial of Baseline Training and Inference on V2X-Real dataset](./Tutorial_V2X-Real_Baseline.md), [Tutorial of Codebook Learning on V2X-Real dataset](./Tutorial_V2X-Real_Codebook.md), and [Tutorial of PTQ on V2X-Real dataset](./Tutorial_V2X-Real_PTQ.md) before reading this documentation. 

### Train the model

We uses yaml file to configure all the parameters for training. We have already configured the **dataset** and **core method** in the yaml file. To train your own model
from scratch or a continued checkpoint, run the following commonds:

```python
python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```

### Codebook Learning
#### Stage 1: Train the full-precision model

We uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds (with **opv2v** as examples):

```python
python ./opencood/tools/train.py -y ./opencood/hypes_yaml/opv2v/Codebook/Pyramid/lidar_pyramid_stage1.yaml
```

#### Stage 2: Codebook-only Training
```python
python ./opencood/tools/train_stage2.py --hypes_yaml ./opencood/hypes_yaml/opv2v/Codebook/Pyramid/lidar_pyramid_stage2.yaml --stage1_model your_path_to_stage1_model.pth
```

- `stage1_model` points to the pretrained checkpoint from **Stage 1**.

#### Stage 3: Codebook Co-training
```python
python ./opencood/tools/train_stage3.py --hypes_yaml ./opencood/hypes_yaml/opv2v/Codebook/Pyramid/lidar_pyramid_stage3.yaml --stage2_model your_path_to_stage2_model.pth
```

- `stage2_model` points to the pretrained checkpoint from **Stage 2**.

### Test the model

Note that we use `inference.py` instead of `inference_mc.py` for evaluating on other datasets:

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
```

### Post-Training Quantization (PTQ)

Note that we use `inference_quant.py` instead of `inference_mc_quant.py` for evaluating on other datasets:

```python
python opencood/tools/inference_quant.py ${CHECKPOINT_FOLDER} [--fusion_method intermediate] --num_cali_batches 16 --n_bits_w 8 --n_bits_a 8 --iters_w 5000
```

- `num_cali_batches` refers to the size of the calibration dataset.
- `n_bits_w` refers to the bitwidth for weight quantization.
- `n_bits_a` refers to the bitwidth for activation quantization.
- `iters_w` refers to the number of calibration steps.


### Notes:
- You could refer to `/scripts` folder for more examples.
- When modifying yaml files in `./opencood/hypes_yaml`, pay attention to the `assignment_path`, `core_method`, `dataset` fields to ensure the consistency of the functions and corresponding datasets.
