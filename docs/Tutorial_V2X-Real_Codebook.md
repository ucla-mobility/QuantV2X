# Tutorials on Codebook Learning on V2X-Real dataset

Please refer to [Tutorial of Baseline Training and Inference on V2X-Real dataset](./Tutorial_V2X-Real_Baseline.md) and before reading this documentation. Codebook learning involves three stages: (i) full-precision pretraining, where we train a full-precision model that serves as the foundation for codebook learning; (ii) codebook-only training, where we randomly initialize codebook and only update the codebook features based on pretrained models and freeze other model parameters; (iii) codebook co-training, where we tune the whole pipeline to achieve best perception performance.

### Stage 1: Train the full-precision model

We uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:

```python
python ./opencood/tools/train.py -y ./opencood/hypes_yaml/v2x_real/Codebook/stage1/lidar_pyramid_stage1.yaml
```

### Stage 2: Codebook-only Training
```python
python ./opencood/tools/train_stage2.py --hypes_yaml ./opencood/hypes_yaml/v2x_real/Codebook/stage2/lidar_pyramid_stage2.yaml --stage1_model your_path_to_stage1_model.pth
```

- `stage1_model` points to the pretrained checkpoint from **Stage 1**.

### Stage 3: Codebook Co-training
```python
python ./opencood/tools/train_stage3.py --hypes_yaml ./opencood/hypes_yaml/v2x_real/Codebook/stage3/lidar_pyramid_stage3.yaml --stage2_model your_path_to_stage2_model.pth
```

- `stage2_model` points to the pretrained checkpoint from **Stage 2**.

### Test the model

```python
python opencood/tools/inference_mc.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
```


### Notes:
- You could refer to `/scripts/train_codebook_mc` folder to for example running scripts. `mc` stands for `multi-class`, which differentiates itself from `single-class` training and inference.
