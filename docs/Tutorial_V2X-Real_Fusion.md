# Tutorials on Early/Late Fusion Learning on V2X-Real dataset

Please refer to [Tutorial of Baseline Training and Inference on V2X-Real dataset](./Tutorial_V2X-Real_Baseline.md) and before reading this documentation. Early fusion involves fusing only raw LiDAR point cloud data from neighboring agents to create a more holistic view of the enviornment, leading to better predictions. Late fusion invovles receiving independent detections from neighboring agents to produce consistent and more accurate predictions. 

### Stage 1: Train the full-precision model

We uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:

```python
python ./opencood/tools/train.py -y ./opencood/hypes_yaml/v2x_real/LiDAROnly/lidar_[early/late]_mc_fusion.yaml
```

### Test the model

```python
python opencood/tools/inference_mc.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method early/late]
```

### Notes:
- You could also run single class early/late fusion with yaml files in the `./opencood/hypes_yaml/dairv2x/LiDAROnly` folder under `lidar_early_fusion.yaml` and `lidar_late_fusion.yaml` respectively. You will need to test the model with `inference.py` as opposed to `inference_mc.py` however.
