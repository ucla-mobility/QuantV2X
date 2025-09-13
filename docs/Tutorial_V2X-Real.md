# Basic Train / Test Command on Running on V2X-Real dataset

Since V2X-Real utilizes multi-class predictions, the exact commands would be slightly different from running on OPV2V and DAIR-V2X. These training and testing instructions apply to all end-to-end training methods. Note that we adopt HEAL as the codebase structure and currently we only feature collaboration base training.

### Train the model

We uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:

```python
python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```

Arguments Explanation:

- `-y` or `hypes_yaml` : the path of the training configuration file, e.g. `opencood/hypes_yaml/opv2v/LiDAROnly/lidar_fcooper.yaml`, meaning you want to train
  a FCooper model. **We elaborate each entry of the yaml in the exemplar config file `opencood/hypes_yaml/exemplar.yaml`.**
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
  given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,

### Train the model in DDP

```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```

`--nproc_per_node` indicate the GPU number you will use.

### Test the model

```python
python opencood/tools/inference_mc.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
```

- `inference_mc.py` has more optional args, you can inspect into this file.
- `[--fusion_method intermediate]` the default fusion method is intermediate fusion. According to your fusion strategy in training, available fusion_method can be:
  - **single**: only ego agent's detection, only ego's gt box. _[only for late fusion dataset]_
  - **no**: only ego agent's detection, all agents' fused gt box. _[only for late fusion dataset]_
  - **late**: late fusion detection from all agents, all agents' fused gt box. _[only for late fusion dataset]_
  - **early**: early fusion detection from all agents, all agents' fused gt box. _[only for early fusion dataset]_
  - **intermediate**: intermediate fusion detection from all agents, all agents' fused gt box. _[only for intermediate fusion dataset]_

### Notes:
- You could refer to `/scripts` folder to for example running scripts. `mc` stands for `multi-class`, which differentiates itself from `single-class` training and inference.
