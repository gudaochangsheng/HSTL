# HSTL

This is the code for the paper [Hierarchical Spatio-Temporal Representation Learning for Gait Recognition](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Hierarchical_Spatio-Temporal_Representation_Learning_for_Gait_Recognition_ICCV_2023_paper.pdf).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-spatio-temporal-representation/gait-recognition-in-the-wild-on-gait3d)](https://paperswithcode.com/sota/gait-recognition-in-the-wild-on-gait3d?p=hierarchical-spatio-temporal-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-spatio-temporal-representation/gait-recognition-on-gait3d)](https://paperswithcode.com/sota/gait-recognition-on-gait3d?p=hierarchical-spatio-temporal-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-spatio-temporal-representation/gait-recognition-on-oumvlp)](https://paperswithcode.com/sota/gait-recognition-on-oumvlp?p=hierarchical-spatio-temporal-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-spatio-temporal-representation/multiview-gait-recognition-on-casia-b)](https://paperswithcode.com/sota/multiview-gait-recognition-on-casia-b?p=hierarchical-spatio-temporal-representation)

# Operating Environments
## Hardware Environment
Our code is running on a server with 8 GeForce RTX 3090 GPUs 
and a CPU model Intel(R) Core(TM) i7-9800X @ 3.80GHz.
## Software Environment
- pytorch = 1.10
- torchvision
- pyyaml
- tensorboard
- opencv-python
- tqdm

# Checkpoints
* The checkpoints for CASIA-B [link](https://drive.google.com/file/d/1keZBtWr9O8gfeqBB9qHNbZ-96Eh6LggB/view?usp=sharing)
* The checkpoints for OUMVLP [link](https://drive.google.com/file/d/1VNYC0QbHxw1aaBTFLj4DMIC2D36B1-ng/view?usp=sharing)

# Train and test
## Train
Train a model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/htsl/hstl.yaml --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
<!-- - `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 

You can run commands in [train.sh](train.sh) for training different models.

## Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/htsl/hstl.yaml --phase test
```
- `--phase` Specified as `test`.
- `--iter` Specify a iteration checkpoint.

**Tip**: Other arguments are the same as train phase.

You can run commands in [test.sh](test.sh) for testing different models.

# Acknowledgement
* The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait).

# Citation
```
@InProceedings{Wang_2023_ICCV,
    author    = {Wang, Lei and Liu, Bo and Liang, Fangfang and Wang, Bincheng},
    title     = {Hierarchical Spatio-Temporal Representation Learning for Gait Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {19639-19649}
}
```
