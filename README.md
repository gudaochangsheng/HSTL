<div align="center">

# HSTL

### Hierarchical Spatio-Temporal Representation Learning for Gait Recognition

<p>
  <b>ICCV 2023</b> · Hierarchical Gait Representation Learning · Official PyTorch Implementation
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Hierarchical_Spatio-Temporal_Representation_Learning_for_Gait_Recognition_ICCV_2023_paper.pdf">
    <img src="https://img.shields.io/badge/ICCV-2023-2b6cb0?style=flat-square" alt="ICCV 2023">
  </a>
  <a href="https://arxiv.org/abs/2307.09856">
    <img src="https://img.shields.io/badge/arXiv-2307.09856-b31b1b?style=flat-square" alt="arXiv">
  </a>
  <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Hierarchical_Spatio-Temporal_Representation_Learning_for_Gait_Recognition_ICCV_2023_paper.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square" alt="Paper PDF">
  </a>
  <a href="https://github.com/gudaochangsheng/HSTL">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=flat-square&logo=github" alt="Code">
  </a>
  <a href="https://www.researchgate.net/publication/376260699_Hierarchical_Spatio-Temporal_Representation_Learning_for_Gait_Recognition">
    <img src="https://img.shields.io/badge/ResearchGate-Publication-00CCBB?style=flat-square&logo=researchgate&logoColor=white" alt="ResearchGate">
  </a>
  <a href="https://www.youtube.com/watch?v=gOt0JjfGxBM">
    <img src="https://img.shields.io/badge/YouTube-Video%20Presentation-FF0000?style=flat-square&logo=youtube&logoColor=white" alt="YouTube Video">
  </a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=gudaochangsheng/HSTL" alt="Visitors">
</p>

<p align="center">
  <a href="https://xinxi.hebau.edu.cn/info/1061/2291.htm">
    <img src="https://img.shields.io/badge/News-College%20Report-07C160?style=flat-square" alt="College News">
  </a>
  <a href="https://www.hebau.edu.cn/info/1272/21545.htm">
    <img src="https://img.shields.io/badge/News-University%20Report-1677ff?style=flat-square" alt="University News">
  </a>
</p>

</div>

---

## 🔥 News

- **Jul. 2023**: Our paper **"Hierarchical Spatio-Temporal Representation Learning for Gait Recognition"** was accepted by **ICCV 2023**.
- This work marks the **first paper from Hebei Agricultural University accepted by ICCV** since the university was founded more than 120 years ago, achieving a historic **0-to-1 breakthrough** for the university in top-tier computer vision conferences.
- This paper was completed with **Hebei Agricultural University as the independent affiliation**, without external institutional collaboration.
- Related reports: [College News](https://xinxi.hebau.edu.cn/info/1061/2291.htm) · [University News](https://www.hebau.edu.cn/info/1272/21545.htm)

## 📝 Introduction

This repository provides the official implementation of our ICCV 2023 paper:

> **Hierarchical Spatio-Temporal Representation Learning for Gait Recognition**  
> Lei Wang, Bo Liu, Fangfang Liang, Bincheng Wang  
> IEEE/CVF International Conference on Computer Vision (**ICCV**), 2023

HSTL learns hierarchical spatio-temporal representations for gait recognition. Instead of modeling gait sequences with only flat or isolated part-level representations, HSTL captures motion patterns from coarse to fine levels and improves the representation ability for challenging gait recognition scenarios.

## 🧭 Project Navigation

| Section | Description |
|---|---|
| [Operating Environments](#-operating-environments) | Hardware and software dependencies |
| [Checkpoints](#-checkpoints) | Pre-trained checkpoints for CASIA-B and OUMVLP |
| [Train and Test](#-train-and-test) | Commands for training and evaluation |
| [Acknowledgement](#-acknowledgement) | Codebase acknowledgement |
| [Citation](#-citation) | BibTeX citation |

## 🖥️ Operating Environments

### Hardware Environment

Our code runs on a server with:

```text
8 × NVIDIA GeForce RTX 3090 GPUs
Intel(R) Core(TM) i7-9800X CPU @ 3.80GHz
```

### Software Environment

```text
PyTorch = 1.10
torchvision
pyyaml
tensorboard
opencv-python
tqdm
```

## 📦 Checkpoints

| Dataset | Checkpoint |
|---|---|
| CASIA-B | [Google Drive](https://drive.google.com/file/d/1keZBtWr9O8gfeqBB9qHNbZ-96Eh6LggB/view?usp=sharing) |
| OUMVLP | [Google Drive](https://drive.google.com/file/d/1VNYC0QbHxw1aaBTFLj4DMIC2D36B1-ng/view?usp=sharing) |

## 🚀 Train and Test

### Train

Train a model with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  opengait/main.py \
  --cfgs ./configs/htsl/hstl.yaml \
  --phase train
```

Arguments:

| Argument | Description |
|---|---|
| `python -m torch.distributed.launch` | DDP launch instruction |
| `--nproc_per_node` | Number of GPUs to use; it must equal the length of `CUDA_VISIBLE_DEVICES` |
| `--cfgs` | Path to the config file |
| `--phase train` | Run the training phase |
| `--log_to_file` | Save terminal logs to disk if specified |

You can also run commands in:

```bash
bash train.sh
```

### Test

Evaluate a trained model with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  opengait/main.py \
  --cfgs ./configs/htsl/hstl.yaml \
  --phase test
```

Arguments:

| Argument | Description |
|---|---|
| `--phase test` | Run the testing phase |
| `--iter` | Specify the iteration checkpoint for evaluation |

Other arguments are the same as the training phase.

You can also run commands in:

```bash
bash test.sh
```

## 🙏 Acknowledgement

This codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait). We sincerely thank the authors for their excellent work.

## 📚 Citation

If you find this project useful, please consider citing our paper:

```bibtex
@InProceedings{Wang_2023_ICCV,
    author    = {Wang, Lei and Liu, Bo and Liang, Fangfang and Wang, Bincheng},
    title     = {Hierarchical Spatio-Temporal Representation Learning for Gait Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {19639-19649}
}
```


## ⭐ Star History

<a href="https://www.star-history.com/?repos=gudaochangsheng%2FHSTL&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=gudaochangsheng/HSTL&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=gudaochangsheng/HSTL&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=gudaochangsheng/HSTL&type=date&legend=top-left" />
 </picture>
</a>
