# FramePainter

Official pytorch implementation of "FramePainter: Endowing Interactive Image Editing with Video Diffusion Priors"

[![arXiv](https://img.shields.io/badge/arXiv-2501.08225-b31b1b.svg)](https://arxiv.org/abs/2501.08225)<a href="https://huggingface.co/Yabo/FramePainter"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>![visitors](https://visitor-badge.laobi.icu/badge?page_id=YBYBZhang/FramePainter)

# Demo

https://github.com/user-attachments/assets/8e04dfce-2750-4196-8a73-b6bab833fdb1

## News

* [01/25/2025] Inference demo and [pre-trained weights](https://huggingface.co/Yabo/FramePainter) are available now!
* [01/15/2025] Paper of [FramePainter](https://arxiv.org/abs/2501.08225) released!

## Gallery

<p align="center">
<img src="intro_teaser.png" width="1080px"/> 
</p>

FramePainter allows users to manipulate images through intuitive sketches.
Benefiting from powerful video diffusion priors, it not only enables intuitive and plausible edits in common scenarios, but also exhibits exceptional generalization in out-of-domain cases, e.g., transform the fish into shark-like shape.

## Setup

### 1. Download Weights

Download [pre-trained weights](https://huggingface.co/Yabo/FramePainter) of FramePainter into `checkpoints/` directory, including finetuned U-Net and sparse control encoder. The `app.py` will automatically download `stabilityai/stable-video-diffusion-img2vid-xt-1-1` during inference.

### 2. Requirements

```shell
conda create -n framepainter python=3.10
conda activate framepainter
pip install -r requirements.txt
```

### 3. Inference

Directly run `python app.py`.

## Acknowledgement

This repository borrows code from [Diffusers](https://github.com/huggingface/diffusers) and [ControlNext](https://github.com/dvlab-research/ControlNeXt). Thanks for their contributions!