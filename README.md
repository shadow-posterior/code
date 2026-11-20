<div align="center">

<h1>Reinforcing Few-Step Diffusion Models via<br>
Shadow Posterior Matching</h1>

[![arXiv 2611.25998](https://img.shields.io/badge/arXiv-2611.25998-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.25998)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

Shadow Posterior Matching (SPM) fine-tunes four-step image generators with black-box rewards. It re-noises generated samples to assign reward credit at every denoising step and uses the undistilled teacher to regularize the generator. This repository provides LoRA training recipes for Stable Diffusion 3.5 Medium.

## Setup

Use Python 3.10 or 3.11 with an NVIDIA GPU. Install the pinned dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

For GenEval, install MMDetection with MMCV CUDA operators built against the installed PyTorch and CUDA toolkit.

```bash
python -m pip install setuptools wheel ninja
MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-build-isolation \
  'mmcv-full @ git+https://github.com/open-mmlab/mmcv.git@4e85793e58fb220ed91be8c075396b16a385f349'
python -m pip install 'mmdet @ git+https://github.com/open-mmlab/mmdetection.git@e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1'
```

For the DrawBench evaluation in the recipes, install the additional HPSv2 and ImageReward scorer packages using the dependencies installed above.

```bash
python -m pip install 'clip @ git+https://github.com/openai/CLIP.git@ded190a052fdf4585bd685cee5bc96e0310d2c93'
python -m pip install --no-deps hpsv2x==1.2.0 image-reward==1.5
```

Set `model.pretrained_path` to a local Diffusers copy of SD3.5 Medium. Supply the generator and fake-score adapters from the same four-step SD3.5 Medium DMD2 run in `model.generator_lora.load_path` and `model.fake_score_lora.load_path`.

## Training

| Task | Recipe | Training prompts | Reward assets |
| --- | --- | --- | --- |
| Compositional generation | [GenEval](configs/sd35m_geneval.yaml) | `dataset/geneval/train_metadata.jsonl` | Mask2Former checkpoint in `weights/geneval/` and CLIP weights at `weights/geneval/open_clip/ViT-L-14.pt` |
| Visual text rendering | [OCR](configs/sd35m_ocr.yaml) | `dataset/ocr/train.txt` | PP-OCRv4 model files in `weights/paddleocr/{det,rec,cls}/` |

GenEval JSONL rows include a `prompt` and its object metadata. OCR prompts put nonempty target text in straight double quotes.

Launch either recipe with `torchrun`. This example uses eight GPUs for GenEval.

```bash
torchrun --standalone --nproc-per-node=8 main.py configs/sd35m_geneval.yaml \
  --override eval.enabled=false logging.enabled=false
```

To enable evaluation, supply the task's held-out prompts under `dataset/geneval/` or `dataset/ocr/`, DrawBench prompts at `dataset/drawbench/test.txt`, and the benchmark reward weights under `weights/`. Remove the `eval.enabled=false` override to evaluate during training.

## Acknowledgments

We thank the authors for sharing their work: [Flow-GRPO](https://github.com/yifan123/flow_grpo), [DMD](https://github.com/tianweiy/DMD2), and [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT).

## Citation

```bibtex
@article{xiu2026spm,
  title={{Reinforcing Few-Step Diffusion Models via Shadow Posterior Matching}},
  author={Jingqiao Xiu and Yicong Li},
  journal={arXiv preprint arXiv:2611.25998},
  year={2026},
}
```
