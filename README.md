# *HUMANISE*: Language-conditioned Human Motion Generation in 3D Scenes

<p align="left">
    <a href='https://arxiv.org/abs/2210.09729'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://silvester.wang/HUMANISE/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://silverster98.github.io/HUMANISE/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
     <a href='https://docs.google.com/forms/d/e/1FAIpQLSfzhj2wrRLqAXFVOTn8K5NDN-J_5HueRTohMAlayqBuPPWA1w/viewform?usp=sf_link'>
      <img src='https://img.shields.io/badge/Dataset-Data-yellow?style=plastic&logo=Databricks&logoColor=yellow' alt='Dataset'>
    </a>
</p>

[Zan Wang](https://silvester.wang),
[Yixin Chen](https://yixchen.github.io/),
[Tengyu Liu](http://tengyu.ai/),
[Yixin Zhu](https://yzhu.io/),
[Wei Liang](https://liangwei-bit.github.io/web/),
[Siyuan Huang](https://siyuanhuang.com/)

This repository is an official implementation of "HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes".

In this work, we propose a large-scale and semantic-rich human-scene interaction dataset, HUMANISE. It has language description for each human-scene interaction. HUMANISE enables a new task: language-conditioned human motion generation in 3D scenes. 

[Paper](https://silvester.wang/HUMANISE/paper.pdf) | [arXiv](https://arxiv.org/abs/2210.09729) | [Project Page](https://silverster98.github.io/HUMANISE) | [Data](https://docs.google.com/forms/d/e/1FAIpQLSfzhj2wrRLqAXFVOTn8K5NDN-J_5HueRTohMAlayqBuPPWA1w/viewform?usp=sf_link)

<!-- <img src='./figure/teaser_md.png' weight="50%"> -->
<div align=center>
<img src='./figure/teaser_md.png' width=60%>
</div>

## Abstract

Learning to generate diverse scene-aware and goal-oriented human motions in 3D scenes remains challenging due to the mediocre characteristics of the existing datasets on Human-Scene Interaction(HSI); they only have limited scale/quality and lack semantics. To fill in the gap, we propose a large-scale and semantic-rich synthetic HSI dataset, denoted as HUMANISE, by aligning the captured human motion sequences with various 3D indoor scenes. We automatically annotate the aligned motions with language descriptions that depict the action and the unique interacting objects in the scene; e.g., sit on the armchair near the desk. HUMANISE thus enables a new generation task, language-conditioned human motion generation in 3D scenes. The proposed task is challenging as it requires joint modeling of the 3D scene, human motion, and natural language. To tackle this task, we present a novel scene-and-language conditioned generative model that can produce 3D human motions of the desirable action interacting with the specified objects. Our experiments demonstrate that our model generates diverse and semantically consistent human motions in 3D scenes.

## Preparation

### 1. Environment Setup

### 2. Data Preparation

1. [ScanNet V2](http://www.scan-net.org/) Dataset

Remember to change the dataset folder configuration in `utils/configuration.py`. 

2. Our pre-synthesized [data](https://docs.google.com/forms/d/e/1FAIpQLSfzhj2wrRLqAXFVOTn8K5NDN-J_5HueRTohMAlayqBuPPWA1w/viewform?usp=sf_link), or you can generate your own data with our pipeline, see [HUMANISE Synthesis](./dataset/README.md) for more details.

3. [SMPLX v1.1](https://smpl-x.is.tue.mpg.de/download.php)

## HUMANISE Dataset

### 1. Synthesis

See [HUMANISE Synthesis](./dataset/README.md) for more details.

### 2. Visualization

For HUMANISE dataset visualization, we provide rendering script `visualize_dataset.py` which will render an animation video with top-down view. The result will be saved in `./tmp/`.

- on-screen rendering

```bash
python visualize_dataset.py --pkl ${PKL} --index ${index} --vis
# python visualize_dataset.py --pkl your_path/lie/scene0000_001810_c71dc702-1f1d-4381-895c-f07e9a10876b/anno.pkl --index 0 --vis
```

Notes: `--vis` will render the static human-scene interaction with [trimesh](https://trimsh.org/trimesh.html) on screen.

- off-screen rendering

```bash
PYOPENGL_PLATFORM=egl python visualize_dataset.py --pkl ${PKL} --index ${index}
# PYOPENGL_PLATFORM=egl python visualize_dataset.py --pkl your_path/lie/scene0000_001810_c71dc702-1f1d-4381-895c-f07e9a10876b/anno.pkl --index 0
```

See more information about the [data format](./dataset/README.md#file-format).

## Our Model

### Action-Specific Model

- Train

  ```bash
  bash scripts/train.sh "${ACTION}"
  # bash scripts/train.sh "walk"
  ```

- Eval (Quantitative)

  ```bash
  bash scripts/eval_metric.sh ${STAMP} "${ACTION}"
  # bash scripts/eval_metric.sh 20220829_194320 "walk"
  ```

- Eval (Qualitative)

  ```bash
  bash scripts/eval.sh ${STAMP} "${ACTION}"
  # bash scripts/eval.sh 20220829_194320 "walk"
  ```

### Action-Agnostic Model

## Citation

If you find our project useful, please consider citing us:

```bibtex
@inproceedings{wang2022humanise,
  title={HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes},
  author={Wang, Zan and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Huang, Siyuan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## Acknowledgements

