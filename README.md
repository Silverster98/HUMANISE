# HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes

Created by 
[Zan Wang](https://silvester.wang),
[Yixin Chen](https://yixchen.github.io/),
[Tengyu Liu](http://tengyu.ai/),
[Yixin Zhu](https://yzhu.io/),
[Wei Liang](https://liangwei-bit.github.io/web/),
[Siyuan Huang](https://siyuanhuang.com/)

This repository is an official implementation of "HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes".

In this work, we propose a large-scale and semantic-rich human-scene interaction dataset, HUMANISE. It has language description for each human-scene interaction. HUMANISE enables a new task: language-conditioned human motion generation in 3D scenes. 

[Paper](https://silvester.wang/HUMANISE/paper.pdf) | [arXiv](https://arxiv.org/abs/2210.09729) | [Project Page](https://github.com/Silverster98/HUMANISE) | [Data](https://docs.google.com/forms/d/e/1FAIpQLSfzhj2wrRLqAXFVOTn8K5NDN-J_5HueRTohMAlayqBuPPWA1w/viewform?usp=sf_link)

<!-- <img src='./figure/teaser_md.png' weight="50%"> -->
<div align=center>
<img src='./figure/teaser_md.png' width=60%>
</div>

## Abstract

Learning to generate diverse scene-aware and goal-oriented human motions in 3D scenes remains challenging due to the mediocre characteristics of the existing datasets on Human-Scene Interaction(HSI); they only have limited scale/quality and lack semantics. To fill in the gap, we propose a large-scale and semantic-rich synthetic HSI dataset, denoted as HUMANISE, by aligning the captured human motion sequences with various 3D indoor scenes. We automatically annotate the aligned motions with language descriptions that depict the action and the unique interacting objects in the scene; e.g., sit on the armchair near the desk. HUMANISE thus enables a new generation task, language-conditioned human motion generation in 3D scenes. The proposed task is challenging as it requires joint modeling of the 3D scene, human motion, and natural language. To tackle this task, we present a novel scene-and-language conditioned generative model that can produce 3D human motions of the desirable action interacting with the specified objects. Our experiments demonstrate that our model generates diverse and semantically consistent human motions in 3D scenes.

## Preparation

### 1. Installation

### 2. Data Preparation

1. Download [ScanNet](http://www.scan-net.org/) Dataset.

Notes, change the dataset folder configuration in `utils/configuration.py`. 

## HUMANISE Dataset

### 1. Synthesis

```bash

```


### 2. Visualization

For HUMANISE dataset visualization, we provide rendering script as following, which will render an animation video with top-down view. The result will be saved in `./tmp/`.

- on-screen rendering

```bash
python visualize_dataset.py --pkl your_path/lie/scene0000_001810_c71dc702-1f1d-4381-895c-f07e9a10876b/anno.pkl --index 0 --vis
```

Notes: `--vis` will render the static human-scene interaction with [trimesh](https://trimsh.org/trimesh.html) on screen.

- off-screen rendering

```bash
PYOPENGL_PLATFORM=egl python visualize_dataset.py --pkl your_path/lie/scene0000_001810_c71dc702-1f1d-4381-895c-f07e9a10876b/anno.pkl --index 0
```

## Our Model

### Train

### Test

### Results

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

