# SAPD
The code of our IJCAI 2023 paper "Detecting Adversarial Faces Using Only Real Face Self-Perturbations."

## Code for paper "Detecting Adversarial Faces Using Only Real Face Self-Perturbations". 

## Get Started

Datasets are LFW and CelebA-HQ

Our codebase accesses the datasets from `./data/` and pretrained models from `./results/checkpoints/` by default.
```
├── ...
├── data
│   └── SP_new3
├── results
│   └──checkpoints
├── main.py
├── ...
```

### 1. Train Detector:
sh SP_train.sh

### 2. Test:
sh SP_test.sh

```
## Dependencies:
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
