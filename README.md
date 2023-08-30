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
```

### 3. Datasets:

LFW and CelebaHQ dataset we used in this program are [here](https://pan.baidu.com/s/1mWNC4NkJrVkMWWwTxdTb2A?pwd=koof). The generated adv-faces are also provided. You could generate advfaces by torchattack. The attack code is in attack_utils.

### 4. Checkpoints:

We provide somde checkpoints for you to test. You can download them [here](https://pan.baidu.com/s/1cDnb8CFzihI3dbvUsheq2g?pwd=jmao)
