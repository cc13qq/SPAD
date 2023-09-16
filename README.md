## Code for paper ["Detecting Adversarial Faces Using Only Real Face Self-Perturbations"](https://arxiv.org/abs/2304.11359). 

## Get Started

Datasets are LFW and CelebA-HQ.

Our codebase accesses the datasets from `./data/` and checkpoints from `./results/checkpoints/` by default.
```
├── ...
├── data
│   └── SP_new3
├── results
│   └──checkpoints
├── main.py
├── ...
```

### Train Detector
sh SP_train.sh

### Test
sh SP_test.sh

```
## Dependencies
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
```

### Datasets

LFW and Celeba-HQ datasets we used in this program are [here](https://pan.baidu.com/s/1mWNC4NkJrVkMWWwTxdTb2A?pwd=koof). The generated adv-faces are also provided. You could generate adv-faces by torchattack. The attack code is in attack_utils.

### Checkpoints

We provide some checkpoints for you to test. You can download them [here](https://pan.baidu.com/s/1cDnb8CFzihI3dbvUsheq2g?pwd=jmao). You can put them into the folder './results/checkpoints'.

To Test gradient-based adv-faces on LFW, run:
python main.py --config configs/datasets/SP_LFW.yml configs/pipelines/test/SP_test.yml --network.name X --network.checkpoint 'results/checkpoints/net-best_LFW.ckpt'
To Test gradient-based adv-faces on CelebA-HQ, run:
python main.py --config configs/datasets/SP_celebahq.yml configs/pipelines/test/SP_test.yml --network.name X --network.checkpoint 'results/checkpoints/net-best_celebahq.ckpt'
To Test GAN-based adv-faces on CelebA-HQ, run:
python main.py --config configs/datasets/SP_GC.yml configs/pipelines/test/SP_test.yml --network.name X_sep --network.checkpoint 'results/checkpoints/net-best_GC.ckpt'

### Citation

If you find our repository useful for your research, please consider citing our paper:
```
@inproceedings{ijcai2023p165,
  title     = {Detecting Adversarial Faces Using Only Real Face Self-Perturbations},
  author    = {Wang, Qian and Xian, Yongqin and Ling, Hefei and Zhang, Jinyuan and Lin, Xiaorui and Li, Ping and Chen, Jiazhong and Yu, Ning},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {1488--1496},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/165},
  url       = {https://doi.org/10.24963/ijcai.2023/165},
}
```
