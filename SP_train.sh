#!/bin/bash
# sh scripts/ood/vos/cifar10_train_vos.sh

PYTHONPATH='.':$PYTHONPATH \
python main.py \
--config configs/datasets/SP_LFW.yml \
configs/pipelines/train/SP_train.yml \
--num_workers 8 \
--optimizer.num_epochs 5

