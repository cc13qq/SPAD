#!/bin/bash
# sh scripts/ood/vos/cifar10_test_vos.sh

PYTHONPATH='.':$PYTHONPATH \
python main.py \
--config configs/datasets/SP_LFW.yml \
configs/pipelines/test/SPtest.yml \
--num_workers 8 \
--network.checkpoint 'results/xception_net_sep_SP_e50_lr0.1_SP/net-model_epoch4.ckpt' 
