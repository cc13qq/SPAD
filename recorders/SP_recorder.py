import os
import time
from pathlib import Path

import torch
import csv


class SPRecorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_acc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics):
        # train_metrics: ['loss', 'epoch_idx']

        if train_metrics is not None:
            print('\nEpoch {:03d} | Batch {:03d} | Time {:5d}s | Train Loss {:.4f} | '.format(
                    train_metrics['epoch_idx'], train_metrics['batch_idx'], int(time.time() - self.begin_time), train_metrics['loss']), flush=True)
    
    def save_model(self, nets, train_metrics):

        net_name='net'

        if self.config.recorder.save_all_models:
            # for net_name in nets:
            save_fname = 'model_epoch{}_batch{}_acc{:.4f}.ckpt'.format(train_metrics['epoch_idx'], train_metrics['batch_idx'])
            save_pth = os.path.join(self.output_dir, net_name+'-'+save_fname)
            print('\nsaving ' + net_name+'-'+save_fname)
            torch.save(nets[net_name].state_dict(), save_pth)

        # save last path
        if train_metrics['epoch_idx'] == self.config.optimizer.num_epochs:
            save_fname = 'last_epoch{}_acc{:.4f}.ckpt'.format(train_metrics['epoch_idx'])

            # for net_name in nets:
            save_pth = os.path.join(self.output_dir, net_name+'-'+save_fname)
            print('\nsaving ' + net_name+'-'+save_fname)
            torch.save(nets[net_name].state_dict(), save_pth)

    def summary(self):
        print('Training Completed! '
              'Best accuracy: {:.4f} '
              'at epoch {:d}'.format(self.best_acc, self.best_epoch_idx),
              flush=True)
