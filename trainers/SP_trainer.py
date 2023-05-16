import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import Config
import utils.comm as comm

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class SPTrainer: # only on real
    def __init__(self, net, train_loader, config: Config):
        self.train_loader = train_loader
        self.config = config
        self.net = net
        self.weight_energy = torch.nn.Linear(config.num_classes, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)
        self.logistic_regression = torch.nn.Linear(1, 2).cuda()
        
        self.optimizer = torch.optim.Adam(list(net.parameters()) + list(self.weight_energy.parameters()) + list(self.logistic_regression.parameters()),
            lr=config.optimizer['learning_rate'], 
            betas=(0.9, 0.999),
            weight_decay=config.optimizer['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(step, config.optimizer['num_epochs'] * len(train_loader), 1, 1e-6 / config.optimizer['learning_rate']))

        self.number_dict = {} 
        for i in range(self.config['num_classes']):
            self.number_dict[i] = 0
        self.data_dict = torch.zeros(self.config['num_classes'], self.config['sample_number'], self.config['feature_dim']).cuda() 
         
        self.cls_real_list = self.config['class_real'] # record real class only
        self.nets = dict()
        self.nets['net'] = self.net
        self.nets['weight_energy'] = self.weight_energy
        self.nets['logistic_regression'] = self.logistic_regression


    def train_epoch(self, epoch_idx, batch_idx, evaluator, postprocessor, recorder):
        # self.net.train()
        for net_name in self.nets:
            self.nets[net_name].train()
        loss_avg = 0.0
        sample_number = self.config['sample_number']
        num_classes = self.config['num_classes']
        num_classes_real = len(self.cls_real_list)
        train_dataiter = iter(self.train_loader)
        eye_matrix = torch.eye(self.config['feature_dim'], device='cuda')

        for train_step in tqdm(range(1, len(train_dataiter) + 1),
                               desc='Epoch {:03d}'.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            images = batch['data'].cuda()
            labels = batch['label'].cuda()

            logit, feature = self.net.forward(images, return_feature=True)

            sum_temp = 0 
            for index in self.cls_real_list:
                sum_temp += self.number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()[0] 

            if sum_temp == num_classes_real * sample_number and epoch_idx < self.config['start_epoch']:
                gt_numpy = labels.cpu().data.numpy()
                for index in range(len(labels)):
                    cls_idx = gt_numpy[index]

                    if cls_idx in self.cls_real_list: # record real class only
                        self.data_dict[cls_idx] = torch.cat((self.data_dict[cls_idx][1:], feature[index].detach().view(1, -1)), 0) 

            elif sum_temp == num_classes_real * sample_number and epoch_idx >= self.config['start_epoch']:
                gt_numpy = labels.cpu().data.numpy()
                for index in range(len(labels)):
                    cls_idx = gt_numpy[index]

                    if cls_idx in self.cls_real_list: # record real class only
                        self.data_dict[cls_idx] = torch.cat((self.data_dict[cls_idx][1:], feature[index].detach().view(1, -1)), 0) 

                for i, index in enumerate(self.cls_real_list):
                    if i == 0:
                        X = self.data_dict[index] - self.data_dict[index].mean(0)
                        mean_embed_id = self.data_dict[index].mean(0).view(1, -1) 
                    else:
                        X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id, self.data_dict[index].mean(0).view(1, -1)), 0)
                temp_precision = torch.mm(X.t(), X) / len(X) 
                temp_precision += 0.0001 * eye_matrix

                for i, index in enumerate(self.cls_real_list): 
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean_embed_id[i], covariance_matrix=temp_precision) 
                    negative_samples = new_dis.rsample((self.config['sample_from'], ))
                    prob_density = new_dis.log_prob(negative_samples) 
                    cur_samples, index_prob = torch.topk(-prob_density, self.config['select']) 
                    if i == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)

                if len(ood_samples) != 0:

                    energy_score_for_fg = self.log_sum_exp(logit, dim=1) 

                    predictions_ood = self.net.fc(ood_samples)

                    energy_score_for_bg = self.log_sum_exp(predictions_ood, dim=1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(feature)).cuda(), torch.zeros(len(ood_samples)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()
                    output1 = self.logistic_regression(input_for_lr.view(-1, 1))

                    lr_reg_loss = criterion(output1, labels_for_lr.long())

            else: # add data
                gt_numpy = labels.cpu().data.numpy()
                for index in range(len(labels)):
                    cls_idx = gt_numpy[index]

                    if cls_idx in self.cls_real_list and self.number_dict[cls_idx] < sample_number:  # record real class only
                    
                        self.data_dict[cls_idx][self.number_dict[cls_idx]] = feature[index].detach()
                        self.number_dict[cls_idx] += 1

            self.optimizer.zero_grad()
            loss = F.cross_entropy(logit, labels)
            loss += self.config.trainer['loss_weight'] * lr_reg_loss 
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

            batch_idx += 1

        metrics = {}
        metrics['loss'] = loss_avg
        metrics['epoch_idx'] = epoch_idx
        metrics['batch_idx'] = batch_idx

        return self.nets, metrics, batch_idx


    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation."""
        value.exp().sum(dim, keepdim).log() # energy

        # TODO: torch.max(value, dim=None) threw an error at time of writing
        # weight_energy = torch.nn.Linear(num_classes, 1).cuda()
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)

            output = m + torch.log(
                torch.sum(F.relu(self.weight_energy.weight) * torch.exp(value0),
                        dim=dim,
                        keepdim=keepdim))
            # set lower bound
            out_list = output.cpu().detach().numpy().tolist()
            for i in range(len(out_list)):
                if out_list[i] < -1:
                    out_list[i] = -1
                else:
                    continue
            output = torch.Tensor(out_list).cuda()
            return output
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)
