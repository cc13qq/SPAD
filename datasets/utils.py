import torch
from numpy import load
from torch.utils.data import DataLoader

from preprocessors.test_preprocessor import TestStandardPreProcessor
from preprocessors.utils import get_preprocessor
from utils.config import Config

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .imglist_dataset_SP import ImglistDataset_SP

def get_dataloader_SP(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        # all script file need to pass in train_preprocessor config file
        preprocessor = get_preprocessor(config, split) # split
        # for data_aux data augmentation
        data_aux_preprocessor = TestStandardPreProcessor(config, 'test')
        CustomDataset = eval(split_config.dataset_class) # split
        sampler = None

        if split == 'train':
        
            dataset = CustomDataset(name=dataset_config.name + '_' + split,
                                    split=split,
                                    interpolation=split_config.interpolation,
                                    image_size=dataset_config.image_size,
                                    imglist_pth=split_config.imglist_pth,
                                    data_dir=split_config.data_dir,
                                    num_classes=dataset_config.num_classes,
                                    preprocessor=preprocessor,
                                    data_aux_preprocessor=data_aux_preprocessor,
                                    mode=dataset_config.mode)

            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)

            dataloader_dict[split] = dataloader
        
        else:
            dataloader_dict[split] = {}
            for set in split_config.set_names:
                set_config = split_config[set]
                dataloader_dict[split][set] = {}

                if set == 'Real':

                    dataset = CustomDataset(name=dataset_config.name + '_' + split + '_' + set, # dataset
                                            split=set, # set
                                            interpolation=split_config.interpolation, # split
                                            image_size=dataset_config.image_size, # dataset
                                            imglist_pth=set_config.imglist_pth, # set
                                            data_dir=set_config.data_dir, # set
                                            num_classes=dataset_config.num_classes, # dataset
                                            preprocessor=preprocessor,
                                            data_aux_preprocessor=data_aux_preprocessor)

                    if dataset_config.num_gpus * dataset_config.num_machines > 1:
                        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        set_config.shuffle = False

                    dataloader = DataLoader(dataset,
                                            batch_size=split_config.batch_size, # split
                                            shuffle=split_config.shuffle, # split
                                            num_workers=dataset_config.num_workers, # dataset
                                            sampler=sampler)

                    dataloader_dict[split][set] = dataloader
                    continue


                for subset in set_config.subsets:
                    subset_config = set_config[subset]

                    dataset = CustomDataset(name=dataset_config.name + '_' + split + '_' + set + '_' + subset, # dataset
                                            split=subset, # subset
                                            interpolation=split_config.interpolation, # split
                                            image_size=dataset_config.image_size, # dataset
                                            imglist_pth=subset_config.imglist_pth, # subset
                                            data_dir=subset_config.data_dir, # subset
                                            num_classes=dataset_config.num_classes, # dataset
                                            preprocessor=preprocessor,
                                            data_aux_preprocessor=data_aux_preprocessor)

                    if dataset_config.num_gpus * dataset_config.num_machines > 1:
                        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        set_config.shuffle = False

                    dataloader = DataLoader(dataset,
                                            batch_size=split_config.batch_size, # split
                                            shuffle=split_config.shuffle, # split
                                            num_workers=dataset_config.num_workers, # dataset
                                            sampler=sampler)

                    dataloader_dict[split][set][subset] = dataloader

    return dataloader_dict

