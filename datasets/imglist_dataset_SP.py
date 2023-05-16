import logging
import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
import random
from .base_dataset import BaseDataset
from .RAP import rap_process_opt, get_landmark, get_GC_scaller, add_pixel_GC_continue_opt
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataset_SP(BaseDataset):
    def __init__(self,
                 name,
                 split,
                 interpolation,
                 image_size,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 mode = 'SP',
                 **kwargs):
        super(ImglistDataset_SP, self).__init__(**kwargs)

        self.name = name
        self.image_size = image_size
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size

        self.mode = mode
        if mode in ['SP', 'SP_point', 'SP_block']: # 'RAP' or 'GC'
            self.gc_prob = -1
            self.rap = 0.5
        elif mode == 'GC':
            self.gc_prob = 1
            self.rap = 0.5
        elif mode == None:
            self.rap = -1
        else:
            raise ValueError('mode must be "SP", "GC" or None')
            
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')
        
        self.GC_scaller = get_GC_scaller(256)

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)
    
    def SP_process_split(self, ori_img, mode):
        if mode == 'GC':
            ori_img = cv2.resize(ori_img, (256, 256))
            sp_img, mask = add_pixel_GC_continue_opt(ori_img, eps=random.randint(10, 70), prob=random.randint(16, 40)/1000, sl = [2, 25], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=self.GC_scaller, image_post = True) # for 256x256
        elif mode == 'SP_pixel':
            ori_img = cv2.resize(ori_img, (112, 112))
            sp_img, mask = rap_process_opt(ori_img, eps=None, sl=6, noise_type='point', image_post = True, al=3)
        elif mode == 'SP_block':
            ori_img = cv2.resize(ori_img, (112, 112))
            sp_img, mask = rap_process_opt(ori_img, eps=None, sl=6, noise_type='block', image_post = True, al=3)
        elif mode == 'SP':
            ori_img = cv2.resize(ori_img, (112, 112))
            sp_img, mask = rap_process_opt(ori_img, eps=None, sl=6, noise_type=None, image_post = True, al=3)

        sp_img = sp_img.astype(np.uint8)

        return sp_img, mask
    
    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        # print(line)
        tokens = line.split('\t', 1)
        # print(tokens)
        image_name, extra_str = tokens[0], tokens[1]
        # print('image_name',image_name)
        # print('extra_str',extra_str)
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        if self.data_dir == None:
            path = image_name
        else:
            path = os.path.join(self.data_dir, image_name)

        sample = dict()
        sample['image_name'] = image_name

        # TODO: comments
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        self.preprocessor.setup(**kwargs)
        try:
            img = cv2.imread(path)
            # img = cv2.resize(img, (256, 256))
            if np.random.random() <= self.rap:
                # img, _ = self.SP_process(img, mode=self.mode)
                img, _ = self.SP_process_split(img, mode=self.mode)
                sample['label'] = 0
            else:
                sample['label'] = 1
            image = Image.fromarray(img)
            sample['data'] = self.transform_image(image)
            sample['data_aux'] = self.transform_aux_image(image)

            # sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
