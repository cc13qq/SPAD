import torch
import numpy as np
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os
from align_methods import align, re_align
from get_model import getmodel
from torch.nn import DataParallel
from PIL import Image
from torch import Tensor
import torchvision
import torchattacks_wq as torchattacks
from torch import nn
from torchvision import transforms
import random
from utils_wq import *
import time
from tqdm import tqdm
import argparse

th_dict = {'ir152':(0.094632, 0.166788, 0.227922), 'irse50':(0.144840, 0.241045, 0.312703),
           'facenet':(0.256587, 0.409131, 0.591191), 'mobile_face':(0.183635, 0.301611, 0.380878),
           'CosFace':(0.1, 0.260819, 0.378918), 'ArcFace':(0.1, 0.272494, 0.319879)}

def cal_asr(src_model, ori_fetures, adv_fetures):
    # 计算攻击成功率
    print('Calculating ASR')
    sims = []
    for ori_fea, adv_fea in zip(ori_fetures, adv_fetures):
        sim = cosin_metric(ori_fea, adv_fea)
        sims.append(sim)

    th01, th001, th0001 = th_dict[src_model]
    total = len(sims)
    success01 = 0
    success001 = 0
    success0001 = 0
    for v in sims:
        if v < th01:
            success01 += 1
        if v < th001:
            success001 += 1
        if v < th0001:
            success0001 += 1
    print(src_model, " attack success(far@0.1) rate: ", success01 / total)
    print(src_model, " attack success(far@0.01) rate: ", success001 / total)
    print(src_model, " attack success(far@0.001) rate: ", success0001 / total)    

def get_atk_method(atk_method, model, params):
    '''
    CW APGD APGDT, OnePixel, DeepFool 需要label暂时不实现

    TIP-IM暂时不能用

    FGSM, BIM, RFGSM, PGD, MIFGSM(MIM), DIFGSM(DIM), TIFGSM(TIM)
    eps = 8-18
    '''
    if 'eps' in params: eps = params['eps']
    if 'alpha' in params: alpha = params['alpha']
    if 'steps' in params: steps = params['steps']
    if 'std' in params: std = params['std']
    if 'gamma' in params: gamma = params['gamma']

    if atk_method == 'FGSM':
        atk = torchattacks.FGSM_face(model, eps=eps)
    elif atk_method == 'BIM':
        atk = torchattacks.BIM_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'RFGSM':
        atk = torchattacks.RFGSM_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'PGD':
        atk = torchattacks.PGD_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'PGDL2': # 没啥效果
        atk = torchattacks.PGDL2_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'EOTPGD':
        atk = torchattacks.EOTPGD_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'FFGSM': # 没啥效果
        atk = torchattacks.FFGSM_face(model, eps=eps, alpha=alpha)
    elif atk_method == 'TPGD':
        atk = torchattacks.TPGD_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'MIFGSM':
        atk = torchattacks.MIFGSM_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'GN':
        atk = torchattacks.GN_face(model, std=std)
    elif atk_method == 'DIFGSM':
        atk = torchattacks.DIFGSM_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'TIFGSM':
        atk = torchattacks.TIFGSM_face(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'TIPIM': # targeted only
        atk = torchattacks.TIPIM_face(model, eps = eps, gamma = gamma, steps = steps, norm = 'l2', gain = 'gain3')
    else:
        raise Exception
    return atk

def run_lfw_tmp(model_name, atk_method, dataset_name, num_total, num, batch_size, base_data_root, people_txt, model, img_shape, mtcnn, params, save_img, adv_dataset_root, targets_dir, save_noise):

    print('Attack dataset ', dataset_name)
    atk = get_atk_method(atk_method, model, params)
    print('Attack algorithm ', str(atk))

    # root = 'E:/Files/code/Dataset_Face/adv_face'
    sub_root = make_dir_name(atk_method, atk)
    dataset_root = os.path.join(adv_dataset_root, dataset_name, atk_method, model_name, sub_root) # 创建数据保存路径
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root) # 创建数据保存路径
    print('save root: ', dataset_root)

    # 抽取图像
    # if num == num_total:
    people_name, images_gen = sample_images_per_class_tmp(people_txt) # sample 0001 image per class
    # else:
    #     people_name, images_gen, images_ref = sample_images(people_adv_txt, num_total, num) # len=batch_size list
    print('Nums of images_gen ', len(images_gen))

    images_ref, _ = get_img_list(targets_dir)

    images_ref = images_ref[:len(images_gen)]
    print('Nums of images_ref ', len(images_ref))

    total = len(people_name)

    if int(total%batch_size) == 0:
        epoch = int(total/batch_size)
    else:
        epoch = int(total/batch_size)+1
    
    # if targets_dir != '':
    #     # 读取目标图片地址, 每个identity有3个references
    #     tar_images, _ = load_images(targets_dir)
    #     # 选取一个target构造label
    #     tar_images_single = tar_images[0]
    #     tar_images_single = cv2.resize(tar_images_single, (img_shape[1], img_shape[0])) 

    for i in tqdm(range(epoch)):
        # print('turn ', i)
        l = i * batch_size
        r = min((i + 1) * batch_size, total)
        images_gen_i= images_gen[l : r]
        people_name_i = people_name[l : r]
        images_ref_i = images_ref[l : r]

        # print(images_ref_i)

        # 读取原图片
        # origin_images, _ = load_list_images(images_gen_i, people_name_i, base_data_root,) # origin_images (b,H,W,C) numpy.array
        origin_images, _ = load_list_images_resize(images_gen_i, people_name_i, base_data_root, img_shape) # origin_images (b,H,W,C) numpy.array

        #获取图片名
        origin_names = images_gen_i
        align_ori_images = origin_images

        # 构造输入
        input_images = make_inputs(align_ori_images)

        # if targets_dir == '':
        # 读取参考图片地址
        # ref_images, _ = load_list_images_resize(images_ref_i, people_name_i, base_data_root, img_shape) # ref_images (b,H,W,C) numpy.array
        # align_ref_images = ref_images
        #构造labels——每个identity第一张参考图的feature
        # align_lab_images = make_inputs(align_ref_images)
        # align_lab_images = make_inputs(align_ori_images)
        # else:

        # tar_images = []
        # for i in range(input_images.shape[0]):
        #     tar_images.append(tar_images_single)
        # tar_images = np.array(tar_images)
        tar_images = load_images_list(images_ref_i, img_shape) # origin_images (b,H,W,C) numpy.array
        align_tar_images = tar_images
        #构造labels
        align_lab_images = make_inputs(align_tar_images)

        labels = model(align_lab_images.cuda())

        # attack
        adv_images = atk(input_images, labels, targeted=True) # (b,C,H,W) torch.tensor
        adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() # (b,H,W,C) numpy.array

        if save_img:
            # save image
            for i in range(len(adv_images)):
                _ = save_adv_image(adv_images[i], sub_root+'-'+origin_names[i], dataset_root)
        
    print('Attack compete')

def run_lfw(model_name, atk_method, dataset_name, num_total, num, batch_size, base_data_root, people_adv_txt, model, img_shape, mtcnn, params, save_img, adv_dataset_root, targets_dir, save_noise):

    print('Attack dataset ', dataset_name)
    atk = get_atk_method(atk_method, model, params)
    print('Attack algorithm ', str(atk))

    # root = 'E:/Files/code/Dataset_Face/adv_face'
    sub_root = make_dir_name(atk_method, atk)
    dataset_root = os.path.join(adv_dataset_root, dataset_name, atk_method, model_name, sub_root) # 创建数据保存路径
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root) # 创建数据保存路径
    print('save root: ', dataset_root)

    # 抽取图像
    if num == num_total:
        people_name, images_gen, images_ref = sample_images_per_class(people_adv_txt) # sample all images
    else:
        people_name, images_gen, images_ref = sample_images(people_adv_txt, num_total, num) # len=batch_size list
    print('Nums of images_gen ', len(images_gen))

    total = len(people_name)

    if int(total%batch_size) == 0:
        epoch = int(total/batch_size)
    else:
        epoch = int(total/batch_size)+1
    
    if targets_dir != '':
        # 读取目标图片地址, 每个identity有3个references
        tar_images, _ = load_images(targets_dir)
        # 选取一个target构造label
        tar_images_single = tar_images[0]
        tar_images_single = cv2.resize(tar_images_single, (img_shape[1], img_shape[0])) 

    for i in tqdm(range(epoch)):
        # print('turn ', i)
        l = i * batch_size
        r = min((i + 1) * batch_size, total)
        images_gen_i= images_gen[l : r]
        people_name_i = people_name[l : r]
        images_ref_i = images_ref[l : r]

        # 读取原图片
        # origin_images, _ = load_list_images(images_gen_i, people_name_i, base_data_root,) # origin_images (b,H,W,C) numpy.array
        origin_images, _ = load_list_images_resize(images_gen_i, people_name_i, base_data_root, img_shape) # origin_images (b,H,W,C) numpy.array

        #获取图片名
        origin_names = images_gen_i
        align_ori_images = origin_images

        # 构造输入
        input_images = make_inputs(align_ori_images)

        if targets_dir == '':
        # 读取参考图片地址
            ref_images, _ = load_list_images_resize(images_ref_i, people_name_i, base_data_root, img_shape) # ref_images (b,H,W,C) numpy.array
            align_ref_images = ref_images
            #构造labels——每个identity第一张参考图的feature
            align_lab_images = make_inputs(align_ref_images)
        else:
            tar_images = []
            for i in range(input_images.shape[0]):
                tar_images.append(tar_images_single)
            tar_images = np.array(tar_images)
            align_tar_images = tar_images
            #构造labels
            align_lab_images = make_inputs(align_tar_images)
        labels = model(align_lab_images.cuda())

        # attack
        adv_images = atk(input_images, labels) # (b,C,H,W) torch.tensor
        adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() # (b,H,W,C) numpy.array

        if save_img:
            # save image
            for i in range(len(adv_images)):
                _ = save_adv_image(adv_images[i], sub_root+'-'+origin_names[i], dataset_root)
        
    print('Attack compete')

def load_images_list(image_list, img_shape):
    # input image_list and image file path and load image as numpy.array
    # output numpy.array images and path list
    # imgs_path = []
    # for i in range(len(image_list)):
    #     imgs_path.append(os.path.join(file_path, image_list[i]))

    # 转成0-255numpy.tensor
    imgs = []
    for img_path in image_list:
        img = cv2.imread(img_path)
        cropped_img = cv2.resize(img, (img_shape[1], img_shape[0]))
        imgs.append(cropped_img)
    images = np.array(imgs) # (b,H,W,C)numpy.array
    return images

def get_img_list(data_root):
        # get image list
        file_list = os.listdir(data_root)
        img_list = []
        img_names = []
        for file in file_list:
            file_dir = os.path.join(data_root, file)
            if os.path.isfile(file_dir):
                if file_dir.endswith('jpg') or file_dir.endswith('png'):
                    img_list.append(file_dir)
                    img_names.append(file)
            elif os.path.isdir(file_dir):
                for sub_file in os.listdir(file_dir):
                    sub_file_dir = os.path.join(file_dir, sub_file)
                    if sub_file_dir.endswith('jpg') or sub_file_dir.endswith('png'):
                        img_list.append(sub_file_dir)
                        img_names.append(sub_file)
        
        return img_list, img_names

def run_normal_tmp(model_name, atk_method, dataset_name, batch_size, base_data_root, txt_file, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise):

    print('Attack dataset ', dataset_name)
    atk = get_atk_method(atk_method, model, params)
    print('Attack algorithm ', str(atk))

    # root = 'E:/Files/code/Dataset_Face/adv_face'
    sub_root = make_dir_name(atk_method, atk)
    dataset_root = os.path.join(adv_dataset_root, dataset_name, atk_method, model_name, sub_root) # 创建数据保存路径
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root) # 创建数据保存路径
    
    print('dir ', dataset_root)
    
    if save_noise:
        noise_root = os.path.join(adv_dataset_root, dataset_name, 'noise', atk_method, model_name, sub_root) # 创建数据保存路径
        if not os.path.exists(noise_root):
            os.makedirs(noise_root) # 创建数据保存路径

    # 抽取图像
    # img_list = os.listdir(base_data_root) # eg 000291.jpg
    # img_list, img_names = get_img_list(base_data_root) # image dir list

    img_list = []
    img_names = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')[0]
            img_name = line.split('/')[-1].split('.')[0]
            img_names.append(img_name+'.png')
            img_list.append(os.path.join(base_data_root, img_name+'.png'))
        f.close()

    total = len(img_list)
    print('Nums of images', total)

    images_ref, _ = get_img_list(targets_dir)
    if len(images_ref) >= len(img_list):
        images_ref = images_ref[:len(img_list)]
    else:
        images_ref_tmp = []
        for i in range(int(len(img_list)/len(images_ref))+1):
            images_ref_tmp += images_ref
        images_ref = images_ref_tmp[:len(img_list)]
    print('Nums of images_ref ', len(images_ref))

    # print(img_list[-1])
    # print(img_names[-1])

    if int(total%batch_size) == 0:
        epoch = int(total/batch_size)
    else:
        epoch = int(total/batch_size)+1

    for i in tqdm(range(epoch)):
        # print('turn ', i)
        l = i * batch_size
        r = min((i + 1) * batch_size, total)
        # 获取图片名
        images_i= img_list[l : r]
        images_names = img_names[l : r]
        images_ref_i = images_ref[l : r]

        # 读取原图片
        origin_images = []
        for img_dir in images_i:
            img = cv2.imread(img_dir)
            origin_images.append(img)
        origin_images = np.array(origin_images) # origin_images (b,H,W,C) numpy.array

        # 构造输入
        align_ori_images = origin_images
        input_images = make_inputs(align_ori_images)

        tar_images = load_images_list(images_ref_i, img_shape) # origin_images (b,H,W,C) numpy.array
        align_tar_images = tar_images
        #构造labels
        align_lab_images = make_inputs(align_tar_images)

        labels = model(align_lab_images.cuda())

        # attack
        adv_images = atk(input_images, labels, targeted=True) # (b,C,H,W) torch.tensor
        adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() # (b,H,W,C) numpy.array

        if save_img:
            # save image
            for i in range(len(adv_images)):
                _ = save_adv_image(adv_images[i], sub_root+'-'+images_names[i], dataset_root)

        if save_noise:
            noise = extract_noise(adv_images, origin_images)
            for i in range(len(noise)):
                _ = save_adv_image(noise[i], sub_root+'-'+images_names[i], noise_root)
            
    print('Attack compete')

def run_normal_tmp1(model_name, atk_method, dataset_name, batch_size, base_data_root, txt_file, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise):

    print('Attack dataset ', dataset_name)
    atk = get_atk_method(atk_method, model, params)
    print('Attack algorithm ', str(atk))

    # root = 'E:/Files/code/Dataset_Face/adv_face'
    sub_root = make_dir_name(atk_method, atk)
    dataset_root = os.path.join(adv_dataset_root, dataset_name, atk_method, model_name, sub_root) # 创建数据保存路径
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root) # 创建数据保存路径
    
    print('dir ', dataset_root)
    
    if save_noise:
        noise_root = os.path.join(adv_dataset_root, dataset_name, 'noise', atk_method, model_name, sub_root) # 创建数据保存路径
        if not os.path.exists(noise_root):
            os.makedirs(noise_root) # 创建数据保存路径

    # 抽取图像
    # img_list = os.listdir(base_data_root) # eg 000291.jpg
    # img_list, img_names = get_img_list(base_data_root) # image dir list

    # img_list = []
    # img_names = []
    # print('open file ', txt_file)
    # with open(txt_file, 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip('\n').split('\t')[0]
    #         img_name = line.split('-')[-1].split('.')[0]
    #         img_names.append(img_name+'.png')
    #         img_list.append(os.path.join(base_data_root, img_name+'.png'))
    #     f.close()

    exist_name_list = [fn.split('-')[-1] for fn in os.listdir(dataset_root)]
    exist_list = [os.path.join(base_data_root ,name) for name in exist_name_list]
    
    img_list = []
    img_names = []
    print('open file ', txt_file)
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')[0]
            img_name = line.split('-')[-1].split('.')[0]
            fn_dir = os.path.join(base_data_root, img_name+'.png')
            if fn_dir not in exist_list:
                img_names.append(img_name+'.png')
                img_list.append(fn_dir)
        f.close()
    
    assert len(img_names) == len(img_list)

    total = len(img_list)
    print('Nums of images', total)

    images_ref, _ = get_img_list(targets_dir)
    if len(images_ref) >= total:
        images_ref = images_ref[:total]
    else:
        images_ref_tmp = []
        for i in range(int(total/len(images_ref))+1):
            images_ref_tmp += images_ref
        images_ref = images_ref_tmp[:total]
    print('Nums of images_ref ', len(images_ref))

    # print(img_list[-1])
    # print(img_names[-1])

    if int(total%batch_size) == 0:
        epoch = int(total/batch_size)
    else:
        epoch = int(total/batch_size)+1

    for i in tqdm(range(epoch)):
        # print('turn ', i)
        l = i * batch_size
        r = min((i + 1) * batch_size, total)
        # 获取图片名
        images_i= img_list[l : r]
        images_names = img_names[l : r]
        images_ref_i = images_ref[l : r]

        # 读取原图片
        origin_images = []
        for img_dir in images_i:
            img = cv2.imread(img_dir)
            origin_images.append(img)
        origin_images = np.array(origin_images) # origin_images (b,H,W,C) numpy.array

        # 构造输入
        align_ori_images = origin_images
        input_images = make_inputs(align_ori_images)

        tar_images = load_images_list(images_ref_i, img_shape) # origin_images (b,H,W,C) numpy.array
        align_tar_images = tar_images
        #构造labels
        align_lab_images = make_inputs(align_tar_images)

        labels = model(align_lab_images.cuda())

        # attack
        adv_images = atk(input_images, labels, targeted=True) # (b,C,H,W) torch.tensor
        adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() # (b,H,W,C) numpy.array

        if save_img:
            # save image
            for i in range(len(adv_images)):
                _ = save_adv_image(adv_images[i], sub_root+'-'+images_names[i], dataset_root)

        if save_noise:
            noise = extract_noise(adv_images, origin_images)
            for i in range(len(noise)):
                _ = save_adv_image(noise[i], sub_root+'-'+images_names[i], noise_root)
            
    print('Attack compete')


def run_normal(model_name, atk_method, dataset_name, batch_size, base_data_root, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise):

    print('Attack dataset ', dataset_name)
    atk = get_atk_method(atk_method, model, params)
    print('Attack algorithm ', str(atk))

    # root = 'E:/Files/code/Dataset_Face/adv_face'
    sub_root = make_dir_name(atk_method, atk)
    dataset_root = os.path.join(adv_dataset_root, dataset_name, atk_method, model_name, sub_root) # 创建数据保存路径
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root) # 创建数据保存路径
    
    if save_noise:
        noise_root = os.path.join(adv_dataset_root, dataset_name, 'noise', atk_method, model_name, sub_root) # 创建数据保存路径
        if not os.path.exists(noise_root):
            os.makedirs(noise_root) # 创建数据保存路径

    # 抽取图像
    # img_list = os.listdir(base_data_root) # eg 000291.jpg
    img_list, img_names = get_img_list(base_data_root) # image dir list

    start_idx = 50000

    total = len(img_list)
    remain_total = min(total-start_idx, 50000)

    # remain_total = total - start_idx

    if int(remain_total%batch_size) == 0:
        epoch = int(remain_total/batch_size)
    else:
        epoch = int(remain_total/batch_size)+1

    print('Nums of images', remain_total)


    # total = len(img_list)
    # total = min(total, 50000)

    # # img_list = img_list[:total]
    # # img_names = img_names[:total]
    # print('Nums of images', total)

    images_ref, _ = get_img_list(targets_dir)
    if len(images_ref) >= len(img_list):
        images_ref = images_ref[:len(img_list)]
    else:
        images_ref_tmp = []
        for i in range(int(len(img_list)/len(images_ref))+1):
            images_ref_tmp += images_ref
        images_ref = images_ref_tmp[:len(img_list)]
    print('Nums of images_ref ', len(images_ref))

    # print(img_list[-1])
    # print(img_names[-1])

    # if int(total%batch_size) == 0:
    #     epoch = int(total/batch_size)
    # else:
    #     epoch = int(total/batch_size)+1

    for i in tqdm(range(epoch)):
        # print('turn ', i)
        l = i * batch_size + start_idx
        r = min((i + 1) * batch_size + start_idx, total)
        # 获取图片名
        images_i= img_list[l : r]
        images_names = img_names[l : r]
        images_ref_i = images_ref[l : r]

        # 读取原图片
        origin_images = []
        for img_dir in images_i:
            img = cv2.imread(img_dir)
            origin_images.append(img)
        origin_images = np.array(origin_images) # origin_images (b,H,W,C) numpy.array

        # 构造输入
        align_ori_images = origin_images
        input_images = make_inputs(align_ori_images)

        tar_images = load_images_list(images_ref_i, img_shape) # origin_images (b,H,W,C) numpy.array
        align_tar_images = tar_images
        #构造labels
        align_lab_images = make_inputs(align_tar_images)

        labels = model(align_lab_images.cuda())

        # attack
        adv_images = atk(input_images, labels, targeted=True) # (b,C,H,W) torch.tensor
        adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy() # (b,H,W,C) numpy.array

        if save_img:
            # save image
            for i in range(len(adv_images)):
                _ = save_adv_image(adv_images[i], sub_root+'-'+images_names[i], dataset_root)

        if save_noise:
            noise = extract_noise(adv_images, origin_images)
            for i in range(len(noise)):
                _ = save_adv_image(noise[i], sub_root+'-'+images_names[i], noise_root)
            
    print('Attack compete')


def extract_noise(adv_images, origin_images):

    adv_images = adv_images.astype(np.float64)
    origin_images = origin_images.astype(np.float64)

    noise = []
    for (adv_img, ori_img) in zip(adv_images, origin_images):
        mask = np.abs(adv_img - ori_img)
        noise.append(mask.astype(np.uint8))
    
    return noise
        

def align_faces():

    print('Align faces')

    base_data_root = 'E:/Files/code/Dataset_Face/celeba-align'
    dataset_root = 'E:/Files/code/Dataset_Face/celeba-align-112-112'
    batch_size = 100
    img_shape = (112, 112)
    mtcnn = align
    save_img = True
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root) # 创建数据保存路径

    # 抽取图像
    img_list = os.listdir(base_data_root) # eg 000291.jpg
    total = len(img_list)
    print('Nums of images', total)

    # total = min(total, 50000)
    print('Nums of images to align', total)

    start_idx = 51440

    remain_total = total - start_idx

    if int(remain_total%batch_size) == 0:
        epoch = int(remain_total/batch_size)
    else:
        epoch = int(remain_total/batch_size)+1
    

    for i in tqdm(range(epoch)):
        # print('turn ', i)

        l = i * batch_size + start_idx
        r = min((i + 1) * batch_size + start_idx, total)
        # 获取图片名
        images_i= img_list[l : r]
        # print(images_names)

        # 读取原图片
        origin_images = []
        images_names = []
        for img_name in images_i:
            img_dir = os.path.join(base_data_root, img_name)
            img = cv2.imread(img_dir)
            origin_images.append(img)
            images_names.append(img_name)
        origin_images = np.array(origin_images) # origin_images (b,H,W,C) numpy.array
        # print('Origin 0-255 numpy images shape', origin_images.shape)

        # origin人脸对齐
        align_ori_images, _ = align_images(origin_images, mtcnn, img_shape) # M保存的align信息
        # print('Aligned origin 0-255 numpy images shape', align_ori_images.shape)

        if save_img:
            # save image
            for i in range(len(align_ori_images)):
                _ = save_adv_image(align_ori_images[i], images_names[i], dataset_root)
        
    print('align compete')

def input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="The GPU ID")
    parser.add_argument("--people", type=str, default='E:/Files/code/Dataset_Face/lfw/people.txt',
                        help="The people.txt path")
    parser.add_argument("--people_adv", type=str, default='E:/Files/code/Dataset_Face/adv_face/align/people_adv.txt',
                        help="The people_adv.txt path")
    parser.add_argument("--save_dir", type=str, default='./data',
                        help="data list save path")
    parser.add_argument("--batch_size", type=str, default=20,
                        help="training & validation batch size")
    parser.add_argument("--save_img", type=bool, default=True,
                        help="save images or not")
    parser.add_argument("--server", type=bool, default=True,
                        help="run on server")
    parser.add_argument("--dataset", type=str, default='celeba-all', #'celebahq'
                        help="base dataset name")
    parser.add_argument("--frs", type=str, default='ArcFace', # 'CosFace', # 'E:/Files/code/Dataset_Face/deep_fake_detect/image/train/',
                        help="face model ")
    parser.add_argument("--targeted", type=bool, default=False,
                        help="calculate attack success rate or not")
    parser.add_argument("--save_noise", type=bool, default=False,
                        help="save noise mask")
    parser.add_argument("--attack_list", type=str, default=['FGSM','BIM','PGD','RFGSM','MIFGSM','DIFGSM','TIFGSM','TIPIM'],
                        help="face model ")
    return parser.parse_args()

def main():
    args = input_args()
    people_txt = args.people
    people_adv_txt = args.people_adv
    dataset_name = args.dataset
    # atk_method = args.attack
    model_name = args.frs
    batch_size = args.batch_size
    save_img = args.save_img
    targeted = args.targeted
    save_noise = args.save_noise
    attack_list = args.attack_list
    device = torch.device('cuda')

    num_total = 1680 # 数据集类别数，LFW1680
    num = 1680 # 生成对抗样本数目(1-1680)，1680代表生成所有
    # eps = [5, 10, 15] # [6,9,12,15]
    # alphas = [1, 2, 3] # [0.5,1,1.5,2]
    # Steps = [5, 10, 20]
    
    eps = [10] #[5,10,15] 
    alphas = [1] #[1,2,3] 
    Steps = [5] #[1,5,10]
    gamma = [0]

    if model_name == 'ArcFace' or model_name == 'CosFace' or model_name == 'SphereFace':
        model, img_shape = getmodel(model_name)   # image_shape=(112,112) if ArcFace
        mtcnn = align # 我目前使用的与ArcFace配套的MTCNN模型，输入(H,W,C)numpy.array, 输出人脸部分的(H,W,C)numpy_.rray
    elif model_name == 'Facenet':
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device) # vggface2 casia-webface
        img_shape = 112
        margin = 0
        mtcnn = MTCNN(image_size=img_shape, margin=margin, post_process=False)
        # 与FaceNet配套的MTCNN，输入PILImage, 输出人脸部分的(C,H,W)torch.tensor
    else:
        raise Exception

    print('Using ', model_name)

    if dataset_name == 'LFW':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/lfw-112-112'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = 'E:/Files/code/Dataset_Face/lfw-112-96'
        # adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face/align'
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face/tmp'
        run = run_lfw
        targets_dir = ''
        if targeted:
            targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets'
        
    elif dataset_name == 'celebahq':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/CelebA-HQ-112-112'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = 'E:/Files/code/Dataset_Face/CelebA-HQ-112-96'
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face'
        run = run_normal
        targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets'
    
    elif dataset_name == 'LFW-all':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/lfw-112-112'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = 'E:/Files/code/Dataset_Face/lfw-112-96'
        # adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face/align'
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face'
        run = run_normal
        targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets-celeba'
    
    elif dataset_name == 'celebahq-all':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/CelebA-HQ-112-112/all'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = ''
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face'
        run = run_normal
        targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets'
    
    elif dataset_name == 'celeba-all':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/celeba-align-112-112'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = ''
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face'
        run = run_normal
        targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets'

    if args.server:
        model = nn.DataParallel(model)

    time0 = time.time()

    for atk_method in attack_list:
        for ep in eps:
            for alpha in alphas:
                for steps in Steps:
                    params = {'eps':ep, 'alpha':alpha, 'steps':steps, 'gamma':gamma[0]}
                    time1 = time.time()
                    run(model_name, atk_method, dataset_name, num_total, num, batch_size, base_data_root, people_adv_txt, model, img_shape, mtcnn, params, save_img, adv_dataset_root, targets_dir, save_noise)
                    # run_normal(model_name, atk_method, dataset_name, batch_size, base_data_root, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise)

                    time2 = time.time()
                    print('Time: ', time2 - time1)

    time2 = time.time()
    print('Total time: ', time2 - time0)

def tmp():
    args = input_args()
    people_txt = args.people
    people_adv_txt = args.people_adv
    dataset_name = args.dataset
    # atk_method = args.attack
    model_name = args.frs
    batch_size = args.batch_size
    save_img = True
    targeted = args.targeted
    save_noise = False
    attack_list = args.attack_list
    device = torch.device('cuda')

    num_total = 1680 # 数据集类别数，LFW1680
    num = 1680 # 生成对抗样本数目(1-1680)，1680代表生成所有
    # eps = [5, 10, 15] # [6,9,12,15]
    # alphas = [1, 2, 3] # [0.5,1,1.5,2]
    # Steps = [5, 10, 20]
    
    eps = [15] #[5,10,15] 
    alphas = [1] #[1,2,3] 
    Steps = [10] #[1,5,10]
    gamma = [0]

    if model_name == 'ArcFace' or model_name == 'CosFace' or model_name == 'SphereFace':
        model, img_shape = getmodel(model_name)   # image_shape=(112,112) if ArcFace
        mtcnn = align # 我目前使用的与ArcFace配套的MTCNN模型，输入(H,W,C)numpy.array, 输出人脸部分的(H,W,C)numpy_.rray
    elif model_name == 'Facenet':
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device) # vggface2 casia-webface
        img_shape = 112
        margin = 0
        mtcnn = MTCNN(image_size=img_shape, margin=margin, post_process=False)
        # 与FaceNet配套的MTCNN，输入PILImage, 输出人脸部分的(C,H,W)torch.tensor
    else:
        raise Exception

    print('Using ', model_name)

    if dataset_name == 'LFW':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/lfw-112-112'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = 'E:/Files/code/Dataset_Face/lfw-112-96'
        # adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face/align'
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face/align'
        # run = run_lfw
        targets_dir = 'E:/Files/code/Dataset_Face/CelebA-HQ-112-112/celebahq-112-112'
        # if targeted:
        #     targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets'
        for atk_method in attack_list:
            for ep in eps:
                for alpha in alphas:
                    for steps in Steps:
                        params = {'eps':ep, 'alpha':alpha, 'steps':steps, 'gamma':gamma[0]}
                        run_lfw(model_name, atk_method, dataset_name, num_total, num, batch_size, base_data_root, people_adv_txt, model, img_shape, mtcnn, params, save_img, adv_dataset_root, targets_dir, save_noise)
                    
                        run_lfw_tmp(model_name, atk_method, dataset_name, num_total, num, batch_size, base_data_root, people_txt, model, img_shape, mtcnn, params, save_img, adv_dataset_root, targets_dir, save_noise)

    elif dataset_name == 'celebahq':
        if model_name == 'ArcFace': 
            base_data_root = 'E:/Files/code/Dataset_Face/CelebA-HQ-112-112/celebahq-112-112'
        elif model_name == 'CosFace' or model_name == 'SphereFace':
            base_data_root = 'E:/Files/code/Dataset_Face/CelebA-HQ-112-96/celebahq-112-96'
        # adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face/align'
        adv_dataset_root = 'E:/Files/code/Dataset_Face/adv_face'
        # run = run_lfw
        targets_dir = 'E:/Files/code/Dataset_Face/LFW-Augmentation/Origin'
        # if targeted:
        #     targets_dir = 'E:/Files/code/Dataset_Face/lfw-sample/single-targets'
        txt_file = 'E:/Files/code/GAPS/data/SP/celebahq_train.labels.txt'
        # for atk_method in attack_list:
        #     for ep in eps:
        #         for alpha in alphas:
        #             for steps in Steps:
        #                 params = {'eps':ep, 'alpha':alpha, 'steps':steps, 'gamma':gamma[0]}
        #                 # run_normal(model_name, atk_method, dataset_name, batch_size, base_data_root, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise)
        #                 run_normal_tmp1(model_name, atk_method, dataset_name, batch_size, base_data_root, txt_file, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise)
        for atk_method in attack_list:
            for ep in eps:
                for alpha in alphas:
                    for steps in Steps:
                        params = {'eps':ep, 'alpha':alpha, 'steps':steps, 'gamma':gamma[0]}
                        txt_file_val = 'E:/Files/code/GAPS/data/SP/' + dataset_name+'-'+atk_method+'-eps'+str(ep)+'_val.labels.txt'
                        txt_file_test = 'E:/Files/code/GAPS/data/SP/' + dataset_name+'-'+atk_method+'-eps'+str(ep)+'_test.labels.txt'
                        # run_normal(model_name, atk_method, dataset_name, batch_size, base_data_root, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise)
                        run_normal_tmp1(model_name, atk_method, dataset_name, batch_size, base_data_root, txt_file_val, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise)
                        run_normal_tmp1(model_name, atk_method, dataset_name, batch_size, base_data_root, txt_file_test, model, img_shape, params, save_img, adv_dataset_root, targets_dir, save_noise)

    

if __name__=='__main__':
    main()
    # tmp()
    # align_faces()