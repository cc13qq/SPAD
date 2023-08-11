import os
import random
from imageio import imread, imsave
import numpy as np
from torchvision import transforms
import tensorflow.compat.v1 as tf
from PIL import Image
import cv2
import torch
from align_methods import re_align
import shutil

def get_img_list(data_root):
    # get image list
    file_list = os.listdir(data_root)
    img_list = []
    for file in file_list:
        file_dir = os.path.join(data_root, file)
        if os.path.isfile(file_dir):
            if file_dir.endswith('jpg') or file_dir.endswith('png') or file_dir.endswith('JPEG'):
                img_list.append(file_dir)
        elif os.path.isdir(file_dir):
            for sub_file in os.listdir(file_dir):
                sub_file_dir = os.path.join(file_dir, sub_file)
                if sub_file_dir.endswith('jpg') or sub_file_dir.endswith('png') or sub_file_dir.endswith('JPEG'):
                    img_list.append(sub_file_dir)

    return img_list

def sample_images_per_class(people_adv_txt):
    # 抽取batch_size个图像，返回人名people_name，用于生成的图像images_gen，用于测试的图像images_ref
    # input: people_adv.txt dir, num_total, batch_size
    # output: people names list, images_gen list, images_ref list
    # sample identities
    people = []
    with open(people_adv_txt, 'r') as f:
        i = 0
        for line in f.readlines():
            line = line.strip('\n')
            people.append(line)
            i += 1
        f.close()
    # get image names
    images_gen = []
    images_ref = []
    people_name = []
    for line in people:
        splits = line.split()
        name = splits[0]
        num =  int(splits[1])
        # sample image_gen
        for idx in range(1,num):
            if idx < 9:
                images_gen.append(name + '_000'+ str(idx+1)+'.jpg')
            elif idx < 99:
                images_gen.append(name + '_00'+ str(idx+1)+'.jpg')
            else:
                images_gen.append(name + '_0'+ str(idx+1)+'.jpg')
            images_ref.append(name + '_000'+ str(1)+'.jpg')
            people_name.append(name)
    
    return people_name, images_gen, images_ref

def sample_images_per_class_tmp(people_adv_txt):
    # 抽取batch_size个图像，返回人名people_name，用于生成的图像images_gen，用于测试的图像images_ref
    # input: people_adv.txt dir, num_total, batch_size
    # output: people names list, images_gen list, images_ref list
    # sample identities
    people = []
    with open(people_adv_txt, 'r') as f:
        i = 0
        for line in f.readlines():
            line = line.strip('\n')
            people.append(line)
            i += 1
        f.close()
    # get image names
    people_name = []
    images_gen = []
    for line in people:
        splits = line.split()
        if len(splits) < 2:
            continue
        #print(splits)
        name = splits[0]
        num =  int(splits[1])
        # sample image_gen
        images_gen.append(name + '_000'+ str(1)+'.jpg')
        people_name.append(name)
    
    return people_name, images_gen

def sample_images(people_adv_txt, num_total = 1680, num = 10):
    # 抽取num个图像，返回人名people_name，用于生成的图像images_gen，用于测试的图像images_ref
    # input: people_adv.txt dir, num_total, num
    # output: people names list, images_gen list, images_ref list
    # sample identities
    people = []
    idxs = random.sample(range(num_total), num)
    with open(people_adv_txt, 'r') as f:
        i = 0
        for line in f.readlines():
            if i in idxs:
                line = line.strip('\n')
                people.append(line)
            i += 1
        f.close()
    # get image names
    images_gen = []
    images_ref = []
    people_name = []
    for line in people:
        splits = line.split()
        name = splits[0]
        num =  int(splits[1])
        images_ref.append(name + '_000'+ str(1)+'.jpg')
        people_name.append(name)
        # randomly sample image_gen
        idx = random.randint(1, num-1)
        if idx < 9:
            images_gen.append(name + '_000'+ str(idx+1)+'.jpg')
        else:
            images_gen.append(name + '_00'+ str(idx+1)+'.jpg')
    
    return people_name, images_gen, images_ref

def load_images(file_path): 
    # input .txt file and image file path and load image as numpy.array
    # output numpy.array images and path list
    imgs_path = []
    file_names = os.listdir(file_path)
    for file_name in file_names:
        imgs_path.append(os.path.join(file_path, file_name))

    # 转成0-255numpy.tensor
    imgs = [cv2.imread(img_path) for img_path in imgs_path] # (H,W,C)numpy.array
    '''****'''
    # img_tmp1 = imread('E:/Files/code/Dataset_Face/lfw/Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0001.jpg')
    # img_tmp2 = imread('E:/Files/code/Dataset_Face/TALFW/Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0001.jpg')
    # imgs.append(img_tmp1)
    # imgs.append(img_tmp2)
    '''****'''
    images = np.array(imgs) # (b,H,W,C)numpy.array
    return images, imgs_path

def load_txt_images(txt_path, file_path):
    # input .txt file and image file path and load image as numpy.array
    # output numpy.array images and path list
    imgs_path = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            imgs_path.append(os.path.join(file_path, line))
        f.close()

    # 转成0-255numpy.tensor
    imgs = [cv2.imread(img_path) for img_path in imgs_path] # (H,W,C)numpy.array
    images = np.array(imgs) # (b,H,W,C)numpy.array
    return images, imgs_path

def load_list_images(image_list, people_list, file_path):
    # input image_list and image file path and load image as numpy.array
    # output numpy.array images and path list
    imgs_path = []
    for i in range(len(image_list)):
        imgs_path.append(os.path.join(file_path, people_list[i], image_list[i]))

    # 转成0-255numpy.tensor
    imgs = [cv2.imread(img_path) for img_path in imgs_path] # (H,W,C)numpy.array
    images = np.array(imgs) # (b,H,W,C)numpy.array
    return images, imgs_path

def load_list_images_resize(image_list, people_list, file_path, img_shape):
    # input image_list and image file path and load image as numpy.array
    # output numpy.array images and path list
    imgs_path = []
    for i in range(len(image_list)):
        imgs_path.append(os.path.join(file_path, people_list[i], image_list[i]))

    # 转成0-255numpy.tensor
    imgs = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        cropped_img = cv2.resize(img, (img_shape[1], img_shape[0]))
        imgs.append(cropped_img)
    images = np.array(imgs) # (b,H,W,C)numpy.array
    return images, imgs_path

def load_adv_images(image_list, prefix, file_path):
    # input image_list and image file path and load image as numpy.array
    # output numpy.array images and path list
    imgs_path = []
    for i in range(len(image_list)):
        img = prefix + image_list[i]
        imgs_path.append(os.path.join(file_path, img))

    # 转成0-255numpy.tensor
    imgs = [cv2.imread(img_path) for img_path in imgs_path] # (H,W,C)numpy.array
    images = np.array(imgs) # (b,H,W,C)numpy.array
    return images, imgs_path

def make_dir_name(atk_model, atk):
    # 构造子文件夹名, output atk_model-pram1value1-pram2value2
    atk_attrs = str(atk).split()
    dir_name = atk_model
    pass_list = ['device', 'attack_mode', 'return_type', 'random_start', 'gain', 'kernel_name', 'len_kernel', 'nsig', 'stacked_kernel']
    for i in range(1, len(atk_attrs)):
        splits = atk_attrs[i].split('=')
        pram_key = splits[0]
        pram_value = splits[1].split(',')[0]
        if pram_key in pass_list:
            continue
        else:
            dir_name+='-'+pram_key+pram_value
    return dir_name

def get_txt_names(txt_path): 
    # input .txt file and image file path and load image as numpy.array
    # output numpy.array images and path list
    names = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            names.append(line.split('.')[0])
        f.close()
    return names

def align_images(images, mtcnn, img_shape):
    # using align as mtcnn
    # input image list to align images with mtcnn
    # output numpy.array (b,H,W,C) aligned images and M list
    align_images = []
    M = [] # 保存的align信息
    for i in range(images.shape[0]):
        align_img, m_img = mtcnn(images[i]) # 用align人脸对齐，(H,W,C)numpy.array
        align_img = cv2.resize(align_img, (img_shape[1], img_shape[0])) # 裁剪至112
        align_images.append(align_img)
        M.append(m_img)
    align_images = np.array(align_images) # (b,H,W,C)numpy.array
    return align_images, M

def align_images_facenet(images, mtcnn):
    # using facenet_pytorch.MTCNN as mtcnn
    # input image list to align images with mtcnn
    # output numpy.array (b,H,W,C) aligned images
    align_images = []

    for i in range(images.shape[0]):
        align_img = mtcnn(images[i]).permute(1,2,0) # 用MTCNN人脸对齐，(H,W,C)torch.tensor
        align_img = align_img.numpy().astype(np.uint8) # (H,W,C)numpy.array
        align_images.append(align_img)

    align_images = np.array(align_images) # (b,H,W,C)numpy.array
    return align_images

def make_inputs(align_images):
    # input numpy.array aligh_images
    # output (b,C,H,W) tensor images
    input_images = torch.Tensor(align_images)
    input_images = input_images.permute(0, 3, 1, 2) # (b,C,H,W)torch.tensor
    return input_images

def re_align_images(adv_images, align_ori_images, origin_images, M):
    # input numpy.array (b,H,W,C) adversarial images, aligned origin images, origin_images and M
    # output numpy.array (b,H,W,C) realigned images
    realign_images = []
    for adv_img, ali_img, ori_img, m_img in zip(adv_images, align_ori_images, origin_images, M):
        adv_img = cv2.resize(adv_img, (112, 112))
        ali_img = cv2.resize(ali_img, (112, 112))
        realign_img = re_align(adv_img, ali_img, ori_img, m_img)
        realign_images.append(realign_img)
    realign_images = np.array(realign_images)
    return realign_images

def save_adv_image1(image, img_name, output_dir):#保存图片
    """Saves images to the output directory.
    input numpy.array (H,W,C) image, img_name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # path = src + '.jpg'
    with tf.gfile.Open(os.path.join(output_dir, img_name), 'w') as f:
        image = np.clip(image, 0, 255).astype(np.uint8)
        imsave(f, image.astype(np.uint8), format='jpg')
    return os.path.join(output_dir, img_name)

def save_adv_image(image, img_name, output_dir):#保存图片
    """Saves images to the output directory.
    input numpy.array (H,W,C) image, img_name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # path = src + '.jpg'
    image = np.clip(image, 0, 255).astype(np.uint8)
    if img_name.endswith('.png'):
        cv2.imwrite(os.path.join(output_dir, img_name), image.astype(np.uint8))
        # print('saving ', img_name)
    elif img_name.endswith('.jpg'):
        img_name = img_name.split('.jpg')[0]
        cv2.imwrite(os.path.join(output_dir, img_name+'.png'), image.astype(np.uint8))
        # print('saving ', img_name+'.png')
    return os.path.join(output_dir, img_name)

def save_adv_image_resize(image, img_name, output_dir, resize=(250, 250)):#保存图片
    """Saves images to the output directory.
    input numpy.array (H,W,C) image, img_name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # path = src + '.jpg'
    with tf.gfile.Open(os.path.join(output_dir, img_name), 'w') as f:
        cropped_img = cv2.resize(image, resize)
        image = np.clip(cropped_img, 0, 255).astype(np.uint8)
        imsave(f, image.astype(np.uint8), format='jpg')
    return os.path.join(output_dir, img_name)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def similarity_score(x1, x2):
    cos = cosin_metric(x1, x2)
    score = (cos + 1) / 2
    return score

def cal_score(ori_feas, adv_feas, ref_feas, mode):
    if mode == 'cosin':
        metric = cosin_metric
    elif mode == 'sim':
        metric = similarity_score
    else:
        raise Exception
    # calculate metric
    score = []
    for i in range(ori_feas.shape[0]):
        # print('Identity ', i)
        # print('Reference idx ', j, j+1, j+2)
        ori_score = []
        ori_score.append(metric(ori_feas[i], ori_feas[i]))
        ori_score.append(metric(ori_feas[i], adv_feas[i]))
        ori_score.append(metric(ori_feas[i], ref_feas[i]))
        adv_score = []
        adv_score.append(metric(adv_feas[i], ori_feas[i]))
        adv_score.append(metric(adv_feas[i], adv_feas[i]))
        adv_score.append(metric(adv_feas[i], ref_feas[i]))
        score.append([ori_score, adv_score])
    return np.array(score)

def cal_score_muti(ori_feas, adv_feas, ref_feas, mode):
    if mode == 'cosin':
        metric = cosin_metric
    elif mode == 'sim':
        metric = similarity_score
    else:
        raise Exception
    # calculate metric
    score = []
    for i in range(ori_feas.shape[0]):
        # print('Identity ', i)
        j = i*3
        # print('Reference idx ', j, j+1, j+2)
        ori_score = []
        ori_score.append(metric(ori_feas[i], ori_feas[i]))
        ori_score.append(metric(ori_feas[i], adv_feas[i]))
        ori_score.append(metric(ori_feas[i], ref_feas[j]))
        ori_score.append(metric(ori_feas[i], ref_feas[j+1]))
        ori_score.append(metric(ori_feas[i], ref_feas[j+2]))
        adv_score = []
        adv_score.append(metric(adv_feas[i], ori_feas[i]))
        adv_score.append(metric(adv_feas[i], adv_feas[i]))
        adv_score.append(metric(adv_feas[i], ref_feas[j]))
        adv_score.append(metric(adv_feas[i], ref_feas[j+1]))
        adv_score.append(metric(adv_feas[i], ref_feas[j+2]))
        score.append([ori_score, adv_score])
    return np.array(score)


# some transform utils
def PIL_transform(pil_image): # transform PIL image to [0,1]tensor
    transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
# Torchattacks only supports images with a range between 0 and 1.
])
    return transform(pil_image)

def tensor_transform(tensor): # transform [0,255]tensor to [0,1]tensor
    return tensor/255

def numpy_transform(np_array): #transform [0,255]numpy array to [0,1]tensor
    img_tmp = Image.fromarray(np_array)
    return PIL_transform(img_tmp)

def tensor_to_image(tensor, mode=1):
    if mode == 1:
        toimage = transforms.ToPILImage()
    elif mode == 255:
        tensor = tensor_transform(tensor)
        toimage = transforms.ToPILImage()
    else:
        print('mode must be 1 or 255')
        raise Exception
    return toimage(tensor)

def build_data_txt(data_path):
    # data_path: 图片所在文件夹
    # 构造txt文件，包含所有图片名
    # output: None
    img_names = []
    filelist = os.listdir(data_path)
    for img_name in filelist:
        #filename = os.path.join(path , filename)
        img_names.append(img_name)
    txt_file = os.path.join(data_path,'data0.txt')

    with open(txt_file, "w") as f:
        for img_name in img_names:
            f.write(img_name+'\n')
        f.close()
    
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            print(line)
        f.close()

def rename1(path): # BIM
    for sub_root in os.listdir(path):
        adv_ano = sub_root
        sub_root = os.path.join(path, sub_root)
        print('Rename in ', sub_root)
        for name in os.listdir(sub_root):
            tmp_name = name
            if len(tmp_name.split(adv_ano+'_')) > 1:
                tmp_name = tmp_name.split(adv_ano+'_')[1] # 去掉前缀
                tmp_name = adv_ano + '-' + tmp_name
            if len(tmp_name.split('.jpg')) > 2:
                tmp_name = tmp_name.split('.jpg')[0] # 去掉多余的.jpg
                new_name = tmp_name + '.jpg'
            else:
                new_name = name
            if name != new_name:
                os.rename(os.path.join(sub_root,name),os.path.join(sub_root,new_name))
    print('Compelete')


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def set_log(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def copy_file(from_paths, to_paths):
    # copy file
    print('Copy files')
    if isinstance(from_paths, list) and isinstance(to_paths, list):
        for fp, tp in zip(from_paths, to_paths):
            shutil.copyfile(fp, tp)
    elif isinstance(from_paths, str) and isinstance(to_paths, str):
        shutil.copyfile(from_paths, to_paths)
    print('Done')

def merge_lists(list1_txt, list2_txt):
    list_new = []
    with open(list1_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            list_new.append(line)
        f.close()
    with open(list2_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            list_new.append(line)
        f.close()
    random.shuffle(list_new)
    return list_new

def merge_lists_save(list1_txt, list2_txt, save_dir):
    list_new = []
    with open(list1_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            list_new.append(line)
        f.close()
    with open(list2_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            list_new.append(line)
        f.close()
    random.shuffle(list_new)
    name = 'tmp'
    txt_file = os.path.join(save_dir, name+'.txt')
    with open(txt_file, "w") as f:
        for img in list_new:
            f.write(img+'\n')
        f.close()
    print('done')
    return txt_file


def crop_eye_area(img, detector, predictor):
    # import dlib
    # detector = dlib.get_frontal_face_detector()
    # tool_path = 'E:/Files/code/tools'
    # predictor = dlib.shape_predictor(tool_path+"/shape_predictor_68_face_landmarks.dat")
    crop_img = img.copy()
    
    eye_points = [0, 16] + [i for i in range(17,29)] # 眼部区域点
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)

    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    
    left = int(min(landmarks[tt][:,0]))
    right = int(max(landmarks[tt][:,0]))
    top =  int(min(landmarks[tt][:,1]))
    bottom =  int(max(landmarks[tt][:,1]))

    crop_img = crop_img[top:bottom,left:right]
    return crop_img