import os
from PIL import Image
import numpy as np
import dlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

def save_adv_image(image, img_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # path = src + '.jpg'
    image = np.clip(image, 0, 255).astype(np.uint8)
    if img_name.endswith('.png'):
        cv2.imwrite(os.path.join(output_dir, img_name), image.astype(np.uint8))
    elif img_name.endswith('.jpg'):
        img_name = img_name.split('.jpg')[0]
        cv2.imwrite(os.path.join(output_dir, img_name+'.png'), image.astype(np.uint8))
    return os.path.join(output_dir, img_name)

def get_landmark(img):
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(os.path.dirname(__file__), './shape_predictor_81_face_landmarks.dat')
    face_predictor = dlib.shape_predictor(predictor_path)
    faces = face_detector(img, 1)
    if len(faces) == 0:
        # print('detect no faces')
        faces = [dlib.rectangle(0,0,img.shape[0],img.shape[1])]
    landmark = face_predictor(img, faces[0])

    return landmark, faces

def random_resize(img, scale):
    scales = np.random.randint(low=int(scale/2), high=scale*2, size=(2))
    #print('scales ', scales)
    imgr = cv2.resize(img, (scales[0], scales[1]))
    return imgr

def random_rotate_image(img, crop=False):
    angle = np.random.uniform(-360, 360)
    #print('angle ', angle)
    h, w = img.shape[:2]
    
    if h<w: 
        blank = np.zeros((int((w-h)/2),w,3), dtype=np.uint8)
        img = np.concatenate((blank, img, blank),0)
        h = int((w-h)/2)*2
    elif h>w:
        blank = np.zeros((h,int((h-w)/2),3), dtype=np.uint8)
        img = np.concatenate((blank, img, blank),1)
        w = int((h-w)/2)*2    
    h, w = img.shape[:2]
    
    angle %= 360
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    if crop:
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0
        hw_ratio = float(h) / float(w)
        
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator
        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = img[y0:y0+h_crop, x0:x0+w_crop]
    return img_rotated


def random_transform(img, num_high):
    num = np.random.randint(low=1, high=num_high)
    scale=max(img.shape[0], img.shape[1])
    for i in range(num):
        img = random_resize(img, scale)
        img = random_rotate_image(img, crop=False)
    img = random_rotate_image(img, crop=True)
    return img

def grad_image(ori_img, mode = 'plus'):
    grad_x = cv2.Sobel(ori_img, -1, 1, 0)
    grad_y = cv2.Sobel(ori_img, -1, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    if mode == 'werighted':
        gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    elif mode == 'plus':
        gradxy = gradx + grady
    return gradxy

def get_face_matrix_opt(ori_img, loc = 'face', landmark=None, size = 1024):  
    face_lmk_fa = [i for i in range(17)] + [78, 74, 79 ,73, 72, 80, 71, 70, 69, 68, 76, 75, 77]
    face_lmk_le = [i for i in range(17,22)] + [39, 40, 41, 36]
    face_lmk_re = [i for i in range(22,27)] + [45,46,47,42]
    face_lmk_nt = [i for i in range(30,36)]
    face_lmk_mo = [i for i in range(48,60)]
    
    face_lmk = dict()
    face_lmk.update({'face': face_lmk_fa})
    face_lmk.update({'left_eye': face_lmk_le})
    face_lmk.update({'right_eye': face_lmk_re})
    face_lmk.update({'nose_tip': face_lmk_nt})
    face_lmk.update({'mouse': face_lmk_mo})
    face_lmk.update({'outlie': face_lmk_fa})
        
    if landmark == None:
        landmark, _ = get_landmark(ori_img)
    
    img_tmp = Image.new('L', (size, size), 0)
    
    if isinstance(loc, list):
        for pos in loc:
            face_pt = []
            for idx in face_lmk[pos]:
                pt = landmark.parts()[idx]
                face_pt.append((pt.x, pt.y))
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=None, fill=1)
    elif isinstance(loc, str):
        face_pt = []
        for idx in face_lmk[loc]:
            pt = landmark.parts()[idx]
            face_pt.append((pt.x, pt.y))
        if loc == 'outlie':
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=1, fill=None)
        else:
            ImageDraw.Draw(img_tmp).polygon(face_pt, outline=None, fill=1)
    convex_mask = np.asarray(img_tmp) 
    
    idxs_tmp = np.where(convex_mask != 0)
    face_matrix = np.concatenate((np.expand_dims(idxs_tmp[0], axis=-1), np.expand_dims(idxs_tmp[1], axis=-1)), axis = -1)
    
    return convex_mask, face_matrix

def get_GC_scaller(size):
    radius = int(size/2) 
    assert radius*2 == size
    GC_scaler_11 = np.zeros((radius, radius)) 

    for i in range(GC_scaler_11.shape[0]):
        for j in range(GC_scaler_11.shape[1]):
            r = int(np.sqrt(i**2+j**2)) 
            partial = 1 - r / radius
            if partial > 0:
                GC_scaler_11[i,j]= partial
            else:
                continue
    GC_scaler_10 = cv2.flip(GC_scaler_11, 1) 
    GC_scaler_1 = np.concatenate((GC_scaler_10, GC_scaler_11), 1) 
    GC_scaler_0 = cv2.flip(GC_scaler_1, 0) 
    GC_scaler = np.concatenate((GC_scaler_0, GC_scaler_1), 0) 

    GC_scaler = np.expand_dims(GC_scaler, axis=-1)
    GC_scaler = np.repeat(GC_scaler, 3, axis = -1)
    return GC_scaler

def random_gradient_color_opt(size, GC_scaler, low=0, high=255, image_post = True):
    GC = np.ones((size, size, 3))
    assert GC.shape == GC_scaler.shape
    center_color = np.random.randint(low, high, size=(3))
    GC = GC * center_color * GC_scaler
    if image_post :
        return GC.astype(np.uint8)
    else:
        return GC.astype(np.int32)

def add_pixel_GC_continue_opt(ori_img, eps=100, prob=0.5, sl = [10, 200], loc = 'local', GC_size = 256, th = 50, face_matrix=None, GC_scaler=None, image_post = True):
    h, w, c = ori_img.shape
    mask = np.zeros([h, w, c])
    gradxy = grad_image(ori_img, mode = 'plus')
    
    if face_matrix is None:
        face_matrix_size = np.max(ori_img.shape)
        if loc == 'local':
            _, face_matrix = get_face_matrix_opt(ori_img, loc = ['left_eye', 'right_eye', 'nose_tip', 'mouse'], size = face_matrix_size)
        elif loc == 'global':
            _, face_matrix = get_face_matrix_opt(ori_img, loc = 'face', size = face_matrix_size)
        else:
            _, face_matrix = get_face_matrix_opt(ori_img, loc = loc, size = face_matrix_size)
    
    if GC_scaler is None:
        GC_scaler = get_GC_scaller(GC_size)
    
    GCs = [] 
    mask_poss = []
    GC_poss = []

    gradxy_m = gradxy[face_matrix[:,0],face_matrix[:,1],:]
    gradxy_m_idx = np.where(np.mean(gradxy_m, axis=-1)>th)[0]
    idxs_i = face_matrix[:,0][gradxy_m_idx]
    idxs_j = face_matrix[:,1][gradxy_m_idx]
    idx_m = np.concatenate((np.expand_dims(idxs_i,axis=-1), np.expand_dims(idxs_j,axis=-1)), axis = -1)
    num_s_m = int(idx_m.shape[0] * prob)
    np.random.shuffle(idx_m)
    idx_selected = idx_m[:num_s_m,:]

    for idx in idx_selected:
                
        GC = random_gradient_color_opt(GC_size, GC_scaler, 0, 100, image_post = True)
        GC = random_transform(GC, 2) 
        scales = np.random.randint(low=sl[0], high=sl[1], size=(2))
        GC = cv2.resize(GC, (scales[0],scales[1]))
        
        tx = max(int(idx[0] - GC.shape[0]/2), 0) # top x
        ly = max(int(idx[1] - GC.shape[1]/2), 0) # left y
        bx = min(tx + GC.shape[0], h) # bottom x
        ry = min(ly + GC.shape[1], w) # right y
        
        GCs.append(GC)
        mask_poss.append((tx,bx,ly,ry))
        GC_poss.append((min(GC.shape[0], h - tx), min(GC.shape[1], w - ly)))
    
    if len(GCs) > 0:
        select_idx = np.random.randint(low = 0, high = len(GCs))
        GC_m = GCs[select_idx]
        for (GC, mask_pos, GC_pos) in zip (GCs, mask_poss, GC_poss):
            (tx,bx,ly,ry) = mask_pos
            (GC_x1, GC_y1) = GC_pos
            if np.random.rand() < 0.8:
                GC_t = cv2.resize(GC_m, (GC.shape[1], GC.shape[0]))
                mask[tx:bx, ly:ry] += GC_t.astype(np.uint32)[:GC_x1, :GC_y1] 
            else: 
                mask[tx:bx, ly:ry] += GC.astype(np.uint32)[:GC_x1, :GC_y1] 

    mask = mask.clip(min=-eps, max=eps)
    GC_img = (ori_img.astype(np.uint32) + mask).clip(min=0, max=255)
    if image_post :
        return GC_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return GC_img.astype(np.uint32), mask.astype(np.uint32)

def rap_process_opt(ori_img, eps=None, sl=6, noise_type=None, image_post = False, al=3):
    if eps is None:
        eps_dic = [5, 10]
        para_random = np.random.randint(low=0, high=2)
        eps=eps_dic[para_random]
    if noise_type is None:
        type_dic = ["point", "block", "mix"]
        type_random = np.random.randint(low=0, high=3)
        noise_type=type_dic[type_random]
    img = ori_img.astype(np.int32)
    mask = RAP_opt(img, eps=eps, sl=sl, noise_type=noise_type, al=al)
    mask = mask.clip(min=-eps, max=eps)
    rap_img = (img + mask).clip(min=0, max=255)
    if image_post:
        return rap_img.astype(np.uint8), mask.astype(np.uint8)
    else:
        return rap_img.astype(np.int32), mask.astype(np.int32)

def RAP_opt(ori_img, eps=5, sl=1, noise_type='point', al=3):
    assert noise_type in ['point', 'block', 'mix']
    h, w, c = ori_img.shape
    point_mask = np.random.randint(low = -eps,high = eps+1,size = (h,w,c))
    if noise_type=='point':
        return point_mask
    al_mask = np.random.randint(low = -al,high = al+1,size = (h,w,c))
    sl_mask = np.random.randint(low = 0,high = sl+1,size = (h,w))
    steps = np.max(sl_mask)
    sl_step_mask = np.zeros([steps+1,h,w]).astype(np.int32) 
    ex_mask = np.zeros([steps+1,h,w,c]).astype(np.int32)
    for i in range(steps):
        step = steps-i
        sl_step_mask[step] = (sl_mask/step).astype(np.int32)
        sl_step_mask_3d = expand_3d(sl_step_mask[step],channel=c) 
        ex_mask[step] = expansion_matrix_3d(sl_step_mask_3d * al_mask, step)
        sl_mask -= sl_step_mask[step]*step
    block_mask = np.sum(ex_mask,axis = 0) 
    if noise_type=='block':
        return block_mask
    elif noise_type=='mix':
        return point_mask + block_mask

def expand_3d(X,channel=3):
    assert len(X.shape) == 2
    X_tmp = np.expand_dims(X, axis = -1)
    X_3d = np.repeat(X_tmp, channel,axis = -1)
    return X_3d

def expansion_matrix_3d(X, steps=1):
    c = X.shape[2]
    arr_tmp = X.copy()
    arr_tmp = np.concatenate((arr_tmp,np.zeros([arr_tmp.shape[0], steps*2, c])),axis = 1)
    arr_tmp = np.concatenate((arr_tmp,np.zeros([steps*2, arr_tmp.shape[1], c])),axis = 0) 
    arr_tmp_down = np.zeros(arr_tmp.shape) 
    arr_tmp_right = np.zeros(arr_tmp.shape) 
    for i in range(steps*2):
        step = i + 1
        down_tl = [k for k in range(arr_tmp.shape[0]-step)]
        arr_down = np.concatenate((np.zeros((step, arr_tmp.shape[1], c)),arr_tmp[down_tl,:,:]),axis = 0)
        arr_tmp_down += arr_down
    arr_out_down = arr_tmp + arr_tmp_down
    for i in range(steps*2): 
        step = i + 1
        down_tl = [k for k in range(arr_tmp.shape[1]-step)]
        arr_right = np.concatenate((np.zeros((arr_tmp.shape[0], step, c)),arr_out_down[:,down_tl,:]),axis = 1)
        arr_tmp_right += arr_right
    arr_out_right = arr_out_down + arr_tmp_right
    out_mat = arr_out_right[steps : arr_out_right.shape[0] - steps, steps : arr_out_right.shape[1]-steps, :]
    return out_mat
