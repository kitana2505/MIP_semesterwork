import numpy as np

import matplotlib.pyplot as plt
import pickle
import ipdb
import cv2
import time  
# import sklearn
# from sklearn.metrics import mean_squared_error
# from scipy.optimize import minimize

import matplotlib.image as mpimg
# import normal as normalscript

import cv2
from PIL import Image, ImageDraw
import sys
sys.path.append("/home.stud/quangthi/ws/registration_ophthalmological_sequences/segreg_anh/step1_edgebasedregistration")
import detect_keypoint

kernel_size_gauss = 11
kernel_size_soebel = 27
threshold = 0.2
distance = 50
num_of_sample = 31
# x_trans = 10
# y_trans = 0
# rot_angle = 2
sc = 1.0 # scale of the synthetic transformation

rec_size = 2
color = (200, 200, 200)
thickness = -1 
alpha = 0.5
cm_red = plt.get_cmap('Reds')
cm_green = plt.get_cmap("Greens")
#path = "/home.stud/quangthi/ws/semester_work/spam/bloodvessel.jpg"
#path = "/home.stud/quangthi/ws/semester_work/spam/fix.png"
# path = "/home.stud/quangthi/ws/data/mnist/mnist_five_fix.png"
path = "/home.stud/quangthi/ws/data/mnist_five/153.png"
RESIZE_HEIGHT = 128

def generate_image(x_trans,y_trans,rot_angle,sc, img_fix_path=path,rs=(128, 128),change_size=False):  
  
    img_fix= detect_keypoint.load_image(img_fix_path)
    
    height, width = img_fix.shape[:2]
    if change_size:

        # resize by the ratio of heigh / width
        # resize_width = int(RESIZE_HEIGHT * width / height / 10) * 10
        # inshape = (RESIZE_HEIGHT, resize_width)
        # new_size = (resize_width, RESIZE_HEIGHT)
    # create moving image
        img_fix = cv2.resize(img_fix, rs)
    
        img_moving = detect_keypoint.transform_img(img_fix,center=(rs[0] / 2, rs[1] / 2), angle=rot_angle, scale=1, tx=x_trans,ty=y_trans)

    else: 
        img_moving = detect_keypoint.transform_img(img_fix,center=(width / 2, height / 2), angle=rot_angle, scale=1, tx=x_trans,ty=y_trans)
    # find keypoints with threshold and distance
    img_blurred = detect_keypoint.gauss_blurring(img_fix,kernel_size_gauss) 
    gradient_x, gradient_y = detect_keypoint.img_gradient( img_blurred,kernel_size=kernel_size_soebel)
    # magnitude of the gradient
    gradient_norm = detect_keypoint.norm(gradient_x,gradient_y)
    (minVal_norm, maxVal_norm, minLoc_norm, maxLoc_norm, keypoint) = detect_keypoint.find_keypoint(gradient_norm,threshold=threshold,distance=distance)

     # groundtruth keypoints after synthetic transformation
    rot_matrix = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=rot_angle, scale = sc)
    row_ones = np.ones((1, len(keypoint)))
    gt_point = rot_matrix  @ np.vstack((keypoint.T, row_ones))

    gt_point[0, :]+=x_trans
    gt_point[1, :]+=y_trans

    gt_point = gt_point.T
    img_fix_with_kp = img_fix.copy()
    img_move_with_kp = img_moving.copy()

    # draw key_rectangle on both images
    '''for kp_fix, kp_move in zip(keypoint, gt_point):
        x_fix, y_fix = int(kp_fix[0]), int(kp_fix[1])
        top_left_fix = (x_fix - rec_size//2, y_fix - rec_size//2)
        bottom_right_fix = (x_fix + rec_size//2, y_fix + rec_size//2)

        x_move, y_move = int(kp_move[0]), int(kp_move[1])
        top_left_move = (x_move - rec_size//2, y_move - rec_size//2)
        bottom_right_move = (x_move + rec_size//2, y_move + rec_size//2)

        cv2.rectangle(img_fix_with_kp, top_left_fix, bottom_right_fix, color, thickness)
        cv2.rectangle(img_move_with_kp, top_left_move, bottom_right_move, color, thickness)'''

    # draw random lines on image    
    img_fix_with_kp = cv2.line(img_fix_with_kp, (50,50), (100,100), color=(255, 255, 255), thickness=2)
    img_move_with_kp=detect_keypoint.transform_img(img_fix_with_kp,center=(width / 2, height / 2), angle=rot_angle, scale=1, tx=x_trans,ty=y_trans)

    return_dict = dict()
    return_dict['img_fix'] = img_fix
    return_dict['keypoint'] = keypoint
    return_dict['img_moving'] = img_moving
    return_dict['gt'] = gt_point
    return_dict['img_fix_with_kp'] = img_fix_with_kp
    return_dict['img_move_with_kp'] = img_move_with_kp

    return return_dict

def plot_overlay(img_fix_with_kp,img_transformed, saved_path="out.png"):

    img_fix_red_with_kp = cm_red(img_fix_with_kp)
    img_move_green_with_kp = cm_green(img_transformed)
    # ipdb.set_trace()
    overlay_image = cv2.addWeighted(img_fix_red_with_kp, 1 - alpha, img_move_green_with_kp, alpha, 0)
    Image.fromarray((overlay_image[:, :, :3] * 255).astype(np.uint8)).save(saved_path)

    print(f"Image is saved at {saved_path}")

    return overlay_image

def save_info(x_trans,y_trans,rot_angle,sc,mse,method,option1,pathname=path):
    str_option = str(option1)
    with open(str(method)+'.txt', 'a') as f:
        f.write("Translation x: %d\nTranslation y: %d\nRotation: %d\nMSE: %d \nDisplacemnet: %s\nImage name: %s -------------------------\n" %(x_trans,y_trans,rot_angle,mse,str_option,pathname))
    return

if __name__ == "__main__":
    ipdb.set_trace()