import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm
import voxelmorph as vxm
import neurite as ne
from keras.callbacks import ModelCheckpoint
import datetime
import ipdb
import json
import time
import ipdb
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work/mutual")
from generate_synthetic_img import generate_image, plot_overlay, save_info


num_iter = 5000
x_trans = 10
y_trans = 0
rot_angle = 10
alpha = 0.5
sc = 1

cm_red = plt.get_cmap('Reds')
cm_green = plt.get_cmap("Greens")
RESIZE_HEIGHT = 32
new_size = (32, 32)

# load data
returned_dict = generate_image(x_trans, y_trans, rot_angle, sc,rs=new_size,change_size=True)

img_fix_with_kp = returned_dict['img_fix_with_kp']
img_move_with_kp = returned_dict['img_move_with_kp']
keypoint = returned_dict['keypoint']
gt_point_load = returned_dict['gt']
img_fix =  returned_dict['img_fix']
img_moving =  returned_dict['img_moving']


# load model
# checkpoint_path = "/home.stud/quangthi/ws/semester_work/voxelmorp/backup/voxel_morp_40e.h5"
checkpoint_path = "/home.stud/quangthi/ws/semester_work/voxelmorp/voxelmorp_mnist_five.h5"
height, width = img_fix.shape[:]

# resize by the ratio of heigh / width
resize_width = int(RESIZE_HEIGHT * width / height / 10) * 10

# build model using VxmDense
# inshape = (RESIZE_HEIGHT, resize_width)
inshape = (32, 32)
l = (resize_width, RESIZE_HEIGHT)

# configure unet features
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

# load traied checkpoint
vxm_model.load_weights(checkpoint_path)

# normalize
img_moving = img_moving /  255.0
img_fix = img_fix / 255.0

data_input = [
    img_moving[np.newaxis, :, :, np.newaxis], 
    img_fix[np.newaxis, :, :, np.newaxis]]

img_transformed_raw, flow = vxm_model.predict(data_input)
img_transformed = img_transformed_raw[0, :, :, 0]

img_fix = cv2.resize(img_fix, (300, 300))
img_moving = cv2.resize(img_moving, (300, 300))
img_transformed = cv2.resize(img_transformed, (300, 300))

overlay_ori = plot_overlay(img_fix, img_moving, saved_path="overlay_ori.png")

cv2.imwrite("fix_img.png", np.array(img_fix * 255).astype(np.uint8))
# cv2.imwrite("deformation.png", np.array(flow[0,:,:,0] * 255).astype(np.uint8))
cv2.imwrite("transformed.png", np.array(img_transformed * 255).astype(np.uint8))
saved_path = "./overlay_voxelmorph.png"
plot_overlay(img_fix,img_transformed, saved_path)
