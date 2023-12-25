import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from tqdm import tqdm
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work/mutual")
from generate_synthetic_img import generate_image
from generate_synthetic_img import plot_overlay
from generate_synthetic_img import save_info

import pickle
import json
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
from keras.callbacks import ModelCheckpoint
import ipdb

import cv2
num_iter = 300
sc = 1

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/segreg_anh"

    increment_trans = np.linspace(-5, 5, 100)
    increment_rot = np.linspace(-60, 60, 100)

    # translate with synthetic data
    trans_loss = list()
    rot_loss = list()
    
    inshape = (32, 32)     
    # configure unet features
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)


    checkpoint_path = "/home.stud/quangthi/ws/semester_work/voxelmorp/voxelmorp_mnist_five.h5"
    vxm_model.load_weights(checkpoint_path)   

    for trans_val in tqdm(increment_trans):
        x_trans = trans_val
        y_trans = trans_val 
        rot_angle = 0

        returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
        img_fix =  returned_dict['img_fix']
        img_moving = returned_dict['img_moving']

        fix_image = np.pad(img_fix, ((2, 2), (2, 2)), mode='constant') / 255.0  # Pad with zeros to have shape (32, 32)
        moving_image = np.pad(img_moving, ((2, 2), (2, 2)), mode='constant') / 255.0


        data_input = [
            moving_image[np.newaxis, :, :, np.newaxis], 
            fix_image[np.newaxis, :, :, np.newaxis]]

        img_transformed_raw, flow = vxm_model.predict(data_input)
        img_transformed = img_transformed_raw[0, :, :, 0]

        # mse_trans_loss = register_airlab(returned_dict, num_iter=num_iter)

        loss = (fix_image - img_transformed)**2
        trans_loss.append(np.mean(loss))

    
    for rot_val in tqdm(increment_rot):
        x_trans = 0
        y_trans = 0
        rot_angle = rot_val

        returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
        img_fix =  returned_dict['img_fix']
        img_moving = returned_dict['img_moving']

        fix_image = np.pad(img_fix, ((2, 2), (2, 2)), mode='constant') / 255.0  # Pad with zeros to have shape (32, 32)
        moving_image = np.pad(img_moving, ((2, 2), (2, 2)), mode='constant') / 255.0

        data_input = [
            moving_image[np.newaxis, :, :, np.newaxis], 
            fix_image[np.newaxis, :, :, np.newaxis]]

        img_transformed_raw, flow = vxm_model.predict(data_input)
        img_transformed = img_transformed_raw[0, :, :, 0]

        loss = (fix_image - img_transformed)**2
        rot_loss.append(np.mean(loss))

    # Save trans_loss and rot_loss in a json file
    mse_synthetic = {
        "trans_loss": trans_loss,
        "rot_loss": rot_loss
    }

    with open("mse_synthetic.json", "w") as f:
        json.dump(mse_synthetic, f, indent = 4)

    
            


