import cv2
import numpy as np
import ipdb
from tqdm import tqdm
import os
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
import spam.deformation
import spam.DIC
import ipdb

import cv2
num_iter = 300
sc = 1

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/segreg_anh"
    # define translation and rotation parameter
    increment_trans = np.linspace(-5, 5, 100)
    increment_rot = np.linspace(-60, 60, 100)

    # translate with synthetic data
    trans_loss = list()
    rot_loss = list()

    for trans_val in tqdm(increment_trans):
        # translate in x- and y-direction same, no rotation
        x_trans = trans_val
        y_trans = trans_val 
        rot_angle = 0

        returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
        img_fix =  returned_dict['img_fix']
        img_moving = returned_dict['img_moving']

        phi = spam.DIC.register(img_moving, img_fix,
                    margin=10,
                    maxIterations=num_iter,
                    deltaPhiMin=0.001,
                    verbose=True,                 # Show updates on every iterationclearcl
        )   
        
        
        img_transformed = spam.DIC.deform.applyPhiPython(img_moving, phi['Phi'])

        loss = (img_fix/255.0 - img_transformed/255.0)**2
        trans_loss.append(np.mean(loss))

    for rot_val in tqdm(increment_rot):
        # only rotation, no translation
        x_trans = 0
        y_trans = 0
        rot_angle = rot_val

        returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
        img_fix =  returned_dict['img_fix']
        img_moving = returned_dict['img_moving']

        phi = spam.DIC.register(img_moving, img_fix,
                    margin=10,
                    maxIterations=num_iter,
                    deltaPhiMin=0.001,
                    verbose=True,                 # Show updates on every iterationclearcl
        )   
        
        
        img_transformed = spam.DIC.deform.applyPhiPython(img_moving, phi['Phi'])

        loss = (img_fix/255.0 - img_transformed/255.0)**2
        rot_loss.append(np.mean(loss))

    # Save trans_loss and rot_loss in a json file
    mse_synthetic = {
        "trans_loss": trans_loss,
        "rot_loss": rot_loss
    }

    with open("mse_synthetic.json", "w") as f:
        json.dump(mse_synthetic, f, indent = 4)

    
            


