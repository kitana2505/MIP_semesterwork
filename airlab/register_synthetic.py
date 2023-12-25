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

import airlab as al
import torch
import pickle
import json
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from airlab_helper import register_airlab
import ipdb
from tqdm import tqdm

import cv2
num_iter = 2000
sc = 1

cm_red = plt.get_cmap('Reds')
cm_green = plt.get_cmap("Greens")

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/segreg_anh"
    dtype = torch.float32
    device = torch.device("cuda:0")

    increment_trans = np.linspace(-5, 5, 100)
    increment_rot = np.linspace(-60, 60, 100)

    # translate with synthetic data
    trans_loss = list()
    rot_loss = list()

    for trans_val in tqdm(increment_trans):
        x_trans = trans_val
        y_trans = trans_val 
        rot_angle = 0

        returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
        mse_trans_loss = register_airlab(returned_dict, num_iter=num_iter)
        trans_loss.append(mse_trans_loss)

    for rot_val in tqdm(increment_rot):
        x_trans = 0
        y_trans = 0
        rot_angle = rot_val

        returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
        mse_rot_loss = register_airlab(returned_dict, num_iter=num_iter)
        rot_loss.append(mse_rot_loss)    

    # Save trans_loss and rot_loss in a json file
    mse_synthetic = {
        "trans_loss": trans_loss,
        "rot_loss": rot_loss
    }

    with open("mse_synthetic.json", "w") as f:
        json.dump(mse_synthetic, f, indent = 4)

    
            


