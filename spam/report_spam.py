import cv2
import numpy as np
import ipdb
from tqdm import tqdm
import os
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work/mutual")
from generate_synthetic_img import generate_image, plot_overlay

import pickle
import json
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import spam.deformation
import spam.DIC

num_iter = 300
x_trans = 10
y_trans = 0
rot_angle = 0
alpha = 0.5
sc = 1

cm_red = plt.get_cmap('Reds')
cm_green = plt.get_cmap("Greens")

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/segreg_anh"

    returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
    img_fix_with_kp = returned_dict['img_fix_with_kp']
    img_move_with_kp = returned_dict['img_move_with_kp']
    keypoint = returned_dict['keypoint']
    gt_point_load = returned_dict['gt']
    img_fix =  returned_dict['img_fix']
    img_moving =  returned_dict['img_moving']

    height, width = img_fix.shape[:2]

    phi = spam.DIC.register(img_moving, img_fix,
                  margin=10,
                  maxIterations=num_iter,
                  deltaPhiMin=0.001,
                  verbose=True,                 # Show updates on every iterationclearcl
    )

    img_transformed = spam.DIC.deform.applyPhiPython(img_move_with_kp, phi['Phi'])
    # phi2d = phi['Phi'][1:, 1:]

    # ipdb.set_trace()
    overlay = plot_overlay(img_fix_with_kp, img_transformed/255.0, saved_path="overlay_spam.png")





