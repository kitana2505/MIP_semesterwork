# this function is used to create the overlay image of the simpler vessels

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

num_iter = 2000
x_trans = 10
y_trans = 5
rot_angle = 1
alpha = 0.5
sc = 1

cm_red = plt.get_cmap('Reds')
cm_green = plt.get_cmap("Greens")
RESIZE_HEIGHT = 128

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/segreg_anh"
    dtype = torch.float32
    device = torch.device("cuda:0")

    returned_dict = generate_image(x_trans, y_trans, rot_angle, sc)
    #overlay_ori = plot_overlay(returned_dict['img_fix_with_kp'], returned_dict['img_move_with_kp'], saved_path="overlay_ori.png")
    overlay_ori = plot_overlay(returned_dict['img_fix'], returned_dict['img_moving'], saved_path="overlay_ori.png")
    img_fix = returned_dict['img_fix']
    img_fix_with_kp = returned_dict['img_fix_with_kp']
    img_move_with_kp = al.image_from_numpy(returned_dict['img_move_with_kp'], [1, 1], [0, 0], dtype=dtype, device=device)
  
    fixed_image_ori = al.image_from_numpy(returned_dict['img_fix'], [1, 1], [0, 0], dtype=dtype, device=device)

    moving_image_ori = al.image_from_numpy(returned_dict['img_moving'], [1, 1], [0, 0], dtype=dtype, device=device)


    fixed_image, moving_image = al.utils.normalize_images(fixed_image_ori, moving_image_ori)

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    #transformation = al.transformation.pairwise.SimilarityTransformation(moving_image, opt_cm=True)
    transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)
    #move_temp = transformation.set_parameters([x_trans,y_trans],[np.deg2rad(rot_angle)])
    #ipdb.set_trace()
    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(num_iter)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    #ipdb.set_trace()

    _, img_move_with_kp = al.utils.normalize_images(moving_image, img_move_with_kp)
    # warped_image = al.transformation.utils.warp_image(img_move_with_kp, displacement)


    warped_image = al.transformation.utils.warp_image(moving_image, displacement)
 
    img_transformed = warped_image.image.to('cpu').numpy()[0,0]

    saved_path = "./overlay_airlab.png"
 
    plot_overlay(img_fix,img_transformed*255, saved_path)



    print(f"Image is saved at {saved_path}")
    dist = al.transformation.utils.unit_displacement_to_displacement(displacement)
    print('Displacement: ',np.mean(np.mean(np.array(dist.cpu()),0),0))
    dist_temp = np.mean(np.mean(np.array(dist.cpu()),0),0)
    save_info(x_trans,y_trans,rot_angle,sc,registration.loss,"airlab",dist_temp)
    print("MSE = ", registration.loss)

    
