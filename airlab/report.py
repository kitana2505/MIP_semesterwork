import cv2
import numpy as np
import ipdb
from tqdm import tqdm
import os
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work")
from helper import load_data
import airlab as al
import torch
import pickle
import json
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

x_trans = 10
y_trans = 0
rot_angle = 0
alpha = 0.5

cm_red = plt.get_cmap('Reds')
cm_green = plt.get_cmap("Greens")

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/segreg_anh"
    dtype = torch.float32
    device = torch.device("cuda:0")


    fixed_image_ori = al.read_image_as_tensor(os.path.join(data_root, "fix.png"), dtype=dtype, device=device)
    moving_image_ori = al.read_image_as_tensor(os.path.join(data_root, "moving.png"), dtype=dtype, device=device)
    img_move_with_kp = al.read_image_as_tensor(os.path.join(data_root, "img_move_with_kp.png"), dtype=dtype, device=device)

    # ipdb.set_trace()

    fixed_image, moving_image = al.utils.normalize_images(fixed_image_ori, moving_image_ori)

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    transformation = al.transformation.pairwise.SimilarityTransformation(moving_image, opt_cm=True)

    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(1000)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()

    _, img_move_with_kp = al.utils.normalize_images(moving_image, img_move_with_kp)
    
    warped_image = al.transformation.utils.warp_image(img_move_with_kp, displacement)
    img_transformed = warped_image.image.to('cpu').numpy()[0,0]

    img_fix_with_kp = cv2.imread(os.path.join(data_root, "img_fix_with_kp.png"), cv2.IMREAD_GRAYSCALE)
    img_fix_red_with_kp = cm_red(img_fix_with_kp)
    img_move_green_with_kp = cm_green(img_transformed)
    overlay_image = cv2.addWeighted(img_fix_red_with_kp, 1 - alpha, img_move_green_with_kp, alpha, 0)

    Image.fromarray((overlay_image[:, :, :3] * 255).astype(np.uint8)).save('overlay.png')
    ipdb.set_trace()


    # cv2.imwrite("out.png", img_transformed*255)

    # plt.figure(3)    
    # plt.imshow(img_fix,cmap='Reds')
    # plt.scatter(keypoint[:,0], keypoint[:,1], color='red', marker='.', s=5)  

    # plt.imread(os.path.join(data_root, "img_fix_red_with_kp.png"))
    # plt.imshow(img_transformed, cmap="Greens", alpha=0.5)

    # plt.scatter(gt_transformed[:,0], gt_transformed[:,1], color='green', marker='.', s=5,)  
    # plt.title("Shift x = %s, shift y = %s, rotation = %s" % (str(x_trans), str(y_trans), str(rot_angle)))
    # plt.savefig("overlay.png")
    
