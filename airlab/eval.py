import cv2
import numpy as np
import ipdb
from tqdm import tqdm
import os
from glob import glob
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work")
from helper import load_data
import airlab as al
import torch
import pickle
import json
import time

RESIZE_HEIGHT = 128

if __name__ == "__main__":

    dtype = torch.float32
    device = torch.device("cuda:0")
    iterations = 500

    val_data_root = "/home.stud/quangthi/ws/data/Sada_val"
   
    val_data_paths = glob(os.path.join(val_data_root, "*/*.png"))   

    first_sample = cv2.imread(val_data_paths[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_sample.shape[:]
    # resize by the ratio of heigh / width
    resize_width = int(RESIZE_HEIGHT * width / height / 10) * 10

    # build model using VxmDense
    inshape = (RESIZE_HEIGHT, resize_width)
    new_size = (resize_width, RESIZE_HEIGHT)

    # load data
    # val_data = load_data(val_data_paths, resize=new_size)
    val_data = load_data(val_data_paths)
    fix_frame = val_data[0]
    frame_height, frame_width = fix_frame.shape[:2]
    frame_size = tuple(map(int, [frame_width, frame_height]))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_airlab.avi', fourcc, 20.0, frame_size)

    fix_frame_al = al.image_from_numpy(fix_frame, [1, 1], [0, 0], dtype=dtype, device=device)
    
    # create pairwise registration object
    registration = al.PairwiseRegistration()

    squared_error = list()
    duration = []
    for move_frame in tqdm(val_data[1:]):
        start = time.perf_counter()
        move_frame_al = al.image_from_numpy(move_frame, [1, 1], [0, 0], dtype=dtype, device=device)

        fixed_frame, moving_frame = al.utils.normalize_images(fix_frame_al, move_frame_al)

        fixed_frame.image = 1 - fixed_frame.image
        moving_frame.image = 1 - moving_frame.image

        # choose the affine transformation model
        transformation = al.transformation.pairwise.SimilarityTransformation(moving_frame, opt_cm=True)
        # initialize the translation with the center of mass of the fixed image
        transformation.init_translation(fixed_frame)

        registration.set_transformation(transformation)
        # choose the Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE(fixed_frame, moving_frame)

        registration.set_image_loss([image_loss])

        # choose the Adam optimizer to minimize the objective
        optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(iterations)

        # start the registration
        registration.start()

        # set the intensities back to the original for the visualisation
        fixed_frame.image = 1 - fixed_frame.image
        moving_frame.image = 1 - moving_frame.image

        # warp the moving image with the final transformation result
        displacement = transformation.get_displacement()
        moved_frame = al.transformation.utils.warp_image(moving_frame, displacement)
        duration.append(time.perf_counter() - start)

        moved_frame = moved_frame.numpy()

        loss = (moved_frame - fix_frame)**2
        squared_error.append(np.mean(loss))

        # ipdb.set_trace()
        video_frame = cv2.cvtColor(moved_frame, cv2.COLOR_GRAY2BGR) * 255        
        out.write(np.uint8(video_frame))
    

    fps = sum(duration) / len(duration)
    mse = sum(squared_error) / len(squared_error)
    print(f"Mean square error: {mse}")
    print(f"Frame per second: {fps}")

    saved_dict = {"frame_loss" : squared_error, "mse": mse, "fps": fps}

    with open("mse.json", 'w') as file:
        json.dump(saved_dict, file, indent=4)
    