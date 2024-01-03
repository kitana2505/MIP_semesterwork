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
import time
import ipdb
import glob
import json

if __name__== "__main__":
    # load MNIST data
    mnist_five_dir = "../../data/mnist_five"
    mnist_five_files = glob.glob(os.path.join(mnist_five_dir, "*.png"))

    mnist_five = []
    for file in mnist_five_files:
        img = cv2.imread(file, 0)
        mnist_five.append(img)
    
    mnist_five = np.array(mnist_five)

    fix_image = mnist_five[0]
    fix_image = np.pad(fix_image, ((2, 2), (2, 2)), mode='constant') / 255.0  # Pad with zeros to have shape (32, 32)
    moving_images = mnist_five[1:]
    #define input shape and number of features
    inshape = (32, 32) 
    
    # configure unet features
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    # build model
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # load trained checkpoint
    checkpoint_path = "/home.stud/quangthi/ws/semester_work/voxelmorp/voxelmorp_mnist_five.h5"
    vxm_model.load_weights(checkpoint_path)    

    stacked_size = (inshape[0], inshape[1]*3)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('mninst_voxelmorp.avi', fourcc, 20.0, stacked_size)
    
    squared_error = list()
    duration = list()
    
    for moving_image in tqdm(moving_images):
        start = time.perf_counter()
        moving_image = np.pad(moving_image, ((2, 2), (2, 2)), mode='constant') / 255.0

        data_input = [
            moving_image[np.newaxis, :, :, np.newaxis], 
            fix_image[np.newaxis, :, :, np.newaxis]]
        
        img_transformed_raw, flow = vxm_model.predict(data_input)
        img_transformed = img_transformed_raw[0, :, :, 0]
        duration.append(time.perf_counter() - start)

        loss = (fix_image - img_transformed)**2
        squared_error.append(np.mean(loss))
        img_stacked = np.hstack((fix_image, moving_image, img_transformed))
        # cv2.imwrite("img_stacked.png", img_stacked*255)
        out.write(np.uint8(img_stacked*255))


    fps = len(duration) / sum(duration)
    mse = sum(squared_error) / len(squared_error)
    print(f"Mean square error: {mse}")
    print(f"Frame per second: {fps}")
    # save loss and fps to json file
    saved_dict = {"frame_loss" : squared_error, "mse": mse, "fps": fps}

    with open("mse_mnist.json", 'w') as file:
        json.dump(saved_dict, file, indent=4)