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
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work")
from helper import load_data, train_test_split

RESIZE_HEIGHT = 128

if __name__ == "__main__":


    val_data_root = "/home.stud/quangthi/ws/data/Sada_val"
    checkpoint_path = "/home.stud/quangthi/ws/semester_work/voxelmorp/voxel_morp.h5"
    
    
    val_data_paths = glob(os.path.join(val_data_root, "*/*.png"))   


    first_sample = cv2.imread(val_data_paths[0], cv2.IMREAD_GRAYSCALE)
    height, width = first_sample.shape[:]
    # resize by the ratio of heigh / width
    resize_width = int(RESIZE_HEIGHT * width / height / 10) * 10

    # build model using VxmDense
    inshape = (RESIZE_HEIGHT, resize_width)
    new_size = (resize_width, RESIZE_HEIGHT)

    # load data
    val_data = load_data(val_data_paths, resize=new_size)

    # configure unet features
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # load traied checkpoint
    vxm_model.load_weights(checkpoint_path)
    
    fix_frame = val_data[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_voxelmorp.avi', fourcc, 20.0, new_size)

    squared_error = list()
    duration = list()
    for move_frame in tqdm(val_data[1:]):
        start = time.perf_counter()
        data_input = [
            move_frame[np.newaxis, :, :, np.newaxis], 
            fix_frame[np.newaxis, :, :, np.newaxis]]
        moved_frame, flow = vxm_model.predict(data_input)
        duration.append(time.perf_counter() - start)

        loss = (moved_frame[0,:,:,0] - fix_frame)**2
        squared_error.append(np.mean(loss))

        video_frame = cv2.cvtColor(moved_frame[0,:,:,0], cv2.COLOR_GRAY2BGR) * 255        
        out.write(np.uint8(video_frame))
    
    fps = sum(duration) / len(duration)
    mse = sum(squared_error) / len(squared_error)
    print(f"Mean square error: {mse}")
    print(f"Frame per second: {fps}")

    saved_dict = {"frame_loss" : squared_error, "mse": mse, "fps": fps}

    with open("mse.json", 'w') as file:
        json.dump(saved_dict, file, indent=4)