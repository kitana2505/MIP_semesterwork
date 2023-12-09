import spam.deformation
import spam.DIC
import spam.datasets
import cv2
import numpy as np
import ipdb
from tqdm import tqdm
import os
from glob import glob
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work")
from helper import load_data
import json
import time 

RESIZE_HEIGHT = 128

if __name__ == "__main__":
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
    val_data = load_data(val_data_paths, resize=new_size)

    fix_frame = val_data[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_spam.avi', fourcc, 20.0, new_size)

    squared_error = list()
    duration = list()
    for move_frame in tqdm(val_data[1:]):
        start = time.perf_counter()
        
        Phi = spam.DIC.register(
                move_frame, fix_frame,
                margin=10,
                maxIterations=300,
                deltaPhiMin=0.001,
                verbose=True
            )
        
        moved_frame = spam.DIC.deform.applyPhiPython(move_frame, Phi=Phi['Phi'])
        duration.append(time.perf_counter() - start)
        
        loss = (moved_frame - fix_frame)**2
        squared_error.append(np.mean(loss))

        video_frame = cv2.cvtColor(moved_frame, cv2.COLOR_GRAY2BGR) * 255        
        out.write(np.uint8(video_frame))
    
    fps = sum(duration) / len(duration)
    mse = sum(squared_error) / len(squared_error)
    print(f"Mean square error: {mse}")
    print(f"Frame per second: {fps}")

    saved_dict = {"frame_loss" : squared_error, "mse": mse, "fps": fps}

    with open("mse.json", 'w') as file:
        json.dump(saved_dict, file, indent=4)
    
    