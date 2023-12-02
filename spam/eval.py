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
    for move_frame in tqdm(val_data[1:]):
        Phi = spam.DIC.register(
                move_frame, fix_frame,
                margin=10,
                maxIterations=300,
                deltaPhiMin=0.001,
                verbose=True
            )
        
        moved_frame = spam.DIC.deform.applyPhiPython(move_frame, Phi=Phi)

        loss = (moved_frame - fix_frame)**2
        squared_error.append(np.sum(loss))

        video_frame = cv2.cvtColor(moved_frame, cv2.COLOR_GRAY2BGR) * 255        
        out.write(np.uint8(video_frame))
    
    print(f"Mean square error: {sum(squared_error) / len(squared_error)}")
    