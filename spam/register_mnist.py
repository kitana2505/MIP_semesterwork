import cv2
import glob
import numpy as np
import os
import ipdb
from tqdm import tqdm
import spam.deformation
import spam.DIC
import time
import json

num_iter = 300
alpha = 0.5
sc = 1


if __name__== "__main__":

    mnist_five_dir = "../../data/mnist_five"
    mnist_five_files = glob.glob(os.path.join(mnist_five_dir, "*.png"))

    mnist_five = []
    for file in mnist_five_files:
        img = cv2.imread(file, 0)
        mnist_five.append(img)
    
    mnist_five = np.array(mnist_five)

    fix_image = mnist_five[0]
    moving_images = mnist_five[1:]
    
    duration = []
    squared_error = []

    for moving_image in moving_images:
        start = time.perf_counter()

        phi = spam.DIC.register(moving_image, fix_image,
        margin=10,
        maxIterations=num_iter,
        deltaPhiMin=0.001,
        verbose=True,                 # Show updates on every iterationclearcl
    )

        # ipdb.set_trace()
        # stack fixed_image and img_transformed
        img_transformed = spam.DIC.deform.applyPhiPython(moving_image, phi['Phi'])

        img_stacked = np.hstack((fix_image, moving_image, img_transformed))
        cv2.imwrite("img_stacked.png", img_stacked*255)

        duration.append(time.perf_counter() - start)
        loss = (img_transformed - fix_image)**2
        squared_error.append(np.mean(loss))


    fps = len(duration) / sum(duration)
    mse = sum(squared_error) / len(squared_error)
    print(f"Mean square error: {mse}")
    print(f"Frame per second: {fps}")

    saved_dict = {"frame_loss" : squared_error, "mse": mse, "fps": fps}

    saved_dict["frame_loss"] = [float(x) for x in saved_dict["frame_loss"]]
    saved_dict["mse"] = float(saved_dict["mse"])
    saved_dict["fps"] = float(saved_dict["fps"])
    with open("mse_mnist.json", 'w') as file:
        json.dump(saved_dict, file, indent=4)   