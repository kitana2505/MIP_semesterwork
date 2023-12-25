import cv2
import glob
import numpy as np
import os
import ipdb
from tqdm import tqdm
import torch
import time
import airlab as al
import json

num_iter = 1000
alpha = 0.5
sc = 1


if __name__== "__main__":

    dtype = torch.float32
    device = torch.device("cuda:0")

    mnist_five_dir = "../../data/mnist_five"
    mnist_five_files = glob.glob(os.path.join(mnist_five_dir, "*.png"))

    mnist_five = []
    for file in mnist_five_files:
        img = cv2.imread(file, 0)
        mnist_five.append(img)
    
    mnist_five = np.array(mnist_five)

    fix_image = mnist_five[0]
    moving_images = mnist_five[1:]
    
    fixed_image_ori = al.image_from_numpy(fix_image, [1, 1], [0, 0], dtype=dtype, device=device)

    squared_error = list()
    duration = list()
    
    inshape = fix_image.shape
    stacked_size = (inshape[0], inshape[1]*3)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('mninst_airlab.avi', fourcc, 20.0, stacked_size)

    for moving_image in tqdm(moving_images):        
        start = time.perf_counter()
        moving_image_ori = al.image_from_numpy(moving_image, [1, 1], [0, 0], dtype=dtype, device=device)

        fixed_image, moving_image = al.utils.normalize_images(fixed_image_ori, moving_image_ori)

        # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
        # calculate the center of mass of the object
        fixed_image.image = 1 - fixed_image.image
        moving_image.image = 1 - moving_image.image

        # create pairwise registration object
        registration = al.PairwiseRegistration()

        # transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)

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
        registration.set_number_of_iterations(num_iter)

        # start the registration
        registration.start()

        # set the intensities back to the original for the visualisation
        fixed_image.image = 1 - fixed_image.image
        moving_image.image = 1 - moving_image.image

        # warp the moving image with the final transformation result
        displacement = transformation.get_displacement()
        #ipdb.set_trace()

        warped_image = al.transformation.utils.warp_image(moving_image, displacement)
        img_transformed = warped_image.image.to('cpu').numpy()[0,0]
        duration.append(time.perf_counter() - start)
        loss = (fixed_image.image.to('cpu').numpy()[0,0] - img_transformed)**2
        print(np.mean(loss))

        squared_error.append(np.mean(loss))

        
        # ipdb.set_trace()
        # stack fixed_image and img_transformed
        img_stacked = np.hstack((fix_image, moving_image.image.to('cpu').numpy()[0,0], img_transformed))
        out.write(np.uint8(img_stacked*255))

        # cv2.imwrite("img_stacked.png", img_stacked*255)

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