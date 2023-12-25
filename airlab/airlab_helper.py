import airlab as al
import torch
import numpy as np 
import cv2


def register_airlab(returned_dict, dtype=torch.float32, device=torch.device("cuda:0"), num_iter=5000):
        #overlay_ori = plot_overlay(returned_dict['img_fix_with_kp'], returned_dict['img_move_with_kp'], saved_path="overlay_ori.png")
        img_fix = returned_dict['img_fix']
        img_moving = returned_dict['img_moving']
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

        warped_image = al.transformation.utils.warp_image(moving_image, displacement)
        #warped_image = al.transformation.utils.warp_image(img_move_with_kp, displacement)
        img_transformed = warped_image.image.to('cpu').numpy()[0,0]

        loss = (img_transformed - img_fix/255.0) ** 2
        return np.mean(loss)