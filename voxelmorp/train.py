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
from helper import load_data, train_test_split

def vxm_data_generator(x_data, batch_size=8):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/Sada_imgs"

    train_data = load_data(data_root)
    x_train, x_val = train_test_split(train_data, val_per=0.1)
    
    # build model using VxmDense
    inshape = x_train.shape[1:]
    # configure unet features
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

    # voxelmorph has a variety of custom loss classes
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

    # usually, we have to balance the two losses by a hyper-parameter
    lambda_param = 0.05
    loss_weights = [1, lambda_param]

    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

    train_generator = vxm_data_generator(x_train)

    checkpoint = ModelCheckpoint(
        'voxel_morp.h5',  # Filepath to save the model
        monitor='loss',  # Monitor validation loss
        save_best_only=True,  # Save only the best models
        mode='min',  # Save the model when the monitored quantity is minimized
        verbose=1  # 1: display messages, 0: silent
    )

    # Add the ModelCheckpoint callback to the list of callbacks
    callbacks_list = [checkpoint]


    nb_epochs = 100
    steps_per_epoch = 100
    hist = vxm_model.fit(train_generator, 
    epochs=nb_epochs, steps_per_epoch=steps_per_epoch, 
    callbacks=callbacks_list,
    verbose=2)