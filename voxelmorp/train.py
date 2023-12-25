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
import sys
sys.path.append("/home.stud/quangthi/ws/semester_work")
from helper import load_data, train_test_split

RESIZE_HEIGHT = 128

def vxm_data_generator(data_paths, resize, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    data_paths = np.array(data_paths)

    # preliminary sizing
    ndims = len(resize)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, resize[1], resize[0], ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)

        moving_images = load_data(data_paths[idx1], resize)[:,:,:, np.newaxis]

        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = load_data(data_paths[idx2], resize)[:,:,:, np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]
        yield (inputs, outputs)

if __name__=="__main__":
    data_root = "/home.stud/quangthi/ws/data/Sada_imgs"

    # train_data = load_data(data_root)
    # x_train, x_val = train_test_split(train_data, val_per=0.1)
    
    data_paths = glob(os.path.join(data_root, "*/*.png"))   
    x_data = cv2.imread(data_paths[0], cv2.IMREAD_GRAYSCALE)

    height, width = x_data.shape[:]
    # resize by the ratio of heigh / width
    resize_width = int(RESIZE_HEIGHT * width / height / 10) * 10

    # build model using VxmDense
    inshape = (RESIZE_HEIGHT, resize_width)
    new_size = (resize_width, RESIZE_HEIGHT)
    # configure unet features
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    # print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

    # voxelmorph has a variety of custom loss classes
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

    # usually, we have to balance the two losses by a hyper-parameter
    lambda_param = 0.05
    loss_weights = [1, lambda_param]
    # total_lost = loss_mse + lambda_param * loss_l2
    
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

    batch_size = 64
    train_generator = vxm_data_generator(data_paths, resize=new_size, batch_size=batch_size)

    checkpoint = ModelCheckpoint(
        'voxel_morp.h5',  # Filepath to save the model
        monitor='loss',  # Monitor training loss
        save_best_only=True,  # Save only the best models
        mode='min',  # Save the model when the monitored quantity is minimized
        verbose=1  # 1: display messages, 0: silent
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Add the ModelCheckpoint callback to the list of callbacks
    callbacks_list = [checkpoint, tensorboard_callback]

    nb_epochs = 100
    steps_per_epoch = len(data_paths) // batch_size

    print("[INFO] Start training....")
    hist = vxm_model.fit(train_generator, 
    epochs=nb_epochs, steps_per_epoch=steps_per_epoch, 
    callbacks=callbacks_list,
    verbose=1)