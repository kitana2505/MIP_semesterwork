import cv2
import numpy as np
from glob import glob
import os 
from tqdm import tqdm
import ipdb; 

def load_data(img_directory, resize_height=128):
  train_data = []

  # Get paths of all images
#   data_path = glob(os.path.join(img_directory, "*/*.png"))   

  data_path = glob(os.path.join(img_directory, "Gacr_01_019_01_560_n/*.png"))   

  # Read the first image to get height and width
  _frame = cv2.imread(data_path[0], cv2.IMREAD_GRAYSCALE)
  height, width = _frame.shape[:]

  # resize by the ratio of heigh / width
  resize_width = int(resize_height * width / height / 10) * 10

  # read, resize and append image
  for path in tqdm(data_path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (resize_width, resize_height))
    train_data.append(img_resize)

  return np.array(train_data)

def train_test_split(dataset, val_per=0.05, seed=32):
  np.random.seed(seed)
  num_train = int(len(dataset) * (1-val_per))
  x_train = dataset[:num_train] / 255.0
  x_val = dataset[num_train:] / 255.0

  np.random.shuffle(x_train)
  np.random.shuffle(x_val)

  print(f"Training data: {len(x_train)} samples")
  print(f"Validation data: {len(x_val)} samples")

  return x_train, x_val