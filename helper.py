import cv2
import numpy as np
from glob import glob
import os 
from tqdm import tqdm
import ipdb; 

def load_data(data_paths, resize, shuffle=False, seed=32):
  np.random.seed(seed)
  img_data = []
  # read, resize and append image
  for path in data_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, resize)
    img_data.append(img_resize/255.0)

  img_data = np.array(img_data)

  if shuffle:
    np.random.shuffle(img_data)
  
  print(f"Loaded {len(img_data)} samples")
  return img_data

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