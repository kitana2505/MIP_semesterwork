import cv2
import numpy as np
import pickle 

def displacement_to_transform(displacement_map):
    # Get the height and width of the displacement map
    dx = displacement_map[:, :, 0]
    dy = displacement_map[:, :, 1]

    # Define the corresponding points for the transformation
    src_pts = np.array([[0, 0], [0, displacement_map.shape[0]-1], [displacement_map.shape[1]-1, 0]], dtype=np.float32)
    dst_pts = np.array([[0+dx[0, 0], 0+dy[0, 0]], [0+dx[-1, 0], (displacement_map.shape[0]-1)+dy[-1, 0]], [(displacement_map.shape[1]-1)+dx[0, -1], 0+dy[0, -1]]], dtype=np.float32)
                       
    # import ipdb; ipdb.set_trace()

    # Calculate the 2D affine transformation matrix
    transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)

    return transform_matrix

with open("./displacement.pkl", "rb") as file:
    displacement_map = pickle.load(file)

# Example usage

# Get the transformation matrix
transform_matrix = displacement_to_transform(displacement_map)

import ipdb; ipdb.set_trace()