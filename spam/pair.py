"""
Image correlation basics
=========================

Here we synthetically apply a rigid-body transformation to an image
and try to measure this transformation using the ``register`` image correlation
function.
"""

######################
# Import modules
######################

import matplotlib.pyplot as plt
import spam.deformation
import spam.DIC
import spam.datasets
import cv2
import numpy as np
import ipdb
import sys
sys.path.append("/home.stud/quangthi/ws/registration_ophthalmological_sequences/segreg_anh/step1_edgebasedregistration")
import detect_keypoint
from detect_keypoint import transform_img
import pickle
#############################################

x_trans = 10
y_trans = 0
rot_angle = 0
sc = 1.0 # scale of the synthetic transformation

# Load data
img_fix =  detect_keypoint.load_image("./fix.png")
height, width = img_fix.shape[:2]

transformation = {'t': [0.0, y_trans, x_trans],
                  'r': [0.0, 0.0, 0.0]}


# Convert this into a deformation function
Phi = spam.deformation.computePhi(transformation)

img_moving = spam.DIC.deform.applyPhiPython(img_fix, Phi=Phi)
ipdb.set_trace()
with open("./fix_samples.pkl", "rb") as file:
        fix_samples = pickle.load(file)
c = fix_samples.shape[2]
fixed_keypoints = fix_samples[:, :, c//2] 

#gt_point = rot_matrix  @ np.vstack((keypoint.T, row_ones))
gt_point = fixed_keypoints
gt_point[:, 0]+=x_trans
gt_point[:, 1]+=y_trans

# Define transformation to apply
# 't' : Z - Y - X
# 



# Now we will use the image correlation function to try
# to measure the Phi between `snow` and `snowDeformed`.
'''spam.DIC.register(snowDeformed, snow,
                  margin=10,
                  maxIterations=50,
                  deltaPhiMin=0.001,
                  verbose=True,                 # Show updates on every iteration
                  imShowProgress=True,           # Show horizontal slice
                  imShowProgressNewFig=True) '''   
# New figure at every iteration

phi = spam.DIC.register(img_moving, img_fix,
                  margin=10,
                  maxIterations=300,
                  deltaPhiMin=0.001,
                  verbose=True,                 # Show updates on every iterationclearcl
)
# plt.show()

ipdb.set_trace()