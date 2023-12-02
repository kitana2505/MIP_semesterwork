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
#############################################
# Load snow data and create a deformed image
#############################################

################################################
# Here we will load the data,
# define a deformation function and apply it to the data
# in order to obtain a deformed data set.
#
# We will then visualise the difference between the images
# -- as explained in the :ref:`imageCorrelationTheory`.

# Load data
snow = spam.datasets.loadSnow()
img_fix =  detect_keypoint.load_image("./fix.png")

# height, width = img_fix.shape
# center = (int(width//2), int(height//2))
# img_moving = detect_keypoint.transform_img(img_fix, center, angle=0.0, scale=1, tx=10, ty=0)
# img_fix_3d = np.tile(img_fix[:,:,None], 100)
# img_moving_3d = np.tile(img_moving[:,:,None], 100)

# Define transformation to apply
# 't' : Z - Y - X
# 
transformation = {'t': [0.0, 8.0, 5.0],
                  'r': [5.0, 0.0, 0.0]}


# Convert this into a deformation function
Phi = spam.deformation.computePhi(transformation)

img_moving = spam.DIC.deform.applyPhiPython(img_fix, Phi=Phi)

# Apply this to snow data
# snowDeformed = spam.DIC.applyPhi(snow, Phi=Phi)
# img_moving_3d = spam.DIC.applyPhi(img_fix_3d, Phi=Phi)

# ipdb.set_trace()
# Show the difference between the initial and the deformed image.
# Here we used the blue-white-red colourmap "coolwarm" 
# which makes 0 white on the condition of the colourmap being symmetric around zero, 
# so we force the values with vmin and vmax.
# plt.figure()
#plt.imshow((snow - snowDeformed)[50], cmap='coolwarm', vmin=-36000, vmax=36000)
# plt.imshow((img_fix - img_moving)[50], cmap='coolwarm', vmin=-36000, vmax=36000)

################################################
# Perform correlation
################################################

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