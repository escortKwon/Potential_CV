import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Set Environment Path
cwd = os.getcwd()
img_path = cwd + '/Practice_CV/monoSLAM/Images/'

# Open image
img = cv2.imread(img_path + 'Sample_1.jpg', 0)
img2 = img

# Initiate STAR detector
orb = cv2.ORB_create()

# Find the keypoints with ORB
kp = orb.detect(img, None)

# Compute the descriptors with ORB
kp, des = orb.compute(img, kp)

img = cv2.drawKeypoints(img,kp,img)

# Draw only keypoints location, not size and orientation
img2 = cv2.drawKeypoints(img2, kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(img2)
plt.show()