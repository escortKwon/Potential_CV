# Params_VO.py

import numpy as np
import cv2
import os
import platform

# Set Paramters
## Environment Path
cwd = os.getcwd()
path_result = cwd + '/Computer_Vision/monoSLAM/VO_Tsukuba/'
Result_image_name = path_result + f'Result_monoSLAM_Tsukuba_{platform.system()}.png'
Result_video_name = path_result + f'Result_Video_monoSLAM_Tsukuba_{platform.system()}.mp4'
## Load datasets
path_imgs_datasets = '/mnt/d/NewTsukubaStereoDataset/illumination/daylight/left/'
path_imgs_seqs = '/mnt/d/NewTsukubaStereoDataset/illumination/daylight/left/tsukuba_daylight_L_'
path_ground_truth = '/mnt/d/NewTsukubaStereoDataset/groundtruth/camera_track.txt'
## VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'X264')
fps = 60
## MAX_FRAME
list_files = os.listdir(path_imgs_datasets)
MAX_FRAME = len(list_files)
## Camera Intrinsic Paramters
focal_length = 615 # specified in units of pixels
principal_point = (320, 240) # in pixels (x, y)
# End of setting parameters

# Visualization
## Text
text_title_f = "ORB Coordinates"
text_title_gt = "Ground Truth Coordinates"
list_text_header = ['X = ', 'Y = ', 'Z = ']
## Window Size of Trajectory
trajSize = (1000, 1000)
## Text Position
### Features
textOrg_title_f = (10, 10)
textOrg_x_f = (10, 60)
textOrg_y_f = (10, 110)
textOrg_z_f = (10, 160)
list_text_Org_f = [textOrg_x_f, textOrg_y_f, textOrg_z_f]
### Ground Truths
textOrg_title_gt = (10, 260)
textOrg_x_gt = (10, 310)
textOrg_y_gt = (10, 360)
textOrg_z_gt = (10, 410)
list_text_Org_gt = [textOrg_x_gt, textOrg_y_gt, textOrg_z_gt]
## Window Position
Pos_trajectory = (2000, 500)
## Scale
trajectoryScale = 1
featureScale = 10
# End of setting visualization

# Emptty Variables
## Create empty numpy arrays for extarct t_f
t_f_extract = np.zeros(shape=(1, 3))
t_f_new = np.zeros(shape=(1, 3))