# Params.py
#
import numpy as np
import cv2
import os
import platform

# Set Paramters
## Environment Path
cwd = os.getcwd()
path_result = cwd + '/Computer_Vision/monoSLAM/Results/'
Result_image_name = path_result + f'Result_monoSLAM_Corridor_{platform.system()}.png'
Result_video_name = path_result + f'Result_Video_monoSLAM_Corridor_{platform.system()}.mp4'
## The number of Sequence
# seq_num = input(">>> Enter the number of Sequence (Ex: 01, 02 ...) : ")
seq_num = "03"
list_seq_corridor = ["01", "02", "03", "04"]
list_seq_campus = ["05"]
if platform.system() == "Linux":
    if seq_num in list_seq_corridor:
        path_seq_imgs = f'/mnt/d/Corridor/Frames/Sequence{seq_num}/'
        path_calib_img = '/mnt/d/Corridor/Images/Checkerboard_Calibration_1.jpg'
    elif seq_num in list_seq_campus:
        path_seq_imgs = f'/mnt/d/Campus/Frames/Sequence{seq_num}/'
        path_calib_img = '/mnt/d/Campus/Images/Checkerboard_Calibration_2.jpg'
elif platform.system() == "Windows":
    if seq_num in list_seq_corridor:
        path_seq_imgs = f'D:/Corridor/Frames/Sequence{seq_num}'
        path_calib_img = 'D:/Corridor/Images/Checkerboard_Calibration_1.jpg'
    elif seq_num in list_seq_campus:
        path_seq_imgs = f'D:/Campus/Frames/Sequence{seq_num}'
        path_calib_img = 'D:/Corridor/Images/Checkerboard_Calibration_2.jpg'
else: # Darwin / Undetermined
    pass
## VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'X264')
fps = 60
## MAX_FRAME
list_files = os.listdir(path_seq_imgs)
MAX_FRAME = len(list_files)
# Size of Sensor (Camera Intrinsic Parameters)
# camera_type = input(">>> Enter the type of camera (Ex: S20P...): ")
camera_type = "S20P"
if camera_type == "S20P":
    sizeSensor = 1/1.76
    sizePixel = 1.8 * 10 ** (-6)
else:
    pass
# End of setting parameters

# Visualization
## Text
text_title = "ORB Coordinates"
list_text_header = ['X = ', 'Y = ', 'Z = ']
## Window Size of Trajectory
trajSize = (1000, 1000)
## Text Position
textOrg_title = (10, 10)
textOrg_x = (10, 60)
textOrg_y = (10, 110)
textOrg_z = (10, 160)
list_text_Org = [textOrg_x, textOrg_y, textOrg_z]
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

# Distance
## Length
len_per_tile = 1
num_tiles = 57
distance_truth = len_per_tile * num_tiles