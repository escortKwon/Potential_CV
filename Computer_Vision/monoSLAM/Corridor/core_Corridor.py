# core.py
#
from matplotlib import projections
import numpy as np
import cv2
import math
import Computer_Vision.monoSLAM.Corridor.params_Corridor as params_Corridor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calibration
print("### Calibration Initiated ###")
CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_corner_points = []
img_corner_points = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

calib_image = cv2.imread(params_Corridor.path_calib_img)
gray = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret == True:
    obj_corner_points.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
    img_corner_points.append(corners2)
    calib_image = cv2.drawChessboardCorners(calib_image, CHECKERBOARD, corners2, ret)

    image_size = (calib_image.shape[1], calib_image.shape[0])
    _, mtx, _, _, _ = cv2.calibrateCamera(obj_corner_points, img_corner_points, image_size, None, None)

    # Extract camera intrinsic parameters from "mtx" as "Pixel Unit"
    fx_px = mtx[0][0]
    fy_px = mtx[1][1]
    cx_px = mtx[0][2]
    cy_px = mtx[1][2]
    focal_px = (fx_px + fy_px)/2
    pp_px = (cx_px, cy_px)

    # Convert "Pixel Unit" to "SI Unit"
    focal = params_Corridor.sizeSensor * focal_px
    pp = (params_Corridor.sizeSensor * pp_px[0], params_Corridor.sizeSensor * pp_px[1])

    print("### Calibration Completed ###")

# Create ORB and BFMatcher
orb = cv2.ORB_create(
                        nfeatures=5000,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        firstLevel=0,
                        WTA_K=2,
                        scoreType=cv2.ORB_FAST_SCORE,
                        patchSize=31,
                        fastThreshold=25,
                        )

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Extract Function
def extract_t_f_coords(t_f):
    """
    This function is for extracting coordinates from "t_f"

    It will return t_f_extract
    """
    for i in range(0, len(t_f)):
        params_Corridor.t_f_new[0][i] = t_f[i][0]
        params_Corridor.t_f_extract = np.append(params_Corridor.t_f_extract, params_Corridor.t_f_new, axis=0)

    return params_Corridor.t_f_extract

# Visualization Function
def visualization_orb_coords(list_text, traj):
    """
    This function will put texts on trajectory following below conditions
    """
    # Text Title
    cv2.putText(traj, params_Corridor.text_title, params_Corridor.textOrg_title, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    for i in range(0, len(list_text)):
        # X, Y and Z Coordinates
        put_text = params_Corridor.list_text_header[i] + str(round(list_text[i], 8)) + "m"
        cv2.putText(traj, put_text, params_Corridor.list_text_Org[i], cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

# Saving Result Function
def save_result(traj):
    # Save mapped image
    cv2.imwrite(params_Corridor.Result_image_name, traj)
    # Save ORB Coordinates for txt file
    params_Corridor.t_f_extract = params_Corridor.t_f_extract[1:]
    np.savetxt(params_Corridor.path_result + 't_f_extract_Corridor.txt', params_Corridor.t_f_extract, fmt="%10s", delimiter=',', header='t_f_extract')

# Calculate total distance of sequence
def calculate_total_distance_and_error():
    # Set Initial Conditions
    i = 0
    distance = 0
    x_prev = 0
    y_prev = 0

    # Calculate
    for i in range(0, len(params_Corridor.t_f_extract)):
        x = params_Corridor.t_f_extract[i][0]
        y = params_Corridor.t_f_extract[i][1]
        distance += math.sqrt((x-x_prev)**2 + (y-y_prev)**2)
        x_prev = x
        y_prev = y
        i += 1
    
    error = (params_Corridor.distance_truth - distance) / params_Corridor.distance_truth
    error_percent = error * 100
    
    print("### Distance Truth: ", params_Corridor.distance_truth, "m ###")
    print("### Distance Calculated: ", distance, "m ###")
    print("### Error: ", error_percent, "% ###") 

# Visualize ORB coordinates as 3D-Plot
def plotting_3D():
    ## Designate paths and load data
    path_data = params_Corridor.path_result + 't_f_extract_Corridor.txt'
    data = open(path_data)
    lines = data.readlines()
    ## Plotting
    x = []
    y = []
    z = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Save coordinates from each axis on the lists
    for i in range(1, len(lines)):
        lines_sp = lines[i].split(',')
        x_coord = float(lines_sp[0])
        y_coord = float(lines_sp[1])
        z_coord = float(lines_sp[2])

        x.append(x_coord)
        y.append(y_coord)
        z.append(z_coord)

        ax.scatter(x, y, z, c='r', alpha=0.5)

    ax.set_xlim3d(min(x), max(x))
    ax.set_ylim3d(min(y), max(y))
    ax.set_zlim3d(min(z), max(z))
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    ax.set_zlabel('$z$', fontsize=20)
    plt.show()