# core_VO.py
#
from matplotlib import projections
import numpy as np
import cv2
import math
import params_VO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(numFrame):

    txt_file = open(params_VO.path_ground_truth)
    line = txt_file.readlines()
    line_sp = line[numFrame].split(',')
    eulerAngles = [float(line_sp[3]), float(line_sp[4]), float(line_sp[5])]

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(eulerAngles[0]), -math.sin(eulerAngles[0]) ],
                    [0,         math.sin(eulerAngles[0]), math.cos(eulerAngles[0])  ]
                    ])

    R_y = np.array([[math.cos(eulerAngles[1]),    0,      math.sin(eulerAngles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(eulerAngles[1]),   0,      math.cos(eulerAngles[1])  ]
                    ])

    R_z = np.array([[math.cos(eulerAngles[2]),    -math.sin(eulerAngles[2]),    0],
                    [math.sin(eulerAngles[2]),    math.cos(eulerAngles[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    rmatrix, _ = cv2.Rodrigues(R)

    return rmatrix

# Caculating scale function
def getScale(numFrame, t_gt):

    txt_file = open(params_VO.path_ground_truth)
    
    x_prev = float(t_gt[0])
    y_prev = float(t_gt[1])
    z_prev = float(t_gt[2])

    line = txt_file.readlines()
    line_sp = line[numFrame].split(',')

    x = float(line_sp[0])
    y = float(line_sp[1])
    z = float(line_sp[2])

    t_gt[0] = x
    t_gt[1] = y
    t_gt[2] = z

    txt_file.close()

    scale = math.sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2)
    return scale, t_gt

# Extract Function
def extract_t_f_coords(t_f):
    """
    This function is for extracting coordinates from "t_f"

    It will return t_f_extract
    """
    for i in range(0, len(t_f)):
        params_VO.t_f_new[0][i] = t_f[i][0]
        params_VO.t_f_extract = np.append(params_VO.t_f_extract, params_VO.t_f_new, axis=0)

    return params_VO.t_f_extract

# Visualization Function
def visualization_coords(list_features, list_ground_truths, traj):
    """
    This function will put texts on trajectory following below conditions
    """
    # Text Title
    cv2.putText(traj, params_VO.text_title_f, params_VO.textOrg_title_f, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.putText(traj, params_VO.text_title_gt, params_VO.textOrg_title_gt, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    # Features
    for i in range(0, len(list_features)):
        put_text_f = params_VO.list_text_header[i] + str(round(list_features[i], 8)) + "cm"
        cv2.putText(traj, put_text_f, params_VO.list_text_Org_f[i], cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    # Ground Truths
    for i in range(0, len(list_ground_truths)):
        put_text_gt = params_VO.list_text_header[i] + str(round(list_ground_truths[i], 8)) + "cm"
        cv2.putText(traj, put_text_gt, params_VO.list_text_Org_gt[i], cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

# Saving Result Function
def save_result(traj):
    # Save mapped image
    cv2.imwrite(params_VO.Result_image_name, traj)
    # Save ORB Coordinates for txt file
    params_VO.t_f_extract = params_VO.t_f_extract[1:]
    np.savetxt(params_VO.path_result + 't_f_extract_Tsukuba.txt', params_VO.t_f_extract, fmt="%10s", delimiter=',', header='t_f_extract')

# Calculate total distance of sequence
def calculate_total_distance():
    # Set Initial Conditions
    i = 0
    distance = 0
    x_prev = 0
    y_prev = 0

    # Calculate
    for i in range(0, len(params_VO.t_f_extract)):
        x = params_VO.t_f_extract[i][0]
        y = params_VO.t_f_extract[i][1]
        distance += math.sqrt((x-x_prev)**2 + (y-y_prev)**2)
        x_prev = x
        y_prev = y
        i += 1

    print("### Distance Calculated: ", distance, "m ###")

# Visualize ORB coordinates as 3D-Plot
def plotting_3D():
    ## Designate paths and load data
    path_data = params_VO.path_result + 't_f_extract_Tsukuba.txt'
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