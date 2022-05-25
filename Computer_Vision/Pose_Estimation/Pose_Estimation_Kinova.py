import numpy as np
import cv2
import os

print("### Pose_Estimation.py [Stable Version] / 1.0.0 ###")

# Set Environment Path
cwd = os.getcwd()
path_results = cwd + '/Computer_Vision/Pose_Estimation/Results/'

# Utility Function
def get_image_size_from_cap(cap:cv2.VideoCapture):
    """
    Get image size from cv2.VideoCapture

    Return is (Width, Height)
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (width, height)

    return image_size

# Draw Function
def draw_axis_3d(img, corners, img_pts):
    """
    This function will draw 3-Dimension axis from

    cv2.findChessboardCorners()
    """
    corner = tuple(corners[0].ravel().astype(np.int64))
    img = cv2.line(img, corner, tuple(img_pts[0].ravel().astype(np.int64)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(img_pts[1].ravel().astype(np.int64)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(img_pts[2].ravel().astype(np.int64)), (0,0,255), 5)
    return img

# Draw Function
def draw_cube_3d(img, corners, img_pts):
    """
    This function will draw 3-Dimension cube from

    cv2.findChessboardCorners()
    """
    img_pts = np.int64(img_pts).reshape(-1,2)
    # Draw ground floor in green
    img = cv2.drawContours(img, [img_pts[:4]],-1,(0,255,0),-3)
    # Draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(img_pts[i]),tuple(img_pts[j]),(255),3)
    # Draw top layer in red color
    img = cv2.drawContours(img, [img_pts[4:]],-1,(0,0,255),3)
    return img

# Boundary condition
CHECKERBOARD_ROW = 6
CHECKERBOARD_COL = 8
CHECKERBOARD = (CHECKERBOARD_ROW, CHECKERBOARD_COL)
CHECKERBOARD_AREA = CHECKERBOARD_ROW * CHECKERBOARD_COL
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Points list
obj_corner_points = []
img_corner_points = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Create axis
axis_simple = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis_cube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

# Set boolean
isCalibrated = False

rtsp_link = 'rtsp://admin:admin@192.168.1.10/color'
cap = cv2.VideoCapture(rtsp_link)
# cap = cv2.VideoCapture(cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(1)

while cap.isOpened():

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    cv2.imshow('Distorted Frame', frame)

    # Key Input
    key = cv2.waitKey(1)

    # Calibration
    if key == ord('c'):
        if ret == True:
            obj_corner_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_corner_points.append(corners2)

            frame_corners = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

            # Slicing lists of corner points
            if len(obj_corner_points[0][0]) >= CHECKERBOARD_AREA:
                obj_corner_points = obj_corner_points[-CHECKERBOARD_AREA:]
            if len(img_corner_points[0]) >= CHECKERBOARD_AREA:
                img_corner_points = img_corner_points[-CHECKERBOARD_AREA:]
            
            cv2.imshow("Corners", frame_corners)

            # Calibration
            print("### Calibration Initiated ###")
            image_size = get_image_size_from_cap(cap)
            _, mtx, dist, _, _ = cv2.calibrateCamera(obj_corner_points, img_corner_points, image_size, None, None)
            print("### Calibration Completed ###")
            isCalibrated = True

    # Pose Estimation_Axis
    if isCalibrated:
        corners = np.float32(corners)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # Find the rotation and translation vectors
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # Project 3D points to image lane
        img_pts, jac = cv2.projectPoints(axis_simple, rvecs, tvecs, mtx, dist)
        axis_3d = draw_axis_3d(frame, corners2, img_pts)
        cv2.imshow("Sample_Axis_3D", axis_3d)

        # Apply Rodrigues method
        rmatrix, _ = cv2.Rodrigues(rvecs)
        rmatrix = rmatrix.reshape(3, 3)

        # Extract rvecs, tvecs
        rvecs_header = np.array(['# rvecs'])
        rvecs_extract = np.append(rvecs_header, rvecs)
        tvecs_header = np.array(['# tvecs'])
        tvecs_extract = np.append(tvecs_header, tvecs)
        Extrinsic_Params_Vecs = np.concatenate((rvecs_extract, tvecs_extract), axis=0)
        np.savetxt(path_results + 'Pose_Estimation_Extrinsic_Params_Vecs.txt', Extrinsic_Params_Vecs, fmt="%10s", delimiter=',', header='Extrinsic_Parameters_Vectors')

        # Extract rmatrix
        np.savetxt(path_results + 'Pose_Estimation_Extrinsic_Params_rmatrix.txt', rmatrix, fmt="%10s", delimiter=',', header='Rotation Matrix')

        # Homogeneous Transformation
        Homo_Trans = np.concatenate((rmatrix, tvecs), axis=1)
        Row_Project = np.array([0, 0, 0, 1])
        Homo_Trans = np.vstack([Homo_Trans, Row_Project])
        
        # Extract Homogeneous Transformation
        np.savetxt(path_results + 'Pose_Estimation_Extrinsic_Params_Homo_Trans.txt', Homo_Trans, fmt="%10s", delimiter=',', header='Homogeneous Transformation')

    # Quit
    elif key == ord('q'):
        print("### VideoCapture Terminated ###")
        cap.release()
        cv2.destroyAllWindows()
        break