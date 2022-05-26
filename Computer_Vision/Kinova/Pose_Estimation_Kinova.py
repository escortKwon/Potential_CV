import numpy as np
import cv2
import os

print("### Pose_Estimation_Kinova.py [Beta Version] / Ver.0.1.0 ###")

# Set environment path
cwd = os.getcwd()
path_data = cwd + '/Computer_Vision/Kinova/Data/'
path_test = cwd + '/Computer_Vision/Kinova/Test/'
path_undistortion = '/Computer_Vision/Kinova/Results/'
path_results = cwd + '/Computer_Vision/Kinova/Results/'

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
CHECKERBOARD = (6, 8) # 체커보드 행과 열당 내부 코너 수
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

# Set key input
key = cv2.waitKey(1)

# Set Boolean
isCalibrated = False

# Set section for loading data images
data_start = 1
data_end = 20

# Set 'image_size as global
global image_size

# Draw chessboard corners
for i in range(data_start, data_end + 1):
    path_data_imgs = path_data + f'{i:02d}.jpg'
    img = cv2.imread(path_data_imgs)
    image_size = img.shape[:2]
    cv2.imshow("src", img)
    key = cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        obj_corner_points.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        img_corner_points.append(corners)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard", img)
        print(f">>> {i} of {data_end} has been captured")

    if i == data_end:
        isCalibrated = True
        print(">>> Go on to the next step")
    
if isCalibrated:
    ret, mtx, dist, rves, tvecs = cv2.calibrateCamera(obj_corner_points, img_corner_points, image_size, None, None)
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    test_start = 1
    test_end = 2
    for i in range(test_start, test_end + 1):
        path_test_imgs = path_test + f'{i:02d}.jpg'
        test_img = cv2.imread(path_test_imgs)
        frame_undistorted_test = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
        frame_undistorted_test_gray = cv2.cvtColor(frame_undistorted_test, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("Undistorted Frame", frame_undistorted_test)
        key = cv2.waitKey(0)
        cv2.imwrite(cwd + path_undistortion + f"Undistorted_Result_Test_{i}.png", frame_undistorted_test)

        corners2 = cv2.cornerSubPix(frame_undistorted_test_gray, corners, (11,11), (-1,-1), criteria)
        # Find the rotation and translation vectors
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # Project 3D points to image lane
        img_pts, jac = cv2.projectPoints(axis_simple, rvecs, tvecs, mtx, dist)
        axis_3d = draw_axis_3d(test_img, corners2, img_pts)
        cv2.imshow("Axis_3D", axis_3d)
        key = cv2.waitKey(0)

        # Apply Rodrigues method
        rmatrix, _ = cv2.Rodrigues(rvecs)
        rmatrix = rmatrix.reshape(3, 3)

        # Extract rvecs, tvecs
        rvecs_header = np.array(['# rvecs'])
        rvecs_extract = np.append(rvecs_header, rvecs)
        tvecs_header = np.array(['# tvecs'])
        tvecs_extract = np.append(tvecs_header, tvecs)
        Extrinsic_Params_Vecs = np.concatenate((rvecs_extract, tvecs_extract), axis=0)
        np.savetxt(path_results + f'Pose_Estimation_Kinova_Vecs_{i}.txt', Extrinsic_Params_Vecs, fmt="%10s", delimiter=',', header='Vectors')

        # Extract rmatrix
        np.savetxt(path_results + f'Pose_Estimation_Kinova_rmatrix_{i}.txt', rmatrix, fmt="%10s", delimiter=',', header='Rotation Matrix')

        # Homogeneous Transformation
        Homo_Trans = np.concatenate((rmatrix, tvecs), axis=1)
        Row_Project = np.array([0, 0, 0, 1])
        Homo_Trans = np.vstack([Homo_Trans, Row_Project])
        
        # Extract Homogeneous Transformation
        np.savetxt(path_results + f'Pose_Estimation_Kinova_HomoMatrix_{i}.txt', Homo_Trans, fmt="%10s", delimiter=',', header='Homogeneous Matrix')