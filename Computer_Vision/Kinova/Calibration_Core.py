import numpy as np
import cv2
import os

# Set environment path
cwd = os.getcwd()
path_detected_corners = '/Potential_CV/Kinvoa/Detected_Corners/Detected_Corners_'
path_undistortion = '/Potential_CV/Kinvoa/Results/'

# Boundary condition
CHECKERBOARD = (6, 8) # 체커보드 행과 열당 내부 코너 수
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Points list
obj_corner_points = []
img_corner_points = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Iteration variable
i = 1

# Set boolean
isCalibrated = False

class Calibration():
    def get_image_size_from_cap(self, cap:cv2.VideoCapture):
        """
        Get image size from cv2.VideoCapture

        Return is (Width, Height)
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_size = (width, height)

        return image_size

    def detection(self, gray, corners, frame, ret):
        self.gray = gray
        self.corners = corners
        self.frame = frame
        self.ret = ret

        obj_corner_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_corner_points.append(corners2)

        frame_corners = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
        print("Detected Coordinates: ", corners2)

        
