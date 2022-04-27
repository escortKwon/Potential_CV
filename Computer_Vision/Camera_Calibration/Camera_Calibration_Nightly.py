import numpy as np
import cv2
import os

# Set environment path
cwd = os.getcwd()
path_detected_corners = '/Practice_CV/Camera_Calibration/Detected_Corners/Detected_Corners_'
path_undistortion = '/Practice_CV/Camera_Calibration/Results/'

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

# Boundary condition
CHECKERBOARD = (6, 10) # 체커보드 행과 열당 내부 코너 수
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

# cap = cv2.VideoCapture(cv2.CAP_DSHOW) # Laptop Cam
cap = cv2.VideoCapture(1) # Third-Party Cam

while cap.isOpened():

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    cv2.imshow('Distorted Frame', frame)

    # Key Input
    key = cv2.waitKey(1)

    # Detection
    if key == ord('d'):
        if ret == True:
            obj_corner_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_corner_points.append(corners2)

            frame_corners = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
            print("Detected Coordinates: ", corners2)

            cv2.imwrite(cwd + path_detected_corners + str(i) + ".png", gray)
            print("### ", i, " of 10 Saved ###")
            i += 1
            cv2.imshow("Corners", frame_corners)
            if i > 10:
                print("### Press 'C' to go on next step ###")

    # Calibration
    if (key == ord('c')) and (len(obj_corner_points) != 0):
        if len(obj_corner_points) == 0:
            print("### None Detected Corners ###")
        print("### Calibration Initiated ###")
        image_size = get_image_size_from_cap(cap)
        ret, mtx, dist, rves, tvecs = cv2.calibrateCamera(obj_corner_points, img_corner_points, image_size, None, None)
        print("### Calibration Completed ###")

        h, w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print("### newcameramtx, roi Generated ###")
        isCalibrated = True

    # Reset Variables
    if key == ord('r'):
        print("### Clear Variables Process Initiated ###")
        clear_input = input(">>> Press 'Y' if you want to clear variables (Y/N): ")
        if clear_input == 'Y':
            obj_corner_points = []
            img_corner_points = []
            i = 0
            print("### Clear Variables Completed ###")
        elif clear_input == 'N':
            obj_corner_points = obj_corner_points
            img_corner_points = img_corner_points
            i = i
            print("### Clear Variables Cancelled")
        else:
            print("### Choose Proper Option ###")

    # Quit
    elif key == ord('q'):
        print("### Program Terminated ###")
        cap.release()
        cv2.destroyAllWindows()
        break

    # Undistortion
    if isCalibrated:
        frame_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        frame_undistorted = frame_undistorted[y:y+h, x:x+w]
        cv2.imshow("Undistorted Frame", frame_undistorted)
        cv2.imwrite(cwd + path_undistortion + "Undistorted_Result.png", frame_undistorted)