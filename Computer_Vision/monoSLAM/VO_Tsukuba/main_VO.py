import cv2
import numpy as np
from tqdm import tqdm
import time

import params_VO
import core_VO

if __name__ == "__main__":
    print("Visual Odometry from Tsukuba datasets [Nightly / 0.0.2]")
    img_1_c = cv2.imread(params_VO.path_imgs_seqs + "00001.png")
    img_2_c = cv2.imread(params_VO.path_imgs_seqs + "00002.png")
    img_1 = cv2.cvtColor(img_1_c,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c,cv2.COLOR_BGR2GRAY)

    kp1, des1 = core_VO.orb.detectAndCompute(img_1,None)
    kp2, des2 = core_VO.orb.detectAndCompute(img_2,None)

    matches = core_VO.bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    idx = matches[0:1500]

    pts1 = []
    pts2 = []

    for i in idx:
        pts1.append(kp1[i.queryIdx].pt)
        pts2.append(kp2[i.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=params_VO.focal_length, pp=params_VO.principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_f, t_f, _ = cv2.recoverPose(E, pts1, pts2, focal=params_VO.focal_length, pp=params_VO.principal_point)

    R_f_seg = R_f
    t_f_seg = t_f

    # t_gt = np.zeros((3,1), dtype=np.float64)
    t_gt = [0 for i in range(3)]

    prevImage = img_2
    kp_prev = kp2
    des_prev = des2

    traj = np.zeros(params_VO.trajSize, dtype=np.uint8)
    traj = cv2.cvtColor(traj, cv2.COLOR_GRAY2BGR)

    # Set variable for creating progress bar
    pbar = tqdm(total=100)
    
    for numFrame in range(1, params_VO.MAX_FRAME):
        # Update progress bar
        time.sleep(0.1)
        pbar.update(100/params_VO.MAX_FRAME)

        filename = params_VO.path_imgs_seqs + f'{numFrame:05d}.png'

        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)

        # feature extraction
        kp_curr, des_curr = core_VO.orb.detectAndCompute(currImage,None)

        # feature matching
        matches = core_VO.bf.match(des_prev,des_curr)
        matches = sorted(matches, key = lambda x:x.distance)
        idx = matches[0:1500]

        pts1 = []
        pts2 = []

        for i in idx:
            pts1.append(kp_prev[i.queryIdx].pt)
            pts2.append(kp_curr[i.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        # caculate R, t
        E_mat, mask_n = cv2.findEssentialMat(pts2, pts1, focal=params_VO.focal_length, pp=params_VO.principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E_mat, pts2, pts1, focal=params_VO.focal_length, pp=params_VO.principal_point)

        # Calculate Rotation Matrix
        core_VO.eulerAnglesToRotationMatrix(numFrame)

        """
        Error: t_gt --> Translation Vector, not coordinates
        Solution: Get 'translation vector' from ground turthes
                    such as converting translation vector from Euler angles
        """

        # get scale
        abs_scale, t_gt = core_VO.getScale(numFrame, t_gt)

        # Update trajectory
        t_f = t_f +  abs_scale * R_f.dot(t)
        R_f = R.dot(R_f)

        # Extract t_f 
        core_VO.extract_t_f_coords(t_f)

        # Save current data for next step
        prevImage = currImage
        kp_prev = kp_curr
        des_prev = des_curr

        # Visualization
        ## Ground truth
        x_gt = int(t_gt[0]) + 250
        y_gt = int(t_gt[2]) + 500
        ## Features
        x = int(t_f[0]) + 250
        y = int(t_f[2]) + 500
        
        cv2.circle(traj, (x, y), 1, (0,0,255), 2) # Features - Red
        cv2.circle(traj, (x_gt, y_gt), 1, (0,255,0), 2) # Ground Truths - Green
        cv2.rectangle(traj, (10,10), (700, 150), (0,0,0), -1)

        # Create list of Coordinates
        list_features = [t_f[0][0], t_f[1][0], t_f[2][0]]
        list_ground_truths = [t_gt[0], t_gt[1], t_gt[2]]

        # Draw Coordinates on Trajectory
        core_VO.visualization_coords(list_features, list_ground_truths, traj)
        feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)

        cv2.imshow("trajectory", traj)
        # cv2.moveWindow("trajectory", Pos_trajectory[0], Pos_trajectory[1])
        cv2.imshow("feat_img", feature_img)
        # cv2.moveWindow("feat_img", Pos_trajectory[0] + trajSize[0], Pos_trajectory[1])
        
        # core_VO.get_3d_trajectory()
        cv2.waitKey(1)
    
    # Save results
    core_VO.save_result(traj)
    # Calculate total distance of sequence
    core_VO.calculate_total_distance()
    # 3D coordinates plot
    core_VO.plotting_3D()