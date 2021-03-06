import cv2
import numpy as np
import time
from tqdm import tqdm

import Computer_Vision.monoSLAM.Corridor.params_Corridor as params_Corridor
import Computer_Vision.monoSLAM.Corridor.core_Corridor as core_Corridor

if __name__ == "__main__":
    img_1_c = cv2.imread(params_Corridor.path_seq_imgs + "00001.png")
    img_2_c = cv2.imread(params_Corridor.path_seq_imgs + "00002.png")
    img_1 = cv2.cvtColor(img_1_c,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c,cv2.COLOR_BGR2GRAY)

    kp1, des1 = core_Corridor.orb.detectAndCompute(img_1,None)
    kp2, des2 = core_Corridor.orb.detectAndCompute(img_2,None)

    matches = core_Corridor.bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    idx = matches[0:1500]

    pts1 = []
    pts2 = []

    for i in idx:
        pts1.append(kp1[i.queryIdx].pt)
        pts2.append(kp2[i.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=core_Corridor.focal, pp=core_Corridor.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_f, t_f, _ = cv2.recoverPose(E, pts1, pts2, focal=core_Corridor.focal, pp=core_Corridor.pp)

    R_f_seg = R_f
    t_f_seg = t_f

    prevImage = img_2
    kp_prev = kp2
    des_prev = des2

    traj = np.zeros(params_Corridor.trajSize, dtype=np.uint8)
    traj = cv2.cvtColor(traj, cv2.COLOR_GRAY2BGR)

    # Set variable for creating progress bar
    pbar = tqdm(total=100)
    
    for numFrame in range(1, params_Corridor.MAX_FRAME):
        # Update progress bar
        time.sleep(0.1)
        pbar.update(100/params_Corridor.MAX_FRAME)

        filename = params_Corridor.path_seq_imgs + f'{numFrame:05d}.png'

        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)

        # feature extraction
        kp_curr, des_curr = core_Corridor.orb.detectAndCompute(currImage,None)

        # feature matching
        matches = core_Corridor.bf.match(des_prev,des_curr)
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
        E_mat, mask_n = cv2.findEssentialMat(pts2, pts1, focal=core_Corridor.focal, pp=core_Corridor.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E_mat, pts2, pts1, focal=core_Corridor.focal, pp=core_Corridor.pp)

        # Update trajectory
        t_f = t_f + params_Corridor.trajectoryScale * R_f.dot(t)
        R_f = R.dot(R_f)

        # Extract t_f 
        core_Corridor.extract_t_f_coords(t_f)

        # Save current data for next step
        prevImage = currImage
        kp_prev = kp_curr
        des_prev = des_curr

        # Visualization
        x = int(t_f[0] * params_Corridor.featureScale + params_Corridor.trajSize[1]/2)
        y = int(t_f[1] * params_Corridor.featureScale + params_Corridor.trajSize[0]/2)
        
        cv2.circle(traj, (x, y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10,10), (200, 200), (0,0,0), -1)

        # Create list of t_f Coordinates
        list_text = [t_f[0][0], t_f[1][0], t_f[2][0]]

        # Draw ORB Coordinates on Trajectory
        core_Corridor.visualization_orb_coords(list_text, traj)
        feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)

        cv2.imshow("trajectory", traj)
        # cv2.moveWindow("trajectory", Pos_trajectory[0], Pos_trajectory[1])
        cv2.imshow("feat_img", feature_img)
        # cv2.moveWindow("feat_img", Pos_trajectory[0] + trajSize[0], Pos_trajectory[1])
        cv2.waitKey(1)
    
    # Save results
    core_Corridor.save_result(traj)
    # Calculate total distance of sequence
    core_Corridor.calculate_total_distance_and_error()
    # 3D coordinates plot
    core_Corridor.plotting_3D()