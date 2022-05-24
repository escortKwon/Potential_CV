import cv2
import numpy as np
import math
import os
import platform
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set Environment Path
cwd = os.getcwd()
# path_poses = cwd + '/Practice_CV/monoSLAM/odometry_ground_truth_poses/poses/'
path_write = cwd + '/Computer_Vision/monoSLAM/Results/'

if platform.system() == "Darwin":
    path_imgs = '/Users/escortkwon/Kitti Dataset/dataset/sequences/' # Mac
elif platform.system() == "Windows":
    path_imgs = 'D:/Kitti Dataset/dataset/sequences/' # Windows
elif platform.system() == "Linux":
    path_imgs = '/mnt/d/Kitti Dataset/dataset/sequences/' # Linux
    path_poses = '/mnt/d/Kitti Dataset/odometry_ground_truth_poses/poses/'

# Create ORB and BFMatcher
orb = cv2.cv2.ORB_create(
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


def getScale(NumFrame, t_gt, seq_num):

    txt_file = open(path_poses + '{0:02d}.txt'.format(seq_num))
    
    x_prev = float(t_gt[0])
    y_prev = float(t_gt[1])
    z_prev = float(t_gt[2])

    line = txt_file.readlines()
    line_sp = line[NumFrame].split(' ')

    x = float(line_sp[3])
    y = float(line_sp[7])
    z = float(line_sp[11])

    t_gt[0] = x
    t_gt[1] = y
    t_gt[2] = z

    txt_file.close()

    scale = math.sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2)
    return scale, t_gt


if __name__ == "__main__":
    MAX_FRAME = 1000
    SEQ_NUM = 2

    # Camera intrinsic parameters
    focal = 718.8560
    pp = (607.1928, 185.2157)

    textOrg1 = (10,30)
    textOrg2 = (10,80)
    textOrg3 = (10,130)

    img_1_c = cv2.imread(path_imgs + "{0:02d}/image_2/000000.png".format(SEQ_NUM))
    img_2_c = cv2.imread(path_imgs + "{0:02d}/image_2/000001.png".format(SEQ_NUM))
    img_1 = cv2.cvtColor(img_1_c,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img_1,None)
    kp2, des2 = orb.detectAndCompute(img_2,None)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    idx = matches[0:1500]

    pts1 = []
    pts2 = []

    for i in idx:
        pts1.append(kp1[i.queryIdx].pt)
        pts2.append(kp2[i.trainIdx].pt)


    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    E, mask = cv2.findEssentialMat(pts1,pts2,focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
    _, R_f, t_f, _ = cv2.recoverPose(E, pts1, pts2, focal = focal, pp = pp)

    R_f_seg = R_f
    t_f_seg = t_f

    t_gt = np.zeros((3,1),dtype=np.float64)

    prevImage = img_2
    kp_prev = kp2
    des_prev = des2

    traj = np.zeros((1000,2000),dtype=np.uint8)
    traj = cv2.cvtColor(traj,cv2.COLOR_GRAY2BGR)

    rmse_total = 0

    # Generate empty numpy arrays for extarct t_f
    t_f_extract = np.zeros(shape=(1, 3))
    
    for numFrame in range(2, MAX_FRAME):
        filename = path_imgs + '{0:02d}/image_2/{1:06d}.png'.format(SEQ_NUM,numFrame)
        
        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)

        # feature extraction
        kp_curr, des_curr = orb.detectAndCompute(currImage,None)

        # feature matching
        matches = bf.match(des_prev,des_curr)
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
        E_mat, mask_n = cv2.findEssentialMat(pts2, pts1, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E_mat, pts2, pts1, focal = focal, pp = pp)

        # get scale
        abs_scale, t_gt = getScale(numFrame, t_gt, SEQ_NUM)
        
        # update trajectory
        t_f = t_f + abs_scale*R_f.dot(t)
        R_f = R.dot(R_f)

        # Extract t_f
        t_f_x = t_f[0][0]
        t_f_y = t_f[1][0]
        t_f_z = t_f[2][0]
        t_f_new = [[t_f_x, t_f_y, t_f_z]]
        t_f_extract = np.append(t_f_extract, t_f_new, axis=0)

        # caculate Error
        error = map(operator.sub,t_gt,t_f)
        error_sum_square = sum(map(lambda x:x*x,error))
        rmse = math.sqrt(error_sum_square/3)
        rmse_total = rmse_total + rmse

        print("rmse     = ", rmse_total/numFrame)

        prevImage = currImage
        kp_prev = kp_curr
        des_prev = des_curr

        # visualization
        x_gt = int(t_gt[0]) + 1000
        y_gt = int(t_gt[2]) + 100

        x = int(t_f[0]) + 1000
        y = int(t_f[2]) + 100

        cv2.circle(traj, (x,y), 1 , (0,0,255), 2)
        cv2.circle(traj, (x_gt,y_gt), 1 , (0,255,0), 2)
        
        cv2.rectangle(traj, (10,10), (700,150), (0,0,0), -1)
        text1 = 'orb Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f[0]),float(t_f[1]),float(t_f[2]))
        cv2.putText(traj, text1, textOrg1, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)
        

        text3 = 'gt  Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_gt[0]),float(t_gt[1]),float(t_gt[2]))
        cv2.putText(traj, text3, textOrg3, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)

        feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)

        cv2.imshow("trajectory", traj)
        cv2.imshow("feat_img", feature_img)

        cv2.waitKey(1)
    
    # Save results
    cv2.imwrite(path_write + 'Result_monoSLAM_{0:02d}_{1}.png'.format(SEQ_NUM, platform.system()), traj)
    t_f_extract = t_f_extract[1:]
    np.savetxt(path_write + 't_f_extract.txt', t_f_extract, fmt="%10s", delimiter=',', header='t_f_extract')

# Visualize ORB coordinates as 3D-Plot
def plotting_3D():
    ## Designate paths and load data
    path_data = path_write + 't_f_extract.txt'
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

plotting_3D()