#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan

import numpy as np
from sklearn import linear_model

LARGE_NUM = 88888

def parse(lidar_scan):

    tuning_params = {
        "n_searched_landmarks": 4,
        "landmark_criteria_angle_spacing": 3,
        "residual_threshold": 0.05,
        "n_inlier_stop_criteria": 8,
        "object_padding": 0.02,
    }

    angle_min = lidar_scan.angle_min
    angle_max = lidar_scan.angle_max
    angle_increment = lidar_scan.angle_increment

    ranges, angles = zip(*[(ele,angle_min+i*angle_increment) for i, ele in enumerate(lidar_scan.ranges) if ele != float('inf') and ele != 0])

    ranges = np.array(ranges)
    angles = np.array(angles)

    landmarks = []

    for _ in range(tuning_params["n_searched_landmarks"]):

        """
        Ax, Ay = np.meshgrid(angles, angles)
        angular_spacings = np.abs(Ax-Ay) + LARGE_NUM*np.eye(len(angles))
        landmark_criterion = tuning_params["landmark_criteria_angle_spacing"]

        mask = (angular_spacings <= landmark_criterion*angle_increment).any(axis=1)

        angles = angles[mask]
        ranges = ranges[mask]
        """

        x = np.expand_dims(ranges * np.cos(angles), -1)
        y = np.expand_dims(ranges * np.sin(angles), -1)

        ransac = linear_model.RANSACRegressor(\
                residual_threshold=tuning_params["residual_threshold"])
        ransac.fit(x, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        if len(x[inlier_mask]) <= tuning_params["n_inlier_stop_criteria"]:
            break

        tol = tuning_params["object_padding"]
        line_X = np.array([x[inlier_mask][1:-1].min()-tol, x[inlier_mask][1:-1].max()+tol])[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        landmarks.append({"x": line_X.tolist(), "y": line_y_ransac.tolist()})

        angles = angles[outlier_mask]
        ranges = ranges[outlier_mask]


    rospy.loginfo(rospy.get_caller_id() + "I found the following landmarks \n %s", landmarks)

    
def ransac_parser():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ransac_parser', anonymous=True)

    rospy.Subscriber("scan", LaserScan, parse)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    ransac_parser()
