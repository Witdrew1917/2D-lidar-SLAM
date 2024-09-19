import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
from copy import deepcopy
from time import perf_counter

from sklearn import linear_model

LARGE_NUM = 88888

parser = argparse.ArgumentParser()
parser.add_argument('-f','--log_file', type=str, required=True)
parser.add_argument('-p','--tuning_params', type=str, required=True, \
        help="yaml file containing tuning parameters of ransac")

args = parser.parse_args()


with open(args.log_file, 'r') as f:
    lidar_scan = yaml.safe_load(f)

with open(args.tuning_params, 'r') as f:
    tuning_params = yaml.safe_load(f)


t0 = perf_counter()

angle_min = lidar_scan["angle_min"]
angle_max = lidar_scan["angle_max"]
angle_increment = lidar_scan["angle_increment"]

ranges, angles = zip(*[(int(ele),angle_min+i*angle_increment) for i, ele in enumerate(lidar_scan["ranges"]) if type(ele) is not type(str()) and int(ele) != 0])

original_ranges = deepcopy(ranges)
original_angles = deepcopy(angles)

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

print(f"{perf_counter() - t0} sec")

plt.figure()
lw = 2
original_x = np.expand_dims(original_ranges * np.cos(original_angles), -1)
original_y = np.expand_dims(original_ranges * np.sin(original_angles), -1)
plt.scatter(original_x, original_y, color="orange", marker=".", label="original points"
)
for i, landmark in enumerate(landmarks):
    plt.plot(landmark['x'], landmark['y'], color="cornflowerblue", \
            linewidth=lw,label=f" landmark {i}",
    )
plt.legend(loc="lower right")
plt.show()


