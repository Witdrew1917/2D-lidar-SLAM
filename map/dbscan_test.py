import argparse
import yaml
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def cluster(short_term_memory, long_term_memory, clustering):

    X = np.array(short_term_memory + long_term_memory)

    clustering.fit(X)
    labels = clustering.labels_

    # Save unclustered landmarks in LTM
    if len(long_term_memory) > 0:
        long_term_memory = np.array(long_term_memory)
        long_term_memory = long_term_memory[labels[-len(long_term_memory):] == -1].tolist()

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True

    for k in unique_labels:
        if k == -1:
            continue
            
        class_member_mask = labels == k
        xy = X[class_member_mask]
        long_term_memory.append(np.mean(xy,axis=0).tolist())
        plt.plot(
            long_term_memory[-1][:2],
            long_term_memory[-1][2:],
            color='k',
            linewidth = 2
        )

    #plt.show()
    #print(long_term_memory)

    return [], long_term_memory


parser = argparse.ArgumentParser()
parser.add_argument('-f','--log_file', type=str, required=True)

args = parser.parse_args()

##tuning
max_len = 100
max_dist_samples = 0.5
min_samples = 20

STM = []
LTM = [[0.,0.,0.,1.]]
clustering = DBSCAN(eps=max_dist_samples, min_samples=min_samples)

with open(args.log_file, 'r') as f:

    landmark_data = yaml.safe_load_all(f)
    t0 = perf_counter()
    
    for i, landmark in enumerate(landmark_data):

        if landmark == None: ## Remove this!
            continue

        feature = landmark['x'] + landmark['y']
        STM.append(feature)

        if len(STM) == max_len:
            STM, LTM = cluster(STM, LTM, clustering)
            t1 = perf_counter()
            print(f"{t1-t0} sec")
            t0 = t1

