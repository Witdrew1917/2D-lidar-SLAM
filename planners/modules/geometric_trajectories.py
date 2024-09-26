import numpy as np
from math import pi

class Square:

    def __init__(self, size, frequency, time):

        self.trajectory = self._create_trajectory(size, frequency, time)
        self.it_trajectory = iter(self.trajectory.tolist())
        self.index = 0


    def __iter__(self):
        return self.it_trajectory



    def __next__(self):
        next(self.it_trajectory) 


    def _create_trajectory(self, size, frequency, time):

        
        R = np.array([[np.cos(pi/2), -np.sin(pi/2)],
                     [np.sin(pi/2), np.cos(pi/2)]])

        n_points = int(time*frequency)

        X = np.zeros((4,n_points))
        X[0,:] = np.linspace(0,size,n_points)

        for it, index in enumerate(range(0, n_points, int(n_points/4))):
            if index == 0:
                continue

            X[:2,index:] = np.dot(R,X[:2,index:]) + \
                    np.expand_dims(np.dot(-R,X[:2,index]) + X[:2,index-1],-1)
            X[2,index:] = it*pi/2*np.ones(len(X[2,index:]))
            X[3,index] = pi/2

        trajectory = X.T

        return trajectory
