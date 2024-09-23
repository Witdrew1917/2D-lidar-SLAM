import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.linalg import expm
from math import pi

from typing import Type


class Square:

    def __init__(self, size: float, frequency: float, time: float):

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

        X = np.zeros((3,n_points))
        X[0,:] = np.linspace(0,size,n_points)

        for i in range(0, n_points, int(n_points/4)):
            if i == 0:
                continue

            X[:2,i:] = R @ X[:2,i:] + np.expand_dims(-R @ X[:2,i] + X[:2,i-1],-1)
            X[2,i:] = i*pi/4*np.ones(len(X[2,i:]))

        #plt.plot(X[0,:], X[1,:])
        #plt.show()

        trajectory = X.T

        return trajectory

class Bicycle_Model:

    def __init__(self, wheelbase: float, step_size: float, velocity: float,\
            initial_state=np.array([[0],[0],[0]]), linear=True):

        self.X = initial_state
        self.wheelbase = wheelbase
        self.step_size = step_size
        self.velocity = velocity


        if linear:
            self._create_linear_model()
        else:
            self._create_non_linear_model()



    def _create_linear_model(self):

        self.A = np.array([[.0, .0, -self.velocity/3],
                           [.0, .0, self.velocity],
                           [.0, .0, .0]])

        self.B = np.array([[.0],
                           [.0],
                           [self.velocity/self.wheelbase]])

        dAB = expm(self.step_size*np.concatenate(
            [np.concatenate([self.A, self.B], axis=1),np.array([[0,0,0,1]])],
            axis=0))

        self.A = dAB[:3,:3]
        self.B = np.expand_dims(dAB[3,:3],axis=-1)
        self.D = np.array([[self.velocity],
                           [.0],
                           [.0]])
    

    def _create_non_linear_model(self):
        pass


    def update(self, U):
        
        self.X = self.A @ self.X + self.B @ U + self.D



if __name__ == '__main__':

    time=10
    frequency=10
    size=4

    square = Square(size, time, frequency)
    speed = size/time

    model = Bicycle_Model(0.3, 1/frequency, speed)

    horizon = time*frequency
    reference_trajectory = np.concatenate([square.trajectory.reshape(3*horizon,1), \
            np.zeros((horizon,1))],axis=0)
    reference_trajectory = np.ndarray.flatten(reference_trajectory)
    

    # Let Q, P, R be unit matrices
    H = np.eye(3*horizon+1*horizon)

    # The below statement also implies an initial condition of (0,0,0)
    eqs_A1 = np.eye(3*horizon) + np.kron(np.diag(np.ones(horizon-1),-1), -model.A)
    eqs_A2 = np.kron(np.eye(horizon), -model.B)
    eqs_A = np.concatenate([eqs_A1, eqs_A2], axis=1)
    eqs_B = np.expand_dims(\
            np.ndarray.flatten(np.array([model.D]*horizon)), axis=-1)

    max_angle = pi/4
    state_dev = 0.2

    ineq1_A = np.concatenate([np.zeros((2*horizon,3*horizon)),\
            np.concatenate([np.eye(horizon),-1*np.eye(horizon)], axis=0)], axis=1)
    ineq1_B = np.concatenate([-max_angle*np.ones((horizon,1)), \
            max_angle*np.ones((horizon,1))], axis=0)

    """
    ineq2_A = np.concatenate([np.kron(np.eye(horizon), \
            np.array([[1,0,0],[0,1,0]])), np.zeros((2*horizon,horizon))], axis=1)
    ineq2_A = np.concatenate([ineq2_A,-ineq2_A], axis=0)
    ineq2_B = np.concatenate([reference_trajectory-state_dev,\
            state_dev+reference_trajectory], axis=0)

    ineqs_A = np.concatenate([ineq1_A, ineq2_A], axis=0)
    ineqs_B = np.concatenate([ineq1_B, ineq2_B], axis=0)
    """

    ineqs_A = ineq1_A
    ineqs_B = ineq1_B


    def loss(x, sign=1.):
        return sign * (0.5 * np.dot((x-reference_trajectory).T, np.dot(H, (x-reference_trajectory))))

    def jac(x, sign=1.):
        return sign * (np.dot((x-reference_trajectory).T, H))

    cons = [{'type':'eq',
        'fun':lambda x: np.ndarray.flatten(eqs_B) - np.dot(eqs_A,x),
        'jac':lambda x: -eqs_A},\
                {'type':'ineq',
        'fun':lambda x: np.ndarray.flatten(ineqs_B) - np.dot(ineqs_A,x),
        'jac':lambda x: -ineqs_A}
            ]
    opt = {'disp':False}

    x0 = np.zeros(4*horizon)
    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)
    print(res_cons)
    print(res_cons.x)




    











