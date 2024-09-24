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

        X = np.zeros((4,n_points))
        X[0,:] = np.linspace(0,size,n_points)

        for it, index in enumerate(range(0, n_points, int(n_points/4))):
            if index == 0:
                continue

            X[:2,index:] = R @ X[:2,index:] + np.expand_dims(-R @ X[:2,index] + X[:2,index-1],-1)
            X[2,index:] = it*pi/2*np.ones(len(X[2,index:]))
            X[3,index] = pi/2

        plt.plot(X[0,:], X[1,:])
        #plt.show()

        trajectory = X.T

        return trajectory


class Bicycle_Model:

    def __init__(self, wheelbase: float, step_size: float, velocity: float,\
            linear=True, dim=3):

        self.X = np.zeros(dim).reshape(dim,1)
        self.wheelbase = wheelbase
        self.step_size = step_size
        self.velocity = velocity
        self.dim = dim


        if linear:
            if dim == 1:
                self._create_1D_linear_model()
            elif dim == 3:
                self._create_linear_model()
            else:
                print(f"dim size must be 1 or 3, was {dim}")
        else:
            print("creating non-linear model")
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
        self.B = np.expand_dims(dAB[:3,3],axis=-1)
        self.D = np.array([[self.velocity],
                           [.0],
                           [.0]])
    

    def _create_1D_linear_model(self):

        self.A = 0
        self.B = self.velocity/self.wheelbase

        dAB = expm(self.step_size*np.array([[self.A, self.B],[0,1]]))
        self.A = np.expand_dims(dAB[0,0],axis=-1)
        self.B = np.expand_dims(dAB[0,1],axis=-1)
        self.D = 0


    def _create_non_linear_model(self):

        assert self.dim == 3

        self.update = self.non_linear_upadate


    def update(self, U):
        
        self.X = self.A @ self.X + self.B * U + self.D

    def non_linear_upadate(self, U):


        A = np.array([self.velocity*np.cos(self.X[2,0]),
                      self.velocity*np.sin(self.X[2,0]),
                      0]).reshape(self.dim,1)

        B = np.array([0,0,self.velocity/self.wheelbase*np.tan(U)]).reshape(
                self.dim, 1)

        self.X += self.step_size*(A + B)



if __name__ == '__main__':

    time=40
    frequency=10
    size=4
    state_size = 1
    control_size = 1

    square = Square(size, time, frequency)
    speed = size/time

    model = Bicycle_Model(0.3, 1/frequency, speed, dim=state_size)

    horizon = time*frequency
    reference_trajectory = np.zeros(((state_size+control_size)*horizon,1))
    if state_size == 1:
        reference_trajectory[:state_size*horizon,0] = \
                square.trajectory[:horizon,2].reshape(-1)
    else:
        reference_trajectory[:state_size*horizon,0] = \
                square.trajectory[:horizon,:2].reshape(-1)
    reference_trajectory[state_size*horizon:,0] = \
            square.trajectory[:horizon,-1].reshape(-1)
    


    # Let Q, P, R be unit matrices
    Q = np.eye(state_size)
    R = 1
    P = 100*np.eye(state_size)
    H = np.eye(state_size*horizon+control_size*horizon)
    q_index = len(Q)*(horizon-1)
    r_index = q_index + len(P)
    #H[:q_index,:q_index] = np.diag(np.arange(horizon-1))
    H[:q_index,:q_index] = np.kron(np.eye(horizon-1),Q)
    H[q_index:r_index, q_index:r_index] = P
    H[r_index:, r_index:] = R*np.eye(horizon)

    # The below statement also implies an initial condition of (0,0,0)
    eqs_A1 = np.eye(state_size*horizon) + np.kron(np.diag(np.ones(horizon-1),-1), -model.A)
    eqs_A2 = np.kron(np.eye(horizon), -model.B)
    eqs_A = np.concatenate([eqs_A1, eqs_A2], axis=1)
    eqs_B = np.expand_dims(\
            np.ndarray.flatten(np.array([model.D]*horizon)), axis=-1)

    max_angle = pi/4
    state_dev = 0.5

    # steering angle constraints
    ineq1_A = np.concatenate([np.zeros((2*horizon,state_size*horizon)),\
            np.concatenate([np.eye(horizon),-1*np.eye(horizon)], axis=0)], axis=1)

    ineq1_B = np.concatenate([max_angle*np.ones((horizon,1)), \
            max_angle*np.ones((horizon,1))], axis=0)

    # state constraint
    ineq2_A = np.concatenate([np.kron(np.eye(horizon), \
            np.eye(state_size)), np.zeros((state_size*horizon,horizon))], axis=1)
    ineq2_A = np.concatenate([-ineq2_A,ineq2_A], axis=0)
    ineq2_B = np.concatenate([reference_trajectory[:state_size*horizon]-state_dev,\
            state_dev+reference_trajectory[:state_size*horizon]], axis=0)

    ineqs_A = np.concatenate([ineq1_A, ineq2_A], axis=0)
    ineqs_B = np.concatenate([ineq1_B, ineq2_B], axis=0)

    ineqs_A = ineq1_A
    ineqs_B = ineq1_B


    reference_trajectory = np.ndarray.flatten(reference_trajectory)

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
    """
    cons = {'type':'eq',
        'fun':lambda x: np.ndarray.flatten(eqs_B) - np.dot(eqs_A,x),
        'jac':lambda x: -eqs_A}
    """

    opt = {'disp':False}

    x0 = np.zeros(state_size*horizon+control_size*horizon)
    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)
    print(res_cons)
    #print(res_cons.x)
    #print(reference_trajectory)
    control_sequence = res_cons.x[state_size*horizon:]
    print(control_sequence)

    model_2d = Bicycle_Model(0.3, 1/frequency, speed,linear=False, dim=3)

    non_linear_state = []
    for u in range(len(control_sequence)):
        model_2d.update(control_sequence[u])
        non_linear_state.append(model_2d.X.tolist())


    state=np.array(non_linear_state)
    plt.plot(state[:,0], state[:,1])
    plt.show()




    











