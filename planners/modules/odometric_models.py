import numpy as np
from scipy.linalg import expm

class Bicycle_Model:

    def __init__(self, wheelbase, step_size, velocity,\
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

        dAB = expm(self.step_size*np.array([[self.A, self.B],[0.,1.]]))
        self.A = np.expand_dims(dAB[0,0],axis=-1)
        self.B = np.expand_dims(dAB[0,1],axis=-1)
        self.D = 0


    def _create_non_linear_model(self):

        assert self.dim == 3

        self.update = self.non_linear_upadate


    def update(self, U):
        
        self.X = np.dot(self.A,self.X) + self.B * U + self.D

    def non_linear_upadate(self, U):


        A = np.array([self.velocity*np.cos(self.X[2,0]),
                      self.velocity*np.sin(self.X[2,0]),
                      0]).reshape(self.dim,1)

        B = np.array([0,0,self.velocity/self.wheelbase*np.tan(U)]).reshape(
                self.dim, 1)

        self.X += self.step_size*(A + B)
