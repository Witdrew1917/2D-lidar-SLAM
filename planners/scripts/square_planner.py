#!/usr/bin/env python
import argparse
import numpy as np
from scipy import optimize
from math import pi

import rospy
from planners.msg import AckermannDrive, AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

import geometric_trajectories
from odometric_models import Bicycle_Model

"""
    This script solves an MPC formulated optimization problem that is based
    of the constraints and properties of the ADEPT racecar.

    Reference trajectories can by set by changing the argument --geometry
    to any of the classes implemented under ../modules/geometric_trajectoris.py

    The bicycle model is the only option of model that this controller can
    use but can easily be extended by following the implementation of said 
    model.
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--geometry', type=str, default='Square')
    parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")

    args = parser.parse_args()

    Trajectory = getattr(geometric_trajectories, args.geometry)

    rospy.init_node('square_planner', anonymous=True)
    print("solving MPC problem")
    speed=0.2
    frequency=10.0
    circumference=4.0
    wheelbase = 0.3
    state_size = 1
    control_size = 1

    time = circumference/speed 
    shape = Trajectory(circumference, time, frequency)

    model = Bicycle_Model(wheelbase, 1/frequency, speed, dim=state_size)

    horizon = int(time*frequency)
    reference_trajectory = np.zeros(((state_size+control_size)*horizon,1))
    if state_size == 1:
        reference_trajectory[:state_size*horizon,0] = \
                shape.trajectory[:horizon,2].reshape(-1)
    else:
        reference_trajectory[:state_size*horizon,0] = \
                shape.trajectory[:horizon,:2].reshape(-1)
    reference_trajectory[state_size*horizon:,0] = \
            shape.trajectory[:horizon,-1].reshape(-1)
    


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
    # solve MPC problem
    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)
    control_sequence = res_cons.x[state_size*horizon:]
    model_2d = Bicycle_Model(0.3, 1/frequency, speed,linear=False, dim=3)
    non_linear_state = []

    # ros initializations
    rate = rospy.Rate(frequency)
    publisher_control_action = rospy.Publisher(\
            "low_level/ackermann_cmd_mux/input/teleop", AckermannDriveStamped, queue_size=1)
    publisher_odometry = rospy.Publisher("odometry", Float32MultiArray,\
            queue_size=1)


    for u in range(len(control_sequence)):
        model_2d.update(control_sequence[u])
        non_linear_state.append(np.ndarray.flatten(model_2d.X).tolist())
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive(control_sequence[u].item(),
                                         0, speed, 0, 0)
        odometry_msg = Float32MultiArray()
        odometry_msg.data = non_linear_state[-1]

        publisher_control_action.publish(drive_msg)
        publisher_odometry.publish(odometry_msg)
        rate.sleep()

    if args.verbose:
        import matplotlib.pyplot as plt
        plt.plot(reference_trajectory)
        


