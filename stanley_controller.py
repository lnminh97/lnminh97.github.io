# This class implements Stanley Controller to control Car's Lateral Motion

import numpy as np

T_SAMP = 0.033

class stanley_controller():
    """docstring for stanley_controller"""
    def __init__(self, yaw_P, yaw_I, yaw_D, cross_P, cross_I, cross_D, Ks):
        self.yaw_P   = yaw_P
        self.yaw_I   = yaw_I
        self.yaw_D   = yaw_D
        self.cross_P = cross_P
        self.cross_I = cross_I
        self.cross_D = cross_D
        self.Ks      = Ks
        self.u_yaw_previous   = 0
        self.u_cross_previous = 0

    def get_u_previous(self):
        return self.u_yaw_previous, self.u_cross_previous
        
    def make_control(self, u_yaw_previous, u_cross_previous,e_yaw, v, e):
        yaw_controller = pid(T_SAMP, P=self.yaw_P, I=self.yaw_I, D=self.yaw_D)
        u_yaw = yaw_controller.make_control(u_yaw_previous, e_yaw)

        self.u_yaw_previous = u_yaw

        cross_controller = pid(T_SAMP, P=self.cross_P, I=self.cross_I, D=self.cross_D)
        u_cross = cross_controller.make_control(u_cross_previous, e)

        self.u_cross_previous = u_cross

        delta = u_yaw + np.arctan2(u_cross, self.Ks + v)

        return delta

class pid():
    """docstring for pid"""
    def __init__(self, t_samp, P, I=0, D=0):
        self.P      = P
        self.I      = I
        self.D      = D
        self.t_samp = t_samp

    def set_param(self, P=0, I=0, D=0):
        self.P = P
        self.I = I
        self.D = D

    def set_t_samp(self, t_samp):
        self.t_samp = t_samp

    def make_control(self, u_k_1, e):
        e_k   = e[0]
        e_k_1 = e[1]
        if len(e) == 3:
            e_k_2 = e[2]
        else:
            e_k_2 = 0

        u_k = (self.P + self.I * self.t_samp / 2 + self.D / self.t_samp) * e_k + \
              (-self.P + self.I * self.t_samp / 2 - 2 * self.D / self.t_samp) * e_k_1 + \
              (self.D / self.t_samp) * e_k_2 + \
              u_k_1

        return u_k
