# This class predict model in runtime

import numpy as np

LAMDA = 0.985

class model():
    """docstring for model"""
    def __init__(self, phi, theta, lamda=LAMDA, P_k_1=None):
        self.phi   = phi
        self.theta = theta
        self.lamda = lamda
        self.P_k_1 = P_k_1

    def get_P_k_1(self):
        return self.P_k_1

    def get_theta(self):
        return self.theta

    def make_control(self, y_k, P_k_1):
        phi_k     = self.phi
        theta_k_1 = self.theta
        lamda     = self.lamda

        P_k = P_k_1 - (P_k_1 @ phi_k @ phi_k.T @ P_k_1) / (lamda + phi_k.T @ P_k_1 @ phi_k)
        L_k = P_k_1 @ phi_k / (lamda + phi_k.T @ P_k_1 @ phi_k)
        e_k = y_k - phi_k.T @ theta_k_1

        self.theta = theta_k_1 + L_k * e_k          # theta_k
        self.P_k_1 = P_k


        
