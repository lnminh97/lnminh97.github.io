# This class implement the Model Reference Adaptive Controller

import numpy as np


class mrac():
    """docstring for mrac"""
    def __init__(self, model, plant):
        # model transfer function represented by a matrix
        # numorator is 0th row
        # denominator is 1st row
        self.model = model
        self.plant = plant
        self.S     = np.zeros(2)

    def calculate_params(self):
        self.S[0] = (self.model[1, 0] - self.plant[1, 0]) / self.plant[0, 0]
        self.S[1] = (self.model[1, 1] - self.plant[1, 1]) / self.plant[0, 0]

    def make_control(self, uc, y, u):
        self.calculate_params()

        output = (np.dot(self.model[0, :], uc) - self.plant[0, 0] * np.dot(self.S, y) - \
                  self.plant[0, 1] * u) / self.plant[0, 0]

        return output
