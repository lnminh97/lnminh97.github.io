import numpy as np
from casadi import *
import do_mpc

def longitudinal_model():
	# Init discrete model
	model = do_mpc.model.Model('discrete')

	# Define state var x and state input u
	x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
	u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))
	y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))

	# State matrix
	model.A = np.array([[1.9349, -0.9351],
	              [1     ,       0]])

	model.B = np.array([[0.25],
	              [   0]])

	model.C = np.array([[0.2428, -0.2126]])

	x_next = model.A@x + model.B@u
	y      = model.C@x

	model.set_rhs('x', x_next)
	model.set_rhs('y', model.C@x_next)

	# speed = model.set_meas('speed', y)

	model.setup()

	return model