import numpy as np
from casadi import *
import do_mpc

def mhe(model, t_samp):
	# Create mpc with model
	mhe = do_mpc.estimator.MHE(model, ['x'])

	setup_mhe = {
	    'n_horizon': 3,
	    't_step': t_samp,
	    'state_discretization': 'discrete',
	    'store_full_solution': True,
	    'meas_from_data': True,
	    # Use MA27 linear solver in ipopt for faster calculations:
	    # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
	}

	mhe.set_param(**setup_mhe)

	P_v = np.eye(1)
	P_x = np.eye(3)

	mhe.set_default_objective(P_x, P_v)

	mhe.setup()

	return mhe
