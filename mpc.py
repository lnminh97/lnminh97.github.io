import numpy as np
from casadi import *
import do_mpc

def mpc(model, t_samp, ref):
	# Create mpc with model
	mpc = do_mpc.controller.MPC(model)

	setup_mpc = {
	    'n_robust': 3,
	    'n_horizon': 7,
	    't_step': t_samp,
	    'state_discretization': 'discrete',
	    # Use MA27 linear solver in ipopt for faster calculations:
	    'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
	}

	mpc.set_param(**setup_mpc)

	# cost
	mterm = 100 * (model.x['y'] - ref)**2
	lterm = 100 * (model.x['y'] - ref)**2

	mpc.set_objective(mterm=mterm, lterm=lterm)

	# mpc.set_rterm(u=1e-2) # input penalty

	# lower bounds of the input
	mpc.bounds['lower','_u','u'] = 0

	# upper bounds of the input
	mpc.bounds['upper','_u','u'] = 1

	# lowe bounds of the output
	mpc.bounds['lower', '_x', 'y'] = 0

	mpc.setup()

	return mpc
