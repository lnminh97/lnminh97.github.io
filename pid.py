import numpy as np

class pid():
	"""docstring for pid"""
	def __init__(self, t_samp, P, I=0, D=0):
		self.P 	   	= P
		self.I 	   	= I
		self.D 	   	= D
		self.t_samp = t_samp

	def set_param(self, P=0, I=0, D=0):
		self.P = P
		self.I = I
		self.D = D

	def set_t_samp(self, t_samp):
		self.t_samp = t_samp

	def make_control(self, u_k_1=0, e_k=0, e_k_1=0, e_k_2=0):
		u_k = (self.P + self.I * self.t_samp / 2 + self.D / self.t_samp) * e_k + \
			  (-self.P + self.I * self.t_samp / 2 - 2 * self.D / self.t_samp) * e_k_1 + \
			  (self.D / self.t_samp) * e_k_2 + \
			  u_k_1

		return u_k


