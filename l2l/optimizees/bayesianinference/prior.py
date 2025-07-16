"""
Definition of the prior
"""

import torch
from sbi import utils

prior = utils.BoxUniform(low=torch.Tensor([20.0, -250.0, 0.1, 5000.0, 2000.0]), high=torch.Tensor([54.0, -190.0, 5.0, 10000.0, 2500.0]))
labels = ['w_ex', 'w_in', 'delay', 'c_ex', 'c_in']
x_obs = [9.158100000000001, 0.8936075145163004, 0.8859034223518725, 4.917083663642022]
