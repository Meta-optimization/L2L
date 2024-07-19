"""
Definition of the prior
"""

import torch
from sbi import utils

prior = utils.BoxUniform(low=torch.Tensor([0.0, -1000.0, 0.1, 400.0, 50.0]), high=torch.Tensor([200.0, 0.0, 5.0, 500.0, 150.0]))
labels = ['w_ex', 'w_in', 'delay', 'c_ex', 'c_in']
x_obs = [10.]
