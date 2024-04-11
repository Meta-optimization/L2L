"""
Definition of the prior
"""

import torch
from sbi import utils

prior = utils.BoxUniform(low=torch.Tensor([0.0, -200.0, 0.1, 0.0, 0.0]), high=torch.Tensor([200.0, 0.0, 5.0, 1.0, 1.0]))
