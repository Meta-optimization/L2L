"""
Definition of the prior
"""

import torch
from sbi import utils

# limits = [(-0.48, -0.06),
#              (0.0, 540.0),
#              (0.6, 1.0),
#              (2.2, 5.9),
#              (67.0, 400.0),
#              (110.0, 400.0)]

limits = [(-0.3300743862429623, -0.3300743862429623),
             (0.5700579512634254, 0.5700579512634254),
             (0.7041337553279835, 0.7041337553279835),
             (3.4583848037533604, 3.4583848037533604),
             (68.3, 68.3),
             (266.3, 266.3)]

prior = utils.BoxUniform(low=torch.Tensor([q[0] for q in limits]),
                         high=torch.Tensor([q[1] for q in limits]))

labels = ['inh_weight', 'noise_rate', 'th_wmax',
                   'cx_wmax', 't_img_ms', 't_pause_ms']
x_labels = ['raw_accuracy', 'contrast', 'first_spike_accuracy']

x_obs = (0.9, # raw accuracy
         0.5, # contrast
         0.9) # first spike accuracy
