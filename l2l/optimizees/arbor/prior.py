"""
Definition of the prior
"""

import torch
from sbi import utils

prior = utils.BoxUniform(low=torch.Tensor([0.03, 0.02, 0.03, 0.0002,
                                           0.15, 0.5, 0.15, 0.7,
                                           5e-5, 0.002, 1e-6, 0.005]),
                         high=torch.Tensor([0.07, 0.05, 0.065, 0.0006,
                                            0.22, 0.7, 0.22, 0.9,
                                            1.4e-4, 0.005, 5e-6, 0.009]))

labels = ['soma_gbar_NaV', 'axon_gbar_NaV', 'dend_gbar_NaV', 'apic_gbar_NaV',
          'soma_gbar_Kv3_1', 'axon_gbar_Kv3_1', 'dend_gbar_Kv3_1', 'apic_gbar_Kv3_1',
          'soma_gbar_Ca_HVA', 'soma_gbar_Ca_LVA', 'axon_gbar_Ca_HVA', 'axon_gbar_Ca_LVA']

# labels = ['axon_gbar_K_T', 'axon_gbar_Kd', 'axon_gbar_Kv2like','axon_gbar_SK',
#           'soma_gbar_SK', 'soma_gbar_Ih',
#           'apic_gbar_Im_v2', 'apic_gbar_Ih', 'dend_gbar_Im_v2',
#           'dend_gbar_Ih']

x_obs = (-82.6047972031602, # resting potential [mV]
         0.2776949999997837, # initial spike time [s]
         3.333347222227444, # spike frequency/average firing rate [Hz]
         0.2992816666714235, # average inter-spike interval [s]
         -69.29435178117059, # trough height [mV]
         10.566454364014849, # action potential height [mV]
         0.7262500000117456) # action potential width [ms]
