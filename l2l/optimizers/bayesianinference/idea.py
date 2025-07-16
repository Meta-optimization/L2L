#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:06:04 2025

@author: todt
"""

optimizer_parameters = SBIOptimizerParameters(pop_size=500, restrict_prior=5, n_iteration=8, seed=0, save_path='/home/todt/Dokumente/L2L/results/data',
                                              inference_method=SNPE, x_obs=[10.], tensorboard=True)

plan = [LoadData('...'),
        SimulateData(n=1000),
        RestrictPrior(load='...'),
        SimulateData(n=5000),
        SimulateData(n=5000),
        RunInference(method=NPE, name='npe'),
        RunInference(method=NLE, name='nle'),
        DoPPC('npe', x_obs=[...], n=1000),
        DoPPC('nle', x_obs=[...], n=1000)] # => n_iteration = 5

optimizer_parameters = SBIOptimizerParameters(plan=plan, seed=0, save_path='/home/todt/Dokumente/L2L/results/data', tensorboard=True)

plan = [LoadModel('...', name='nle'),
        DoPPC(x_obs=[...], n=1000)]

plan = [LoadModel('...', name='nle'),
        DoSBC(n=1000)]

plan = [SimulateData(10000, prior=prior), # prior?
        RunInference(NPE, 'npe', subset=['rate_ex']),
        SimulateData(10000, prior='npe'),
        RunInference('npe', x_obs=[...]),
        SimulateData(10000, prior='npe'),
        RunInference(),
        DoPPC()]

# train on subset of data?
