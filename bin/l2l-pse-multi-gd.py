import os
import warnings
import logging.config

import numpy as np
from l2l.utils.environment import Environment

from l2l.optimizers.multigradientdescent.optimizer import MultiGradientDescentOptimizer
from l2l.optimizers.multigradientdescent.optimizer import MultiRMSPropParameters
from l2l.paths import Paths
from l2l.optimizees.pse_multi.optimizee import PSEOptimizee

import l2l.utils.JUBE_runner as jube
from l2l.utils.experiment import Experiment


def main():
    name = 'L2L-PSE-MULTI-GD'
   
    jube_params = {"exec": "srun -n 1 --exact python"}

    experiment = Experiment(root_dir_path='../results')

    traj, _ = experiment.prepare_experiment(jube_parameter=jube_params, name=name, log_stdout=True)
    

    # NOTE: Innerloop simulator
    optimizee = PSEOptimizee(traj, seed=0)

    # NOTE: Outerloop optimizer initialization
    # n_random_steps controls the number of individuals and thus the number of processes/nodes spawned
    parameters = MultiRMSPropParameters(learning_rate=0.1, exploration_step_size=1,
                                   n_random_steps=1, momentum_decay=0.005,
                                   n_iteration=8, stop_criterion=np.Inf, seed=99, n_inner_params=1024)

    optimizer = MultiGradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizee=optimizee, optimizee_parameters=None, optimizer=optimizer, optimizer_parameters=parameters) 
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()

