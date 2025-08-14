import os

import yaml
import numpy as np

import pickle
import networkx as nx 

from l2l.optimizees.community_detection import CommunityOptimizee, CommunityOptimizeeParameters
from l2l.optimizers.simulatedannealing.optimizer import SimulatedAnnealingParameters, SimulatedAnnealingOptimizer, AvailableCoolingSchedules
from l2l.utils.experiment import Experiment

def main():
    experiment = Experiment(root_dir_path='path')
    name = 'checkpoint_sa'
    loaded_traj = experiment.load_trajectory("path/to/trajectory")
    traj, _ = experiment.prepare_experiment(name=name,checkpoint=loaded_traj, log_stdout=True, debug=True, stop_run=True, overwrite=True)
    
    A = np.genfromtxt(f"connectivity_matrix")

    G = nx.from_numpy_array(A)

    optimizee_parameters = CommunityOptimizeeParameters(APIToken='wEdq-d95ea6c975496e423e1e52a09aad0389ff90e336', 
                                                            config_path='./dwave', 
                                                            seed = 42,
                                                            Graph = G, 
                                                            weight = 'weight',
                                                            result_path='path/to/results')

    optimizee = CommunityOptimizee(traj, optimizee_parameters)

     # Simulated Annealing Optimizer
    optimizer_parameters = SimulatedAnnealingParameters(n_parallel_runs=20, noisy_step=.05, temp_decay=.99, n_iteration=1,
                                              stop_criterion=np.inf, seed=np.random.randint(1e5), cooling_schedule=AvailableCoolingSchedules.QUADRATIC_ADDAPTIVE)

    optimizer = SimulatedAnnealingOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                            optimizee_fitness_weights=(-1,),
                                            parameters=optimizer_parameters,
                                            optimizee_bounding_func=optimizee.bounding_func)
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters)
    experiment.end_experiment(optimizer)

if __name__ == '__main__':
    main()
