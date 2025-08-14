import os

import yaml
import numpy as np

import pickle
import networkx as nx 

from l2l.optimizees.community_detection import CommunityOptimizee, CommunityOptimizeeParameters
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.crossentropy.distribution import NoisyGaussian

from l2l.utils.experiment import Experiment

def main():
    experiment = Experiment(root_dir_path='path')
    name = 'checkpoint_ce'
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

    #Cross_Entropie Optimizee
    optimizer_seed = 1234
    optimizer_parameters = CrossEntropyParameters(pop_size=20, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=16,
                                                  distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                                  stop_criterion=np.inf, seed=optimizer_seed)


    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(1.,),
                                      parameters=optimizer_parameters,
                                      optimizee_bounding_func=optimizee.bounding_func)
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters)
    experiment.end_experiment(optimizer)

if __name__ == '__main__':
    main()
