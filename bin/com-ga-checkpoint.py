import os

import yaml
import numpy as np

import pickle
import networkx as nx 

from l2l.optimizees.community_detection import CommunityOptimizee, CommunityOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters

from l2l.utils.experiment import Experiment

def main():
    experiment = Experiment(root_dir_path='path')
    name = 'checkpoint-ga'
    loaded_traj = experiment.load_trajectory("path/to/trajectory")
    traj, _ = experiment.prepare_experiment(name=name,checkpoint=loaded_traj, log_stdout=True, debug=True, stop_run=True, overwrite=True)
    
    A = np.genfromtxt(f"connectivity_matrix")

    G = nx.from_numpy_array(A)


    optimizee_parameters = CommunityOptimizeeParameters(APIToken='wEdq-d95ea6c975496e423e1e52a09aad0389ff90e336', 
                                                            config_path='./dwave', 
                                                            seed = 42,
                                                            Graph = G, 
                                                            weight = "weight",
                                                            result_path='path/to/results')

    optimizee = CommunityOptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=15, pop_size=20, cx_prob=0.7,
                                            mut_prob=0.7, n_iteration=3,
                                            ind_prob=0.45,
                                            tourn_size=15, mate_par=0.5,
                                            mut_par=1
                                            )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters)
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=parameters)
    experiment.end_experiment(optimizer)

if __name__ == '__main__':
    main()
