import os

import yaml
import numpy as np

import pickle

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters

from l2l.utils.experiment import Experiment


def main():
    experiment = Experiment(root_dir_path='./results')
    name = 'L2L-FUN-GA'
    loaded_traj = experiment.load_trajectory("/home/hanna/Documents/Meta-optimization/results/L2L-FUN-GA/simulation/trajectories/trajectory_1_4.bin")
    print(loaded_traj.current_results)
    print("\n")
    traj, _ = experiment.prepare_experiment(name=name,checkpoint=loaded_traj, log_stdout=True, debug=True, stop_run=True)
    print(traj.current_results)
    print("\n")
    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    parameters = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                            mut_prob=0.3, n_iteration=5,
                                            ind_prob=0.02,
                                            tourn_size=15, mate_par=0.5,
                                            mut_par=1
                                            )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=parameters)
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=parameters)
    experiment.end_experiment(optimizer)

    print(traj.current_results)
    print("\n")

if __name__ == '__main__':
    main()
