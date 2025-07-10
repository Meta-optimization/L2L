import os

import numpy as np

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters


from l2l.utils.experiment import Experiment

def main():
    experiment = Experiment(root_dir_path='./results')
    name = 'L2L-FUN-CE-Checkpoint'
    loaded_traj = experiment.load_trajectory("/home/hanna/Documents/Meta-optimization/L2L/results/L2L-FUN-CE/simulation/trajectories/trajectory_1.bin")
    traj, _ = experiment.prepare_experiment(name=name,checkpoint=loaded_traj, log_stdout=True, debug=True, stop_run=True, overwrite=True)

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
    parameters = CrossEntropyParameters(pop_size=10, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=2,
                                        distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                        stop_criterion=np.inf, seed=102)
    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(-0.1,),
                                      parameters=parameters,
                                      optimizee_bounding_func=optimizee.bounding_func)
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=parameters)
    experiment.end_experiment(optimizer)

if __name__ == '__main__':
    main()
