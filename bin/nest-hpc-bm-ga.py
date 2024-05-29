from l2l.utils.experiment import Experiment
import numpy as np

from l2l.optimizees.nest_hpc_benchmark import HPCBMOptimizee, HPCBMOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer


def run_experiment():
    experiment = Experiment(
        root_dir_path='../results')
    
    jube_params = { "exec": "python3.9"} 
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name=f"HPCBenchmark_GeneticAlgorithm")
        

    # nest HPC benchmark Optimizee
    optimizee_parameters = HPCBMOptimizeeParameters(scale=0.05,
                                                    nrec=1000
                                                    )
    optimizee = HPCBMOptimizee(traj, optimizee_parameters)


    # Genetic Algorithm Optimizer
    optimizer_parameters = GeneticAlgorithmParameters(seed=1580211, 
                                                      pop_size=10,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=10,
                                                      ind_prob=0.45,
                                                      tourn_size=4,
                                                      mate_par=0.5,
                                                      mut_par=1)
    optimizer = GeneticAlgorithmOptimizer(traj, 
                                          optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1,),
                                          parameters=optimizer_parameters,
                                          optimizee_bounding_func=optimizee.bounding_func)


    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    run_experiment()



if __name__ == '__main__':
    main()
