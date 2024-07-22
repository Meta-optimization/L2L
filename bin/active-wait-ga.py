from l2l.utils.experiment import Experiment
import numpy as np

from l2l.optimizees.active_wait import AWOptimizee, AWOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer


def run_experiment():
    experiment = Experiment(
        root_dir_path='../results')
    
    jube_params = { "exec": "python3.9"} 
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name=f"jube_removal_aw_ga", overwrite=True)
        

    # Active Wait Optimizee
    optimizee_parameters = AWOptimizeeParameters(difficulty=10000.0)
    optimizee = AWOptimizee(traj, optimizee_parameters)


    # Genetic Algorithm Optimizer
    optimizer_parameters = GeneticAlgorithmParameters(seed=1580211, 
                                                      pop_size=32,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=400,
                                                      ind_prob=0.45,
                                                      tourn_size=4,
                                                      mate_par=0.5,
                                                      mut_par=1
                                                      )
    optimizer = GeneticAlgorithmOptimizer(traj, 
                                          optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1,),
                                          parameters=optimizer_parameters)


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
