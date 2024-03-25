from l2l.utils.experiment import Experiment
import numpy as np

# for optimizees
from l2l.optimizees.active_wait import AWOptimizee, AWOptimizeeParameters


# for ooptimizers
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer

from l2l.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
from l2l.optimizers.crossentropy.distribution import NoisyGaussian




def run_experiment():
    experiment = Experiment(
        #root_dir_path='/p/scratch/cslns/wilhelm2/L2L/results')
        root_dir_path='/mnt/c/Neuro/dataFolder')
    
    #jube_params = { "exec": "srun -n1 --cpus-per-task=1 --gres=gpu:0 --exact python"} 
    jube_params = { "exec": "python3.9"} 
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name=f"aw_GeneticAlgorithm_popSize32_generations400")
        

    # active wait optimizee
    optimizee_parameters = AWOptimizeeParameters(difficulty=5)
    optimizee = AWOptimizee(traj, optimizee_parameters)








    # Optimizer Genetic Algorithm
    optimizer_parameters = GeneticAlgorithmParameters(seed=1580211, 
                                                      pop_size=32, #32
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=400, #6000
                                                      ind_prob=0.45,
                                                      tourn_size=4,
                                                      mate_par=0.5,
                                                      mut_par=1
                                                      )
    optimizer = GeneticAlgorithmOptimizer(traj, 
                                          optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1,),
                                          parameters=optimizer_parameters)
    

    ## Optimizer Cross Entropy
    #optimizer_seed = 1234
    #optimizer_parameters = CrossEntropyParameters(pop_size=32, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=400,
    #                                              distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
    #                                              stop_criterion=np.inf, seed=optimizer_seed)
    #
    #
    #optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                  optimizee_fitness_weights=(1.,),
    #                                  parameters=optimizer_parameters,
    #                                  optimizee_bounding_func=optimizee.bounding_func)






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
