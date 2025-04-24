from l2l.utils.experiment import Experiment
import numpy as np

from l2l.optimizees.clustering import ClusteringOptimizee, ClusteringOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
from l2l.optimizees.clustering.helpers import Coordinate
def run_experiment():
    experiment = Experiment(
        root_dir_path='../masterarbeit/first-tests')
    
    runner_params = {
        "srun": "",
        "exec": "python3"
    } 
    traj, _ = experiment.prepare_experiment(
        runner_params=runner_params, name=f"cluster", overwrite=True, debug=True)

    scattered_points = [(0, 0), (1, 1), (2, 4), (3, 2)]
    c = [Coordinate(x, y) for x, y in scattered_points]
    optimizee_parameters = ClusteringOptimizeeParameters(APIToken='wEdq-d95ea6c975496e423e1e52a09aad0389ff90e336', path='./dwave', num_reads=100.0, coordinates=c)
    optimizee = ClusteringOptimizee(traj, optimizee_parameters)


    # Genetic Algorithm Optimizer
    optimizer_parameters = GeneticAlgorithmParameters(seed=15, 
                                                      pop_size=2,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=3,
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
