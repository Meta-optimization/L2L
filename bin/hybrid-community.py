from l2l.utils.experiment import Experiment
import networkx as nx
import numpy as np
from l2l.optimizees.community_detection import HybridCommunityOptimizee, HybridCommunityOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
def run_experiment():
    experiment = Experiment(
        root_dir_path='..')
    
    runner_params = {
        "srun": "",
        "exec": "python3"
    } 
    traj, _ = experiment.prepare_experiment(
        runner_params=runner_params, name=f"community_detection_hybrid_ga", overwrite=True, debug=True)

    A = np.genfromtxt(f"connectivity_matrix")

    G = nx.from_numpy_array(A)

    optimizee_parameters = HybridCommunityOptimizeeParameters(APIToken='wEdq-d95ea6c975496e423e1e52a09aad0389ff90e336', 
                                                            config_path='./dwave', 
                                                            num_partitions=5.0,
                                                            one_hot_strength=2.0,
                                                            Graph = G, 
                                                            weight = 'weight',
                                                            result_path='path/to/results')
    optimizee = HybridCommunityOptimizee(traj, optimizee_parameters)


    # Genetic Algorithm Optimizer
    optimizer_parameters = GeneticAlgorithmParameters(seed=15, 
                                                      pop_size=2,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=2,
                                                      ind_prob=0.45,
                                                      tourn_size=4,
                                                      mate_par=0.5,
                                                      mut_par=1
                                                      )
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
