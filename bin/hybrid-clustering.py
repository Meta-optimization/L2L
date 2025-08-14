from l2l.utils.experiment import Experiment
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from l2l.optimizees.clustering import HybridClusteringOptimizee, HybridClusteringOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
def run_experiment():
    experiment = Experiment(
        root_dir_path='..')
    
    runner_params = {
        "srun": "",
        "exec": "python3"
    } 
    traj, _ = experiment.prepare_experiment(
        runner_params=runner_params, name=f"cluster", overwrite=True, debug=True)

    #scattered_points
    num_clusters = 3
    X, y = make_blobs(n_samples=30, centers=num_clusters, cluster_std=1, random_state=42)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.show()

    optimizee_parameters = HybridClusteringOptimizeeParameters(APIToken='test', 
                                                         config_path='./dwave', 
                                                         alpha = 2.0,
                                                         gamma = 4.0,
                                                         delta=3.0,
                                                         one_hot_strength=100.0,
                                                         points=X, 
                                                         num_clusters=num_clusters, 
                                                         result_path='path/to/results')
    optimizee = HybridClusteringOptimizee(traj, optimizee_parameters)


    # Genetic Algorithm Optimizer
    optimizer_parameters = GeneticAlgorithmParameters(seed=15, 
                                                      pop_size=2,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=2,
                                                      ind_prob=0.45,
                                                      tourn_size=4,
                                                      mate_par=0.5,
                                                      mut_par=5
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
