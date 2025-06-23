from l2l.utils.experiment import Experiment
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from l2l.optimizees.community_detection import CommunityOptimizee, CommunityOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
def run_experiment():
    experiment = Experiment(
        root_dir_path='../masterarbeit/community')
    
    runner_params = {
        "srun": "",
        "exec": "python3"
    } 
    traj, _ = experiment.prepare_experiment(
        runner_params=runner_params, name=f"community", overwrite=True, debug=True)

    G = nx.karate_club_graph()
    A = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0]
        ])



    # Graph aus der Matrix erzeugen
    #G = nx.from_numpy_array(A)
    """A = np.genfromtxt(f"/home/hanna/Documents/Meta-optimization/BrainNetViewer_20191031/Data/ExampleFiles/AAL90/Edge_AAL90_Binary.edge",
                      delimiter='	')
    G = nx.from_numpy_array(A)"""
    #TVB
    """A = np.genfromtxt(f"/home/hanna/Documents/tvb_data/tvb_data/connectivity/connectivity_66/weights.txt",
                      delimiter=' ')
    binary_matrix = (A != 0).astype(int)
    G = nx.from_numpy_array(binary_matrix)"""
    A = np.genfromtxt(f"/home/hanna/Downloads/mnorm_H1_left")

    G = nx.from_numpy_array(A)

    optimizee_parameters = CommunityOptimizeeParameters(APIToken='test', 
                                                            config_path='./dwave', 
                                                            num_partitions=4.0,
                                                            num_reads=100.0,
                                                            one_hot_strength=3.0,
                                                            Graph = G, 
                                                            result_path='./community/qpu_UK')
    optimizee = CommunityOptimizee(traj, optimizee_parameters)


    # Genetic Algorithm Optimizer
    optimizer_parameters = GeneticAlgorithmParameters(seed=15, 
                                                      pop_size=2,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=5,
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
