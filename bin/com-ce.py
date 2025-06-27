from l2l.utils.experiment import Experiment
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from l2l.optimizees.community_detection import CommunityOptimizee, CommunityOptimizeeParameters
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.crossentropy.distribution import NoisyGaussian

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

    optimizee_parameters = CommunityOptimizeeParameters(APIToken='test', 
                                                            config_path='./dwave', 
                                                            num_partitions=3.0,
                                                            num_reads=50.0,
                                                            one_hot_strength=5.0,
                                                            Graph = G, 
                                                            weight = None,
                                                            result_path='./community/ce')
    optimizee = CommunityOptimizee(traj, optimizee_parameters)

    #Cross_Entropie Optimizee
    optimizer_seed = 1234
    optimizer_parameters = CrossEntropyParameters(pop_size=2, rho=0.9, smoothing=0.0, temp_decay=0, n_iteration=2,
                                                  distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                                  stop_criterion=np.inf, seed=optimizer_seed)


    optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(1.,),
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
