import numpy as np
from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
#from l2l.optimizees.active_wait import AWOptimizee, AWOptimizeeParameters
from l2l.optimizers.multigradientdescent.optimizer import MultiGradientDescentOptimizer
#from l2l.optimizers.multigradientdescent.optimizer import MultiClassicGDParameters
#from l2l.optimizers.multigradientdescent.optimizer import MultiStochasticGDParameters
#from l2l.optimizers.multigradientdescent.optimizer import MultiAdamParameters
from l2l.optimizers.multigradientdescent.optimizer import MultiRMSPropParameters


def main():
    name = 'L2L-FUN-MGD'
    experiment = Experiment("../results")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name, overwrite=True)

    ## Benchmark function
    function_id = 4
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function,
                                           seed=optimizee_seed)

    #optimizee_parameters = AWOptimizeeParameters(difficulty=100.0)
    #optimizee = AWOptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    #parameters = MultiClassicGDParameters(learning_rate=0.01, exploration_step_size=0.01,
    #                                n_random_steps=5, n_iteration=100,
    #                                stop_criterion=np.inf, seed=99, n_inner_params=2)
    #parameters = MultiAdamParameters(learning_rate=0.01, exploration_step_size=0.01, n_random_steps=5, first_order_decay=0.8,
    #                            second_order_decay=0.8, n_iteration=100, stop_criterion=np.inf, seed=99, n_inner_params=2)
    #parameters = MultiStochasticGDParameters(learning_rate=0.01, stochastic_deviation=1, stochastic_decay=0.99,
    #                                     exploration_step_size=0.01, n_random_steps=5, n_iteration=100,
    #                                     stop_criterion=np.inf, seed=99, n_inner_params=2)
    parameters = MultiRMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=2, momentum_decay=0.5,
                                  n_iteration=2, stop_criterion=np.inf, seed=99, n_inner_params=2)

    optimizer = MultiGradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(0.1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters)

    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
