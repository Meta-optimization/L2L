import numpy as np
from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.optimizees.bayesianinference import MultiOptimizee, MultiOptimizeeParameters
from l2l.optimizers.melissa.optimizer import Multi
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters


def main():
    name = 'L2L-FUN-GD'
    experiment = Experiment("../results")
    runner_params = {
        "srun": "",
        "exec": "python",
        "max_workers": 2
    }
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          runner_params=runner_params)

    # Optimizee
    optimizee_parameters = MultiOptimizeeParameters()
    optimizee = MultiOptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=8+7, momentum_decay=0.5, # TODO n_random_steps=1
                                   n_iteration=100, stop_criterion=np.Inf,
                                   seed=99) # TODO n_inner_params = individuals!

    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1.,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    multi_optimizer = Multi(optimizer, traj, 4)

    experiment.run_experiment(optimizer=multi_optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters)

    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
