"""
This file is a typical example of a script used to run a L2L experiment. Read the comments in the file for more
explanations
"""
from l2l.utils.experiment import Experiment
from l2l.optimizees.optimizee import Optimizee, OptimizeeParameters
from l2l.optimizers.optimizer import Optimizer, OptimizerParameters


def main():
    # TODO: use  the experiment module to prepare and run later the simulation
    # define a directory to store the results
    experiment = Experiment(root_dir_path='~/home/user/L2L/results')
    # TODO when using the template: use keywords to prepare the experiment and
    #  create a dictionary for runner parameters
    # prepare_experiment returns the trajectory and all runner parameters
    runner_params = {
        "srun": "srun -n 1 -c 50",
        "exec": "python3",
        "max_workers": 32, 
    }

    traj, runner_params = experiment.prepare_experiment(name='L2L',
                                                        log_stdout=True,
                                                        runner_parameter=runner_params,
                                                        # To enable more detailed output for debugging
                                                        # set debug to True
                                                        debug=False,
                                                        # If you do not want to restart the optimizee
                                                        # and the simulation should be stopped
                                                        # set stop_run to True 
                                                        stop_run=False,
                                                        # if you want to overwrite previous results,
                                                        # set overwrite to True, 
                                                        # sotherwise specify a different root_dir_path.
                                                        overwrite= False)

    ## Innerloop simulator
    # TODO when using the template: Change the optimizee to the appropriate
    #  Optimizee class
    optimizee = Optimizee(traj)
    # TODO Create optimizee parameters
    optimizee_parameters = OptimizeeParameters()

    ## Outerloop optimizer initialization
    # TODO when using the template: Change the optimizer to the appropriate
    #  Optimizer class and use the right value for optimizee_fitness_weights.
    #  Length is the number of dimensions of fitness, and negative value
    #  implies minimization and vice versa
    optimizer_parameters = OptimizerParameters()
    optimizer = Optimizer(traj, optimizee.create_individual, (1.0,),
                          optimizer_parameters)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
