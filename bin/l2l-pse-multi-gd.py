import os
import warnings
import logging.config
# import yaml

import numpy as np
from l2l.utils.environment import Environment

from l2l.optimizers.multigradientdescent.optimizer import MultiGradientDescentOptimizer
from l2l.optimizers.multigradientdescent.optimizer import MultiRMSPropParameters
from l2l.paths import Paths
from l2l.optimizees.pse_multi.optimizee import PSEOptimizee

import l2l.utils.JUBE_runner as jube
from l2l.utils.experiment import Experiment

#warnings.filterwarnings("ignore")

#logger = logging.getLogger('ltl-pse-multi-gd')


def main():
    name = 'L2L-PSE-MULTI-GD'
    #try:
    #    with open('bin/path.conf') as f:
    #        root_dir_path = f.read().strip()
    #except FileNotFoundError:
    #    raise FileNotFoundError(
    #        "You have not set the root path to store your results."
    #        " Write the path to a path.conf text file in the bin directory"
    #        " before running the simulation"
    #    )
    #paths = Paths(name, dict(run_no='test'), root_dir_path=root_dir_path)

    # with open("logging.yaml") as f:
    #     l_dict = yaml.load(f)
    #     log_output_file = os.path.join(paths.results_path, l_dict['handlers']['file']['filename'])
    #     l_dict['handlers']['file']['filename'] = log_output_file
    #     logging.config.dictConfig(l_dict)
    #
    # print("All output can be found in file ", log_output_file)
    # print("Change the values in logging.yaml to control log level and destination")
    # print("e.g. change the handler to console for the loggers you're interesting in to get output to stdout")

    #traj_file = os.path.join(paths.output_dir_path, 'data.h5')

    # Create an environment that handles running our simulation
    # This initializes a PyPet environment
    #env = Environment(trajectory=name, filename=traj_file, file_title='{} data'.format(name),
    #                  comment='{} data'.format(name),
    #                  add_time=True,
    #                  freeze_input=True,
    #                  multiproc=True,
    #                  #use_scoop=True,
    #                  automatic_storing=True,
    #                  log_stdout=False,  # Sends stdout to logs
    #                  log_folder=os.path.join(paths.output_dir_path, 'logs')
    #                  )

    # Get the trajectory from the environment
    #traj = env.trajectory

    # Set JUBE params
    #traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # The execution command
    # traj.f_add_parameter_to_group("JUBE_params", "exec", "mpirun -n 1 python3 " + root_dir_path +
    #                               "/run_files/run_optimizee.py")

    #traj.f_add_parameter_to_group("JUBE_params", "exec", "srun -n 1 -A slns python3 " + root_dir_path +
                                  #"/LTL-PSE-MULTI-GD/run-no-test/simulation/run_files/run_optimizee.py")
    # Ready file for a generation
    #traj.f_add_parameter_to_group("JUBE_params", "ready_file", root_dir_path + "/readyfiles/ready_w_")
    # Path where the job will be executed
    #traj.f_add_parameter_to_group("JUBE_params", "paths", paths)

    # traj.f_add_parameter_to_group("JUBE_params", "A", 'slns')

    #jube_params = {"exec": "srun -n 2 --exclusive python /p/project/cslns/vandervlag1/L2L/results/LTL-PSE-MULTI-GD/simulation/run_files/run_optimizee.py"}
    #jube_params = {"exec": "srun -n 1 --exclusive python", "nodes": "1", "walltime": "00:10:00", "ppn":"1", "cpu_pp": "1"}
    jube_params = {"exec": "srun -n 1 --exact python"}
    #jube_params={}

    experiment = Experiment(root_dir_path='../results')

    traj, _ = experiment.prepare_experiment(jube_parameter=jube_params, name=name, log_stdout=True)
    

    # NOTE: Innerloop simulator
    optimizee = PSEOptimizee(traj, seed=0)

    # NOTE: Outerloop optimizer initialization
    # n_random_steps controls the number of individuals and thus the number of processes/nodes spawned
    # TODO: Change the optimizer to the appropriate Optimizer class
    parameters = MultiRMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=1, momentum_decay=0.005,
                                   n_iteration=4, stop_criterion=np.Inf, seed=99, n_inner_params=16)

    optimizer = MultiGradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizee=optimizee, optimizee_parameters=None, optimizer=optimizer, optimizer_parameters=parameters) 
    experiment.end_experiment(optimizer)

    # Prepare optimizee for jube runs
    #jube.prepare_optimizee(optimizee, root_dir_path)

    # Add post processing
    #env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    #env.run(optimizee.simulate)

    # NOTE: Outerloop optimizer end
    #optimizer.end(traj)

    # Finally disable logging and close all log-files
    #env.disable_logging()


if __name__ == '__main__':
    main()

