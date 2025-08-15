import logging.config
import os
import pickle
import shutil

from l2l.utils.environment import Environment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.paths import Paths
import l2l.utils.runner as runner


class Experiment(object):
    def __init__(self, root_dir_path):
        """
        Prepares and starts the l2l simulation.

        For an example see `L2L/bin/l2l-template.py`

        :param root_dir_path: str, Path to the results folder. Accepts relative
        paths. Will check if the folder exists and create if not.
        """
        self.root_dir_path = os.path.abspath(root_dir_path)
        self.logger = logging.getLogger('utils.experiment')
        self.paths = None
        self.env = None
        self.traj = None
        self.optimizee = None
        self.optimizer = None


    def prepare_experiment(self, **kwargs):
        """
        Prepare the experiment by creating the enviroment and
        :param kwargs: optional dictionary, contains
            - name: str, name of the run, Default: L2L-run
            - trajectory_name: str, name of the trajectory, Default: trajectory
            - checkpoint: object, trajectory object
            - log_stdout: bool, if stdout should be sent to logs, Default:False
            - runner_params: dict, User specified parameters for the runner.
                See notes section for default runner parameters
            - multiprocessing, bool, enable multiprocessing, Default: False
            - debug, bool, enable verbose mode to get more detailed logs for debugging,
                 Default: False
            - stop_run, bool, when an error occures the execution is stoped, Default: False
            - timeout, bool, stops execution after 2 hours if it is not finished by then,
                Default: True
            -overwrite, bool, specifies whether existing files should be overwritten
                Default: False
        :return traj, trajectory object
        :return all_runner_params, dict, a dictionary with all parameters for the runner
            given by the user and default ones

        :notes
           Default runner parameters are:
            - srun: ""
            - exec: 'python3 + "self.paths.simulation_path/run_optimizee.py"'
            - max_workers: 32
            - work_path: self.paths.root_dir_path,
            - paths_obj: self.paths
        """
        name = kwargs.get('name', 'L2L-run')
        if not os.path.isdir(self.root_dir_path):
            os.mkdir(os.path.abspath(self.root_dir_path))
            print('Created a folder at {}'.format(self.root_dir_path))

        if('checkpoint' in kwargs):
            self.traj = kwargs['checkpoint']
            trajectory_name = self.traj._name
        else:
            trajectory_name = kwargs.get('trajectory_name', 'trajectory')

        self.paths = Paths(name, {},
                           root_dir_path=self.root_dir_path,
                           suffix="-" + trajectory_name)

        overwrite = kwargs.get('overwrite', False)
        if os.path.isdir(self.paths.output_dir_path):
            if overwrite:
                ready_path = 'simulation/ready_files'
                if os.path.isdir(os.path.join(self.paths.output_dir_path, ready_path)):
                    shutil.rmtree(os.path.join(self.paths.output_dir_path, ready_path))
            else:
                raise Exception("There are already exsiting outputfiles in this directory. Please change the path specification.")

        print("All output logs can be found in directory ",
              self.paths.logs_path)

        # Create an environment that handles running our simulation
        # This initializes an environment
        if self.traj:
            self.env = Environment(
                checkpoint=self.traj,
                filename=self.paths.output_dir_path,
                file_title='{} data'.format(name),
                comment='{} data'.format(name),
                add_time=True,
                automatic_storing=True,
                log_stdout=kwargs.get('log_stdout', False),  # Sends stdout to logs
                multiprocessing=kwargs.get('multiprocessing', True),
                debug = kwargs.get('debug', False),
                stop_run = kwargs.get('stop_run',False),
                timeout = kwargs.get('timeout', True)
            )
        else:
            self.env = Environment(
                trajectory=trajectory_name,
                filename=self.paths.output_dir_path,
                file_title='{} data'.format(name),
                comment='{} data'.format(name),
                add_time=True,
                automatic_storing=True,
                log_stdout=kwargs.get('log_stdout', False),  # Sends stdout to logs
                multiprocessing=kwargs.get('multiprocessing', True),
                debug = kwargs.get('debug', False),
                stop_run = kwargs.get('stop_run', False),
                timeout = kwargs.get('timeout', True)
            )
            # Get the trajectory from the environment
            self.traj = self.env.trajectory

        create_shared_logger_data(
            logger_names=['optimizers', 'utils'],
            log_levels=['INFO', 'INFO'],
            log_to_consoles=[True, True],
            sim_name=name,
            log_directory=self.paths.logs_path)
        configure_loggers()


        default_runner_params = {
            "srun": "",
            "exec": 'python3 "' + os.path.join(self.paths.simulation_path, "run_optimizee.py") + '"',
            "max_workers": 32,
            "work_path": self.paths.root_dir_path,
            "paths_obj": self.paths,
        }

        # Will contain all runner parameters
        all_runner_params = {}
        self.traj.f_add_parameter_group("runner_params",
                                        "Contains runner parameters")



        # Go through the parameter dictionary and add to the trajectory
        if kwargs.get('runner_params'):
            for k, v in kwargs['runner_params'].items():
                if k == "exec":
                    val = v + ' "' + os.path.join(self.paths.simulation_path,
                                                 "run_optimizee.py") + '"'
                    self.traj.f_add_parameter_to_group("runner_params", k, val)
                    all_runner_params[k] = val
                else:
                    self.traj.f_add_parameter_to_group("runner_params", k, v)
                    all_runner_params[k] = v

        # Default parameters are added if they are not already set by the user
        for k, v in default_runner_params.items():
            if kwargs.get('runner_params'):
                if k not in kwargs.get('runner_params').keys():
                    self.traj.f_add_parameter_to_group("runner_params", k, v)
                    all_runner_params[k] = v
                    if k == "max_workers":
                        self.logger.info(f"No parameter \'max_workers\' given to runner. Using default value {v}.")

            else:
                self.traj.f_add_parameter_to_group("runner_params", k, v)
                all_runner_params[k] = v




        print('Runner parameters used: {}'.format(all_runner_params))
        return self.traj, all_runner_params



    def run_experiment(self, optimizer, optimizee,
                       optimizer_parameters=None, optimizee_parameters=None):
        """
        Runs the simulation with all parameter combinations

        Optimizee and optimizer object are required as well as their parameters
        as namedtuples.

        :param optimizee: optimizee object
        :param optimizee_parameters: Namedtuple, optional, parameters of the optimizee
        :param optimizer: optimizer object
        :param optimizer_parameters: Namedtuple, optional, parameters of the optimizer
        """
        ind = optimizee.create_individual()
        for key in ind:
            if(isinstance(ind[key], int)):
                raise ValueError('Parameter of type integer is not allowed')
        self.optimizee = optimizee
        self.optimizer = optimizer
        self.optimizer = optimizer
        self.logger.info("Optimizee parameters: %s", optimizee_parameters)
        self.logger.info("Optimizer parameters: %s", optimizer_parameters)
        runner.prepare_optimizee(optimizee, self.paths.simulation_path)
        # Add post processing
        self.env.add_postprocessing(optimizer.post_process)
        # Run the simulation
        self.env.run()

    def end_experiment(self, optimizer):
        """
        Ends the experiment and disables the logging

        :param optimizer: optimizer object
        :return traj, trajectory object
        :return path, Path object
        """
        # Outer-loop optimizer end
        optimizer.end(self.traj)
        # Finally disable logging and close all log-files
        self.env.disable_logging()
        return self.traj, self.paths

    def load_trajectory(self, traj_path):
        """
        Loads a trajectory from a given file
        :param traj_path: path to the trajectory file
        :return traj: trajectory object
        """
        traj_file = open(os.path.join(traj_path),
                          "rb")
        loaded_traj = pickle.load(traj_file)
        traj_file.close()
        return loaded_traj

