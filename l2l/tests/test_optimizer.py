import unittest
import os

from l2l.optimizees.active_wait.optimizee_aw import AWOptimizee, AWOptimizeeParameters
from l2l.utils.experiment import Experiment

from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from collections import namedtuple


class OptimizerTestCase(unittest.TestCase):

    def setUp(self):
        # Test function
        function_id = 14
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)
        home_path =  os.environ.get("HOME")
        root_dir_path = os.path.join(home_path, 'results')

        #set up funtiongenerator optimizee
        self.experiment_functionGenerator = Experiment(root_dir_path=root_dir_path)
        jube_params = {}
        self.trajectory_functionGenerator, all_jube_params = self.experiment_functionGenerator.prepare_experiment(name='L2L',
                                                                              log_stdout=True,
                                                                              jube_parameter=jube_params,
                                                                              overwrite=True)
        self.optimizee_functionGenerator_parameters = namedtuple('OptimizeeParameters', [])
        self.optimizee_functionGenerator = FunctionGeneratorOptimizee(
            self.trajectory_functionGenerator, benchmark_function, seed=1)
        
        #set up activeWait optimizee
        self.experiment_activeWait = Experiment(root_dir_path=root_dir_path)
        self.trajectory_activeWait, all_jube_params = self.experiment_activeWait.prepare_experiment(name='L2L',
                                                                              log_stdout=True,
                                                                              jube_parameter=jube_params,
                                                                              overwrite=True)
        self.optimizee_activeWait_parameters = AWOptimizeeParameters(difficulty=10000.0)
        self.optimizee_activeWait = AWOptimizee(self.trajectory_activeWait, self.optimizee_activeWait_parameters)
