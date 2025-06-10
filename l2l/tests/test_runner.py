import unittest
<<<<<<< HEAD
=======
import os
>>>>>>> master

import numpy as np
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters
from l2l.optimizees.test_cases.optimizee_testcase import TestcaseOptimizee, TestcaseOptimizeeParameters 
<<<<<<< HEAD
=======
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.utils.experiment import Experiment
>>>>>>> master

class RunnerTestCase(OptimizerTestCase):

    def test_setup(self):
        #TODO test restarting individuals
        # use gradient decent for testing
        optimizee_parameters = TestcaseOptimizeeParameters(exit_code=129)
        optimizer_parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                       n_random_steps=1, momentum_decay=0.5,
                                       n_iteration=1, stop_criterion=np.inf, seed=99)
        #test with function generator optimizee
        optimizee = TestcaseOptimizee(self.trajectory_functionGenerator, optimizee_parameters)
        optimizer = GradientDescentOptimizer(self.trajectory_functionGenerator,
                                             optimizee_create_individual=optimizee.create_individual,
                                             optimizee_fitness_weights=(0.1,),
                                             parameters=optimizer_parameters,
                                             optimizee_bounding_func=optimizee.bounding_func)
        try:
            self.experiment_functionGenerator.run_experiment(optimizee=optimizee,
                                    optimizee_parameters=optimizee_parameters,
                                    optimizer=optimizer,
                                    optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(f"Error in Runner Test. Massage: {e}")
        self.experiment_functionGenerator.end_experiment(optimizer)

        #test if executions stops, when stop_run = True
        optimizee = TestcaseOptimizee(self.trajectory_stop_error, optimizee_parameters)
        optimizer = GradientDescentOptimizer(self.trajectory_stop_error,
                                             optimizee_create_individual=optimizee.create_individual,
                                             optimizee_fitness_weights=(0.1,),
                                             parameters=optimizer_parameters,
                                             optimizee_bounding_func=optimizee.bounding_func)
        self.assertRaises(SystemExit, lambda: self.experiment_stop_error.run_experiment(optimizee=optimizee,
                                    optimizee_parameters=optimizee_parameters,
                                    optimizer=optimizer,
                                    optimizer_parameters=optimizer_parameters))
<<<<<<< HEAD
=======
        
        #test if runner can handle spaces in path name
        home_path =  os.environ.get("HOME")
        self.root_dir_path = os.path.join(home_path, ' results')
        runner_params = {}
        self.experiment = Experiment(root_dir_path=self.root_dir_path)
        self.trajectory, _ = self.experiment.prepare_experiment(
                name='test_trajectory',
                log_stdout=True,
                add_time=True,
                automatic_storing=True,
                runner_params=runner_params,
                overwrite = True)

        ## Benchmark function
        function_id = 4
        bench_functs = BenchmarkedFunctions()
        (benchmark_name, benchmark_function), benchmark_parameters = \
            bench_functs.get_function_by_index(function_id, noise=True)

        optimizee_seed = 100
        random_state = np.random.RandomState(seed=optimizee_seed)

        ## Innerloop simulator
        optimizee = FunctionGeneratorOptimizee(self.trajectory, benchmark_function,
                                            seed=optimizee_seed)
    
        optimizer_parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                       n_random_steps=1, momentum_decay=0.5,
                                       n_iteration=1, stop_criterion=np.inf, seed=99)

        #test with function generator optimizee
        optimizer = GradientDescentOptimizer(self.trajectory,
                                             optimizee_create_individual=optimizee.create_individual,
                                             optimizee_fitness_weights=(0.1,),
                                             parameters=optimizer_parameters,
                                             optimizee_bounding_func=optimizee.bounding_func)

        self.experiment.run_experiment(optimizee=optimizee,
                                  optimizer=optimizer,
                                  optimizer_parameters=optimizer_parameters)
>>>>>>> master

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(RunnerTestCase)
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
