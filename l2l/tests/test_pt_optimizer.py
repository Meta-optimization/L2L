import unittest

from l2l.tests.test_optimizer import OptimizerTestCase
import numpy as np
from l2l.optimizers.paralleltempering.optimizer import AvailableCoolingSchedules
from l2l.optimizers.paralleltempering.optimizer import ParallelTemperingParameters, ParallelTemperingOptimizer


class PTOptimizerTestCase(OptimizerTestCase):

    def test_sa(self):
        cooling_schedules = [AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE,
                             AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE,
                             AvailableCoolingSchedules.EXPONENTIAL_ADDAPTIVE,
                             AvailableCoolingSchedules.LINEAR_ADDAPTIVE,
                             AvailableCoolingSchedules.LINEAR_ADDAPTIVE]

        temperature_bounds = np.array([
            [0.8, 0],
            [0.7, 0],
            [0.6, 0],
            [1, 0.1],
            [0.9, 0.2]])

        decay_parameters = np.full(2, 0.99)
        assert (((temperature_bounds.all() <= 1) and (temperature_bounds.all() >= 0)) and (temperature_bounds[:, 0].all(
        ) > temperature_bounds[:, 1].all())), print("Warning: Temperature bounds are not within specifications.")
        assert ((decay_parameters.all() <= 1) and (decay_parameters.all() >= 0)), print(
            "Warning: Decay parameter not within specifications.")

        optimizer_parameters = ParallelTemperingParameters(n_parallel_runs=2, noisy_step=.03, n_iteration=1,
                                                 stop_criterion=np.inf,
                                                 seed=np.random.randint(1e5), cooling_schedules=cooling_schedules,
                                                 temperature_bounds=temperature_bounds,
                                                 decay_parameters=decay_parameters)
        optimizer = ParallelTemperingOptimizer(self.trajectory_functionGenerator, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                               optimizee_fitness_weights=(-1,),
                                               parameters=optimizer_parameters,
                                               optimizee_bounding_func=self.optimizee_functionGenerator.bounding_func)

        self.assertIsNotNone(optimizer.parameters)
        self.assertIsNotNone(self.experiment_functionGenerator)

        try:

            self.experiment_functionGenerator.run_experiment(optimizee=self.optimizee_functionGenerator,
                                  optimizee_parameters=self.optimizee_functionGenerator_parameters,
                                  optimizer=optimizer,
                                  optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(e.__name__)

        #activeWait optimizee
        optimizer = ParallelTemperingOptimizer(self.trajectory_activeWait, optimizee_create_individual=self.optimizee_activeWait.create_individual,
                                               optimizee_fitness_weights=(-1,),
                                               parameters=optimizer_parameters,
                                               optimizee_bounding_func=self.optimizee_activeWait.bounding_func)

        try:
            self.experiment_activeWait.run_experiment(optimizee=self.optimizee_activeWait,
                                  optimizee_parameters=self.optimizee_activeWait_parameters,
                                  optimizer=optimizer,
                                  optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(e.__name__)

def suite():
    suite = unittest.makeSuite(PTOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()