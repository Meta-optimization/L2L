import unittest

import numpy as np
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer


class ESOptimizerTestCase(OptimizerTestCase):

    def test_setup(self):

        optimizer_parameters = EvolutionStrategiesParameters(
            learning_rate=0.1,
            noise_std=1.0,
            mirrored_sampling_enabled=True,
            fitness_shaping_enabled=True,
            pop_size=1,
            n_iteration=1,
            stop_criterion=np.inf,
            seed=1)

        optimizer = EvolutionStrategiesOptimizer(
            self.trajectory_functionGenerator,
            optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
            optimizee_fitness_weights=(-1.,),
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
        best = self.experiment_functionGenerator.optimizer.best_individual['coords']
        self.assertEqual(best[0], 0.7945654106889819)
        self.assertEqual(best[1], 1.5914885207715055)
        self.experiment_functionGenerator.end_experiment(optimizer)

        #active wait optimizee
        optimizer = EvolutionStrategiesOptimizer(
            self.trajectory_activeWait,
            optimizee_create_individual=self.optimizee_activeWait.create_individual,
            optimizee_fitness_weights=(-1.,),
            parameters=optimizer_parameters,
            optimizee_bounding_func=self.optimizee_activeWait.bounding_func)

        try:
            self.experiment_activeWait.run_experiment(optimizee=self.optimizee_activeWait,
                                           optimizee_parameters=self.optimizee_activeWait_parameters,
                                           optimizer=optimizer,
                                           optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(e.__name__)
        #best = self.experiment_activeWait.optimizer.best_individual['difficulty']
        #self.assertEqual(best, 10001.624345363663)
        results = self.experiment_activeWait.optimizer.eval_pop_arr
        expected_results = [[10001.624345363663],[9998.375654636337],[10000.0]]
        self.assertListEqual(results.tolist(),expected_results)
        self.experiment_activeWait.end_experiment(optimizer)

def suite():
    suite = unittest.makeSuite(ESOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
