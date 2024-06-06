import unittest

import numpy as np
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters


class CEOptimizerTestCase(OptimizerTestCase):

    def test_setup(self):

        optimizer_parameters = CrossEntropyParameters(
            pop_size=2,
            rho=0.9,
            smoothing=0.0,
            temp_decay=0,
            n_iteration=1,
            distribution=NoisyGaussian(noise_magnitude=1.0, noise_decay=0.99),
            stop_criterion=np.inf,
            seed=1,
        )
        optimizer = CrossEntropyOptimizer(
            self.trajectory_functionGenerator,
            optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
            optimizee_fitness_weights=(-0.1,),
            parameters=optimizer_parameters,
            optimizee_bounding_func=self.optimizee_functionGenerator.bounding_func,
        )

        self.assertIsNotNone(optimizer.parameters)
        self.assertIsNotNone(self.experiment_functionGenerator)

        try:

            self.experiment_functionGenerator.run_experiment(
                optimizee=self.optimizee_functionGenerator,
                optimizee_parameters=self.optimizee_functionGenerator_parameters,
                optimizer=optimizer,
                optimizer_parameters=optimizer_parameters,
            )
        except Exception as e:
            self.fail(e.__name__)
        best = self.experiment_functionGenerator.optimizer.best_individual["coords"]
        self.assertEqual(best[0], -4.998856251826551)
        self.assertEqual(best[1], -1.9766742736816023)
        self.experiment_functionGenerator.end_experiment(optimizer)

        # test with active wait opimizee
        optimizer_parameters = CrossEntropyParameters(
            pop_size=3,
            rho=0.9,
            smoothing=0.0,
            temp_decay=0,
            n_iteration=1,
            distribution=NoisyGaussian(noise_magnitude=1.0, noise_decay=0.99),
            stop_criterion=np.inf,
            seed=1,
        )

        self.assertRaises(Exception, lambda: {
            CrossEntropyOptimizer(
                self.trajectory_activeWait,
                optimizee_create_individual=self.optimizee_activeWait.create_individual,
                optimizee_fitness_weights=(-0.1,),
                parameters=optimizer_parameters,
                optimizee_bounding_func=self.optimizee_activeWait.bounding_func,
            )
        })


def suite():
    suite = unittest.makeSuite(CEOptimizerTestCase, "test")
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
