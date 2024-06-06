import unittest

from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.evolution import (
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmParameters,
)


class GAOptimizerTestCase(OptimizerTestCase):

    def test_setup(self):

        # test with function generator opimizee
        optimizer_parameters = GeneticAlgorithmParameters(
            seed=0,
            pop_size=1,
            cx_prob=0.5,
            mut_prob=0.3,
            n_iteration=1,
            ind_prob=0.02,
            tourn_size=1,
            mate_par=0.5,
            mut_par=1,
        )

        optimizer = GeneticAlgorithmOptimizer(
            self.trajectory_functionGenerator,
            optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
            optimizee_fitness_weights=(-0.1,),
            parameters=optimizer_parameters,
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
        optimizer = GeneticAlgorithmOptimizer(
            self.trajectory_activeWait,
            optimizee_create_individual=self.optimizee_activeWait.create_individual,
            optimizee_fitness_weights=(-0.1,),
            parameters=optimizer_parameters,
        )
        try:

            self.experiment_activeWait.run_experiment(
                optimizee=self.optimizee_activeWait,
                optimizee_parameters=self.optimizee_activeWait_parameters,
                optimizer=optimizer,
                optimizer_parameters=optimizer_parameters,
            )
        except Exception as e:
            self.fail(e.__name__)
        best = self.experiment_activeWait.optimizer.best_individual["difficulty"]
        self.assertEqual(best, 10000)
        self.experiment_activeWait.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(GAOptimizerTestCase, "test")
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
