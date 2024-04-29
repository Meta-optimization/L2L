import unittest

from l2l.tests.test_optimizer import OptimizerTestCase

from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters

from l2l import list_to_dict
from l2l.utils.experiment import Experiment


class GSOptimizerTestCase(OptimizerTestCase):

    def test_gd(self):
        n_grid_divs_per_axis = 2
        optimizer_parameters = GridSearchParameters(param_grid={
            'coords': (self.optimizee_functionGenerator.bound[0], self.optimizee_functionGenerator.bound[1], n_grid_divs_per_axis)
        })
        optimizer = GridSearchOptimizer(self.trajectory_functionGenerator, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                        optimizee_fitness_weights=(-0.1,),
                                        parameters=optimizer_parameters)
        self.assertIsNotNone(optimizer.parameters)
        self.assertIsNotNone(self.experiment_functionGenerator)

        try:

            self.experiment_functionGenerator.run_experiment(optimizee=self.optimizee_functionGenerator,
                                           optimizee_parameters=self.optimizee_functionGenerator_parameters,
                                           optimizer=optimizer,
                                           optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(e.__name__)
        print(self.experiment_functionGenerator.optimizer)
        best = self.experiment_functionGenerator.optimizer.best_individual['coords']
        self.assertEqual(best[0], 0)
        self.assertEqual(best[1], 0)
        self.experiment_functionGenerator.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(GSOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()
