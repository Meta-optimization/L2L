import unittest

import numpy as np
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.utils.experiment import Experiment

from l2l import list_to_dict


class GDOptimizerTestCase(OptimizerTestCase):

    def test_gd(self):
        optimizer_parameters = RMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                       n_random_steps=1, momentum_decay=0.5,
                                       n_iteration=1, stop_criterion=np.inf, seed=99)

        #test with function generator optimizee
        optimizer = GradientDescentOptimizer(self.trajectory_functionGenerator,
                                             optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                             optimizee_fitness_weights=(0.1,),
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
        print(self.experiment_functionGenerator.optimizer)
        best = list_to_dict(self.experiment_functionGenerator.optimizer.current_individual.tolist(),
                             self.experiment_functionGenerator.optimizer.optimizee_individual_dict_spec)['coords']
        self.assertEqual(best[0],-4.998856251826551)
        self.assertEqual(best[1],-1.9766742736816023)
        self.experiment_functionGenerator.end_experiment(optimizer)

        #test with active wait opimizee
        optimizer = GradientDescentOptimizer(self.trajectory_activeWait,
                                             optimizee_create_individual=self.optimizee_activeWait.create_individual,
                                             optimizee_fitness_weights=(0.1,),
                                             parameters=optimizer_parameters,
                                             optimizee_bounding_func=self.optimizee_activeWait.bounding_func)
        try:

            self.experiment_activeWait.run_experiment(optimizee=self.optimizee_activeWait,
                                           optimizee_parameters=self.optimizee_activeWait_parameters,
                                           optimizer=optimizer,
                                           optimizer_parameters=optimizer_parameters)
        except Exception as e:
            self.fail(Exception.__name__)

        best = list_to_dict(self.experiment_activeWait.optimizer.current_individual.tolist(),
                             self.experiment_activeWait.optimizer.optimizee_individual_dict_spec)['difficulty']
        self.assertEqual(best, 10000)
        self.experiment_activeWait.end_experiment(optimizer)


def suite():
    suite = unittest.makeSuite(GDOptimizerTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())


if __name__ == "__main__":
    run()