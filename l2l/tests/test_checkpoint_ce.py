import unittest
import os

import numpy as np
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters


class CheckpointCETestCase(OptimizerTestCase):

    def test_setup(self):

        iterations = 5
        #test checkpointing with function generator optimizee
        optimizer_parameters = CrossEntropyParameters(pop_size=30, rho=0.9, smoothing=0.0, temp_decay=0, 
                                                      n_iteration=iterations,
                                                      distribution=NoisyGaussian(
                                                    noise_magnitude=1., noise_decay=0.99),
                                                      stop_criterion=np.inf, seed=1)
        optimizer = CrossEntropyOptimizer(self.trajectory_functionGenerator, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=optimizer_parameters,
                                          optimizee_bounding_func=self.optimizee_functionGenerator.bounding_func)
        
        #run experiment to create an trajectory which can be loaeded
        self.load_trajectory(optimizer, optimizer_parameters, iterations, self.optimizee_functionGenerator, self.optimizee_functionGenerator_parameters, self.experiment_functionGenerator)

        #created optimizer with loaded trajectory
        optimizer_parameters_checkpoint = CrossEntropyParameters(pop_size=30, rho=0.9, smoothing=0.0, temp_decay=0, 
                                                                n_iteration=1,
                                                                distribution=NoisyGaussian(
                                                                    noise_magnitude=1., noise_decay=0.99),
                                                                stop_criterion=np.inf, seed=1)
        optimizer_checkpoint = CrossEntropyOptimizer(self.trajectory_checkpoint, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=optimizer_parameters_checkpoint,
                                          optimizee_bounding_func=self.optimizee_functionGenerator.bounding_func)
        
        optimizer_parameters_error = CrossEntropyParameters(pop_size=35, rho=0.9, smoothing=0.0, temp_decay=0, 
                                                                n_iteration=1,
                                                                distribution=NoisyGaussian(
                                                                noise_magnitude=1., noise_decay=0.99),
                                                                stop_criterion=np.inf, seed=1)

        #create lambda function to raise an expected error message
        createErrorOptimizer = lambda : {
            CrossEntropyOptimizer(self.trajectory_checkpoint, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                          optimizee_fitness_weights=(-0.1,),
                                          parameters=optimizer_parameters_error,
                                          optimizee_bounding_func=self.optimizee_functionGenerator.bounding_func)
        }

        #checks
        self.checkpointing(optimizer_checkpoint= optimizer_checkpoint,
                            optimizer_parameters_checkpoint=optimizer_checkpoint,
                            iterations=iterations, errorOptimizer=createErrorOptimizer,
                            experiment=self.experiment_functionGenerator, optimizee=self.optimizee_functionGenerator)
        
        #check if results after the checkpointing are the same as without
        best = optimizer.best_individual['coords']
        best_checkpoint = optimizer_checkpoint.best_individual['coords']
        self.assertEqual(best[0], best_checkpoint[0])
        self.assertEqual(best[1], best_checkpoint[1])

    def load_trajectory(self, optimizer, optimizer_parameters, iterations, optimizee, optimizee_parameters, experiment):
        """
            function to run the optimizer and load the trajectory for checkpointing,
            checks if the loaded trajectory has the right generation
            :param optimizer: optimizer object
            :param optimizer_parameters: parameters corresponding to the optimizer
            :param iterations: number of iterations
            :param optimizee: optimizee object
            :param optimizee_parameters: parameters corresponding to the optimizee
            :param experiment: experiment object
        """
        #first run to create trajectory object to use as checkpoint
        run_optimizer(optimizer=optimizer, optimizer_parameters=optimizer_parameters, experiment=experiment, optimizee=optimizee)

        #load trajectory
        home_path =  os.environ.get("HOME")
        root_dir_path = os.path.join(home_path, 'results')
        loaded_traj = experiment.load_trajectory(root_dir_path + '/L2L/simulation/trajectories/trajectory_4.bin')
        
        self.assertEqual(loaded_traj.individual.generation, iterations-1)

        runner_params = {}
        self.trajectory_checkpoint, runner_params = experiment.prepare_experiment(name='L2L-checkpoint',
                                                                              log_stdout=True,
                                                                              runner_params=runner_params,
                                                                              overwrite=True,
                                                                              checkpoint=loaded_traj)

    def checkpointing(self, optimizer_checkpoint, 
                        optimizer_parameters_checkpoint,iterations,
                        errorOptimizer, experiment, optimizee):
        """
            function to run the checkpointing and check if the optimizer generation is set correctly 
            and if there occurs an error message, when the population size is changed
            :param optimizer_checkpoint: optimizer object 
            :param optimizer_parameters_checkpoint:  parameters corresponding to the optimizer
            :param iterations: number of iterations
            :param errorOptimizer: lambda function creating an optimizer with different popsize to raise an error message
        """

        self.assertEqual(optimizer_checkpoint.g, iterations-1)

        run_optimizer(experiment=experiment, optimizer=optimizer_checkpoint, optimizer_parameters=optimizer_parameters_checkpoint, optimizee=optimizee)

        self.assertRaises(ValueError, errorOptimizer)
        

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(CheckpointCETestCase)
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

def run_optimizer(optimizer, optimizer_parameters, experiment, optimizee):
        experiment.run_experiment(optimizee=optimizee,
                                optimizee_parameters=optimizee,
                                optimizer=optimizer,
                                optimizer_parameters=optimizer_parameters)

        experiment.end_experiment(optimizer)

if __name__ == "__main__":
    run()
