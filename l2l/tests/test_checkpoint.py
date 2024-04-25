import unittest
import os

import numpy as np
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import RMSPropParameters
from l2l.tests.test_optimizer import OptimizerTestCase
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters


class CheckpointTestCase(OptimizerTestCase):

    def test_setup(self):

        iterations = 5

        #test checkpointing for GA
        optimizer_parameters = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=iterations, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1
                                                          )

        optimizer = GeneticAlgorithmOptimizer(self.trajectory, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters)
        
        self.load_trajectory(optimizer, optimizer_parameters, iterations)

        optimizer_parameters_checkpoint = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=1, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1)

        optimizer_checkpoint = GeneticAlgorithmOptimizer(self.trajectory, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters_checkpoint)


        optimizer_parameters_error = GeneticAlgorithmParameters(seed=0, pop_size=55, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=1, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1)
    
        self.checkpointing(optimizer= optimizer,
                            optimizer_checkpoint= optimizer_checkpoint,
                            optimizer_parameters_checkpoint=optimizer_checkpoint,
                            optimizer_parameters_error=optimizer_parameters_error,
                            iterations=iterations, optimizee=self.optimizee_functionGenerator,
                            optimizee_parameters=self.optimizee_parameters_functionGenerator)
        
        #test checkpointing for gd
        
    def load_trajectory(self, optimizer, optimizer_parameters, iterations):
         #first run to create trajectory object to use as checkpoint
        run_optimizer(self=self, optimizer=optimizer, optimizer_parameters=optimizer_parameters, 
                      optimizee=self.optimizee_functionGenerator, 
                      optimizee_parameters=self.optimizee_parameters_functionGenerator)

        #load trajectory
        home_path =  os.environ.get("HOME")
        root_dir_path = os.path.join(home_path, 'results')
        loaded_traj = self.experiment.load_trajectory(root_dir_path + '/L2L/simulation/trajectories/trajectory_4.bin')
        
        self.assertEqual(loaded_traj.individual.generation, iterations-1)

        jube_params = {}
        self.trajectory, all_jube_params = self.experiment.prepare_experiment(name='L2L-checkpoint',
                                                                              log_stdout=True,
                                                                              jube_parameter=jube_params,
                                                                              overwrite=True,
                                                                              checkpoint=loaded_traj)
         

    def checkpointing(self, optimizer, optimizer_checkpoint, 
                           optimizer_parameters_checkpoint, optimizer_parameters_error, iterations,
                           optimizee, optimizee_parameters):

        self.assertEqual(optimizer_checkpoint.g, iterations-1)

        run_optimizer(self=self, optimizer=optimizer_checkpoint, optimizer_parameters=optimizer_parameters_checkpoint,
                      optimizee=optimizee, 
                      optimizee_parameters=optimizee_parameters)

        best = optimizer.best_individual['coords']
        best_checkpoint = optimizer_checkpoint.best_individual['coords']
        self.assertEqual(best[0], best_checkpoint[0])
        self.assertEqual(best[1], best_checkpoint[1])
    
        createOptimizer = lambda : {
            GeneticAlgorithmOptimizer(self.trajectory, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters_error)
        }
        
        self.assertRaises(ValueError, createOptimizer)
        

def suite():
    suite = unittest.makeSuite(CheckpointTestCase, 'test')
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

def run_optimizer(self, optimizer, optimizer_parameters, optimizee, optimizee_parameters):
        self.experiment.run_experiment(optimizee=optimizee,
                                            optimizee_parameters=optimizee_parameters,
                                            optimizer=optimizer,
                                            optimizer_parameters=optimizer_parameters)

        self.experiment.end_experiment(optimizer)

if __name__ == "__main__":
    run()
