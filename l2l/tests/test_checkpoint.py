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

        #test checkpointing for GA with function Generator
        optimizer_parameters = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=iterations, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1
                                                          )

        optimizer = GeneticAlgorithmOptimizer(self.trajectory_functionGenerator, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters)
        
        self.load_trajectory(optimizer, optimizer_parameters, iterations, self.optimizee_functionGenerator, self.optimizee_functionGenerator_parameters, self.experiment_functionGenerator)

        optimizer_parameters_checkpoint = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=1, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1)

        optimizer_checkpoint = GeneticAlgorithmOptimizer(self.trajectory_checkpoint, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
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
                            optimizee_parameters=self.optimizee_functionGenerator_parameters)
        
        #test checkpointing for ga with activeWait
        optimizer_parameters = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=iterations, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1
                                                          )

        optimizer = GeneticAlgorithmOptimizer(self.trajectory_activeWait, optimizee_create_individual=self.optimizee_activeWait.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters)
        
        self.load_trajectory(optimizer, optimizer_parameters, iterations, self.optimizee_activeWait, self.optimizee_activeWait_parameters, self.experiment_activeWait)

        optimizer_parameters_checkpoint = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=1, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1)

        optimizer_checkpoint = GeneticAlgorithmOptimizer(self.trajectory_checkpoint, optimizee_create_individual=self.optimizee_activeWait.create_individual,
                                              optimizee_fitness_weights=(-0.1,),
                                              parameters=optimizer_parameters_checkpoint)


        optimizer_parameters_error = GeneticAlgorithmParameters(seed=0, pop_size=55, cx_prob=0.5,
                                                          mut_prob=0.3, n_iteration=1, ind_prob=0.02,
                                                          tourn_size=1, mate_par=0.5,
                                                          mut_par=1)
    
        #check fuer Korrektheit muss noch gefunden werden
        """self.checkpointing(optimizer= optimizer,
                            optimizer_checkpoint= optimizer_checkpoint,
                            optimizer_parameters_checkpoint=optimizer_checkpoint,
                            optimizer_parameters_error=optimizer_parameters_error,
                            iterations=iterations)"""
        
    def load_trajectory(self, optimizer, optimizer_parameters, iterations, optimizee, optimizee_parameters, experiment):
         #first run to create trajectory object to use as checkpoint
        run_optimizer(optimizer=optimizer, optimizer_parameters=optimizer_parameters, experiment=experiment, optimizee=optimizee, optimizee_parameter=optimizee_parameters)

        #load trajectory
        home_path =  os.environ.get("HOME")
        root_dir_path = os.path.join(home_path, 'results')
        loaded_traj = experiment.load_trajectory(root_dir_path + '/L2L/simulation/trajectories/trajectory_4.bin')
        
        self.assertEqual(loaded_traj.individual.generation, iterations-1)

        jube_params = {}
        self.trajectory_checkpoint, all_jube_params = experiment.prepare_experiment(name='L2L-checkpoint',
                                                                              log_stdout=True,
                                                                              jube_parameter=jube_params,
                                                                              overwrite=True,
                                                                              checkpoint=loaded_traj)
         

    def checkpointing(self, optimizer, optimizer_checkpoint, 
                           optimizer_parameters_checkpoint, optimizer_parameters_error, iterations,
                           optimizee, optimizee_parameters):

        self.assertEqual(optimizer_checkpoint.g, iterations-1)

        run_optimizer(experiment=self.experiment_functionGenerator, optimizer=optimizer_checkpoint, optimizer_parameters=optimizer_parameters_checkpoint, optimizee=self.optimizee_functionGenerator, optimizee_parameter=self.optimizee_functionGenerator_parameters)

        best = optimizer.best_individual['coords']
        best_checkpoint = optimizer_checkpoint.best_individual['coords']
        self.assertEqual(best[0], best_checkpoint[0])
        self.assertEqual(best[1], best_checkpoint[1])
    
        createOptimizer = lambda : {
            GeneticAlgorithmOptimizer(self.trajectory_checkpoint, optimizee_create_individual=self.optimizee_functionGenerator.create_individual,
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

def run_optimizer(optimizer, optimizer_parameters, experiment, optimizee, optimizee_parameter):
        experiment.run_experiment(optimizee=optimizee,
                                            optimizee_parameters=optimizee,
                                            optimizer=optimizer,
                                            optimizer_parameters=optimizer_parameters)

        experiment.end_experiment(optimizer)

if __name__ == "__main__":
    run()
