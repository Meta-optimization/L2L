import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .network import NestBenchmarkNetwork
import numpy as np
import random

HPCBMOptimizeeParameters = namedtuple(
    'HPCBMOptimizeeParameters', ['scale']) # TODO: add pre-sim-time, sim-time and dt as parameters

class HPCBMOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        self.scale = parameters.scale


    def create_individual(self):
        """
        Creates and returns a random individual
        """

        individual = {'weight_ex':  random.uniform(1     , 20),
                      'weight_in':  random.uniform(-100  , -5),
                      'pCE':        random.uniform(0     , 1),
                      'pCI':        random.uniform(0     , 1),
                      'delay':      random.uniform(0.1   , 10),
                      }   

        print("random individual:", individual) 
        
        return individual
    

    def bounding_func(self, individual):
        """
        """
        # TODO what are reasonable bounds?
        # delay             originally: 1.5                                now range: [0.1, 10]?
        # weight_ex         originally: JE_pA = 10.77                      now range: [1, 20]?
        # weight_in         originally: g*JE_pA = -5*10.77 = -53.85        now range: [-100, -5]?
        # CE                originally: 9000 fixed                         now: pairwise bernoulli range: [0, 1]
        # CI                originally: 2250 fixed                         now: pairwise bernoulli range: [0, 1]
        individual = {'weight_ex':  np.clip(individual['weight_ex'] , 1     , 20),
                      'weight_in':  np.clip(individual['weight_in'] , -100  , -5),
                      'pCE':        np.clip(individual['pCE']       , 0     , 1),
                      'pCI':        np.clip(individual['pCI']       , 0     , 1),
                      'delay':      np.clip(individual['delay']     , 0.1   , 10),
                      }    
        return individual
    


    def simulate(self, traj):
        """
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        weight_ex = traj.individual.weight_ex
        weight_in = traj.individual.weight_in

        pCE = traj.individual.pCE
        pCI = traj.individual.pCI
        delay = traj.individual.delay
        # scale, pCE, pCI, weight_excitatory, weight_inhibitory, delay, extra_kernel_params=None
        net = NestBenchmarkNetwork(scale=self.scale, 
                                   pCE=pCE, 
                                   pCI=pCI, 
                                   weight_excitatory=weight_ex, 
                                   weight_inhibitory=weight_in, 
                                   delay=delay
                                   )
        average_rate = net.run_simulation()


        
        desired_rate = 0.1
        fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
        print("fitness:", fitness)
        return (fitness,) 
    

    
    
    




