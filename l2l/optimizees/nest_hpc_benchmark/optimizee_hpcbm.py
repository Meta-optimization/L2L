import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from network import NestBenchmarkNetwork

HPCBMOptimizeeParameters = namedtuple(
    'HPCBMOptimizeeParameters', [])


class HPCBMOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {}
        return individual
    

    def bounding_fun(individual):
        """
        """
        return individual
    


    def simulate(self, traj):
        """
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        
        net = NestBenchmarkNetwork(weight_excitatory=1, weight_inhibitory=1)
        net.run_simulation()

        # TODO: make sim() reuturn a value from which a fitness can be computed
        
        fitness = 0
        return (fitness,) 
    

    
    
    




