import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

AWOptimizeeParameters = namedtuple(
    'AWOptimizeeParameters', ['difficulty'])


class AWOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.difficulty = parameters.difficulty
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.bound = [self.difficulty, self.difficulty]

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {'difficulty':self.difficulty}
        return individual
    
    def is_prime(self, n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def bounding_func(self, individual):
        return individual

    def simulate(self, traj):
        """
        Calculates primes and returns fitness=0
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        
        # Active wait by calculating all primes up to 'difficulty'
        primes = []

        for number in range(1, int(self.difficulty)):
            if self.is_prime(number):
                primes.append(number)
        
        fitness = 0
        return (fitness,) 
    

    
    
    




