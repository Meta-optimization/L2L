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

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {}
        return individual
    
    def is_prime(self, n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True


    def simulate(self, traj):
        """
        Simulate a run and return a fitness
        """

        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        print('Starting with Generation {}'.format(self.generation))
        start_time = time.time()
        
        
        
        ## sleep
        #time.sleep(self.difficulty)
        
        # busy wait
        #while time.time() - start_time < self.difficulty:
        #    pass

        # active wait prime
        primes = []
        for number in range(1, self.difficulty):
            if self.is_prime(number):
                primes.append(number)

        print(f"Calculated {len(primes)} primes in the range 1 to {self.difficulty}")


        fitness = 0
        print("gen, ind, duration in s, fitness")
        print(f"{self.generation}, {self.ind_idx}, {time.time() - start_time}, {fitness}")

        return (fitness,) 
    

    
    
    




