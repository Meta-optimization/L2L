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


    def simulate(self, traj):
        """
        Simulate a run and return a fitness
        """

        print('Starting with Generation {}'.format(self.generation))

        start_time = time.time()
        time.sleep(self.difficulty)
        fitness = 0

        print("gen, ind, duration in s, fitness")
        print(f"{self.generation}, {self.ind_idx}, {time.time() - start_time}, {fitness}")

        return (fitness,) 




