import os
import numpy as np
from collections import namedtuple

from l2l.optimizees.optimizee import Optimizee

TestcaseOptimizeeParameters = namedtuple(
    'TestcaseOptimizeeParameters', ['exit_code'])

class TestcaseOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.exit_code = parameters.exit_code
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.bound = [self.exit_code, self.exit_code]

    def create_individual(self):
        """
        Creates and returns the individual
        """
        individual = {'exit_code': float(self.exit_code)}
        return individual

    def bounding_func(self, individual):
        return individual

    def simulate(self, traj):
        """
        Simulates error and returns fitness = 0
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        if(traj.retry < 2):
            os._exit(self.exit_code)
        fitness = 0
        return (fitness,) 