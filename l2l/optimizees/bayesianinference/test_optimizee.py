from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
import torch
from sbi import utils
from .prior import prior, labels

import numpy as np

TestSBIOptimizeeParameters = namedtuple('TestSBIOptimizeeParameters', ['sim_type'])

class TestSBIOptimizee(Optimizee):
    """
    Implements an optimizer for testing the SBIOptimizer.
    """

    def __init__(self, traj, parameters):
        """
        This is the base class init function. Any implementation must in this class add a parameter add its parameters
        to this trajectory under the parameter group 'individual' which is created here in the base class. It is
        especially necessary to add all explored parameters (i.e. parameters that are returned via create_individual) to
        the trajectory.
        """
        super().__init__(traj)
        self.type = parameters.sim_type
        if self.type not in ['valid', 'invalid', 'mixed']:
            raise ValueError("Invalid type. Type must be 'valid', 'invalid' or 'mixed'")
        self.prior = utils.BoxUniform(low=torch.Tensor([0.0, -200.0, 0.1, 0.0, 0.0]), high=torch.Tensor([200.0, 0.0, 5.0, 1.0, 1.0]))

    def create_individual(self, n=1, prior=prior, labels=labels):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary with dot-separated
        parameter names as keys and parameter values as values. This is used by the optimizers via the
        function create_individual() to initialize the individual/parameters. After that, the change in parameters is
        model specific e.g. In simulated annealing, it is perturbed on specific criteria

        :return dict: A dictionary containing the names of the parameters and their values
        """
        samples = prior.sample((n,))
        pop = [dict(zip(labels, sample)) for sample in samples]
        if n == 1:
            return pop[0], prior # TODO okay?
        return pop, samples

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.

        """
        if self.type == 'valid':
            return np.random.rand(2).tolist()
        elif self.type == 'invalid':
            return (np.nan, np.nan)
        else:
            tmp = np.random.rand(2).tolist() if np.random.rand()<0.5 else (np.nan, np.nan)
            print(traj.individual)
            print(tmp)
            return tmp

    def bounding_func(self, individual):
        """
        placeholder
        """
        return individual