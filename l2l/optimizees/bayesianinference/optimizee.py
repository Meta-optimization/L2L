from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .prior import prior, labels, x_obs
import torch
import numpy as np

SBIOptimizeeParameters = namedtuple('SBIOptimizeeParameters', [])

class SBIOptimizee(Optimizee):
    """
    This is the base class for the Optimizees, i.e. the inner loop algorithms. Often, these are the implementations that
    interact with the environment. Given a set of parameters, it runs the simulation and returns the fitness achieved
    with those parameters.
    """

    def __init__(self, traj, parameters):
        """
        This is the base class init function. Any implementation must in this class add a parameter add its parameters
        to this trajectory under the parameter group 'individual' which is created here in the base class. It is
        especially necessary to add all explored parameters (i.e. parameters that are returned via create_individual) to
        the trajectory.
        """
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.nrec = 500
        self.scale = 0.5
        self.COUNT = 0

    def create_individual(self, n=1, prior=prior, labels=labels):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary with dot-separated
        parameter names as keys and parameter values as values. This is used by the optimizers via the
        function create_individual() to initialize the individual/parameters. After that, the change in parameters is
        model specific e.g. In simulated annealing, it is perturbed on specific criteria

        :return dict: A dictionary containing the names of the parameters and their values
        """
        if n == 1000:
            samples = torch.load(['theta_ppc.pt', 'theta_sbc.pt'][self.COUNT])
            self.COUNT += 1
        else:
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
        from .network_indegree import NestBenchmarkNetwork
        import nest # import mpi4py before or after?
        from mpi4py import MPI

        # TODO Zeitmessung?

        # self.ind_idx = traj.individual.ind_idx
        # self.generation = traj.individual.generation

        w_ex = float(traj.individual.w_ex)
        w_in = float(traj.individual.w_in)
        c_ex = int(traj.individual.c_ex)
        c_in = int(traj.individual.c_in)
        delay = np.round(float(traj.individual.delay), 1)

        net = NestBenchmarkNetwork(scale=self.scale,
                                   CE=c_ex,
                                   CI=c_in,
                                   weight_excitatory=w_ex,
                                   weight_inhibitory=w_in,
                                   delay=delay,
                                   nrec=self.nrec
                                   )
        average_rate = net.run_simulation()

        print(f'gen {self.generation} ind {self.ind_idx} rate {average_rate}')

        # MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        average_rate = comm.reduce(average_rate, MPI.SUM)/size

        print(f'gen {self.generation} ind {self.ind_idx} reduced rate {average_rate}')

        if rank == 0:
            desired_rate = x_obs[0]
            fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
            print(f'gen {self.generation} ind {self.ind_idx} rate {average_rate} fitness {fitness}')
            return (fitness, [average_rate])

    def bounding_func(self, individual):
        """
        placeholder
        """
        return individual
