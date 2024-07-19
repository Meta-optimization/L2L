from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .prior import prior, labels, x_obs

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
        self.nrec = 50
        self.scale = 0.05

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
        from .network_indegree import NestBenchmarkNetwork
        import nest

        # TODO Zeitmessung?

        # self.ind_idx = traj.individual.ind_idx
        # self.generation = traj.individual.generation

        w_ex = traj.individual.w_ex
        w_in = traj.individual.w_in
        c_ex = int(traj.individual.c_ex)
        c_in = int(traj.individual.c_in)
        delay = traj.individual.delay

        net = NestBenchmarkNetwork(scale=self.scale,
                                   CE=500,#c_ex,
                                   CI=100,#c_in,
                                   weight_excitatory=23.3812,#w_ex,
                                   weight_inhibitory=-50.0,#w_in,
                                   delay=0.1,#delay,
                                   nrec=self.nrec
                                   )
        average_rate = net.run_simulation()

        desired_rate = x_obs[0]
        fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
        print(f'gen {self.generation} ind {self.ind_idx} rate {average_rate} fitness {fitness}')
        return (fitness, [average_rate])

    def bounding_func(self, individual):
        """
        placeholder
        """
        return individual
