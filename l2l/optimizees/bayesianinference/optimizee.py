from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

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

    def create_individual(self):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary with dot-separated
        parameter names as keys and parameter values as values. This is used by the optimizers via the
        function create_individual() to initialize the individual/parameters. After that, the change in parameters is
        model specific e.g. In simulated annealing, it is perturbed on specific criteria

        :return dict: A dictionary containing the names of the parameters and their values
        """
        from .prior import prior
        return {'parameters': prior.sample(),
                'prior': prior}

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.

        """
        from .network import Network
        import nest

        # TODO Zeitmessung?
        seed = 1#traj.individual.seed # TODO seed?
        nest.SetKernelStatus({'resolution': 0.1, 'rng_seed': seed, 'local_num_threads': 3})
        simtime = 400#2000

        network = Network(**traj.individual) # TODO geht?
        network.create_nodes()
        network.connect_nodes()
        obs = network.simulate(simtime)
        return obs

    def bounding_func(self, individual):
        """
        placeholder
        """
        return individual
