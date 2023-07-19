import logging
from collections import namedtuple

import numpy as np

from l2l import dict_to_list
from l2l import list_to_dict
from l2l.optimizers.optimizer import Optimizer
from l2l import get_grouped_dict

logger = logging.getLogger("optimizers.gradientdescent")

MultiClassicGDParameters = namedtuple(
    'ClassicGDParameters',
    ['learning_rate', 'exploration_step_size', 'n_random_steps', 'n_iteration', 'stop_criterion', 'seed', 'n_inner_params'])
MultiClassicGDParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param n_inner_params: Number of internal parameters explored per multi-individual
"""

MultiStochasticGDParameters = namedtuple(
    'StochasticGDParameters',
    ['learning_rate', 'stochastic_deviation', 'stochastic_decay', 'exploration_step_size', 'n_random_steps', 'n_iteration',
     'stop_criterion', 'seed', 'n_inner_params'])
MultiStochasticGDParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param stochastic_deviation: The standard deviation of the random vector used to perturbate the gradient
:param stochastic_decay: The decay of the influence of the random vector that is added to the gradient 
    (set to 0 to disable stochastic perturbation)
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param n_inner_params: Number of internal parameters explored per multi-individual
"""

MultiAdamParameters = namedtuple(
    'AdamParameters',
    ['learning_rate', 'exploration_step_size', 'n_random_steps', 'first_order_decay', 'second_order_decay', 'n_iteration',
     'stop_criterion', 'seed', 'n_inner_params'])
MultiAdamParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param first_order_decay: Specifies the amount of decay of the historic first order momentum per gradient descent step
:param second_order_decay: Specifies the amount of decay of the historic second order momentum per gradient descent step
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param n_inner_params: Number of internal parameters explored per multi-individual
"""

MultiRMSPropParameters = namedtuple(
    'RMSPropParameters',
    ['learning_rate', 'exploration_step_size', 'n_random_steps', 'momentum_decay', 'n_iteration', 'stop_criterion', 'seed', 'n_inner_params'])
MultiRMSPropParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param momentum_decay: Specifies the decay of the historic momentum at each gradient descent step
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param seed: The random seed used for random number generation in the optimizer
:param n_inner_params: Number of internal parameters explored per multi-individual
"""


class MultiGradientDescentOptimizer(Optimizer):
    """
    Class for a generic gradient descent solver with individuals which contain multiple parameters inside.
    This type of optimizer allows for the definition and deployment of multi-individuals which inside contain
    several individuals, each with a parameter combination.
    This optimizer is particularly useful when the hardware used to execute the individuals offers a second
    hierarchy of parallelization, e.g. in the case of GPUs, where one multi-individual can be defined per GPU and
    several individuals with independent parameter combinations can be deployed within each GPU.

    In the pseudo code the algorithm does:
    For n iterations do:
        - Explore the fitness of individuals in the close vicinity of the current one
        - Calculate the gradient based on these fitnesses.
        - Create the new 'current individual' by taking a step in the parameters space along the direction
            of the largest ascent of the plane
    NOTE: This expects all parameters of the system to be of floating point
    :param  ~pypet.trajectory.Trajectory traj:
      Use this pypet trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`
    
    :param optimizee_create_individual:
      Function that creates a new individual
    
    :param optimizee_fitness_weights: 
      Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each
      element of the fitness vector)
    
    :param parameters: 
      Instance of :func:`~collections.namedtuple` :class:`.ClassicGDParameters`,
      :func:`~collections.namedtuple` :class:`.StochasticGDParameters`,
      :func:`~collections.namedtuple` :class:`.RMSPropParameters` or
      :func:`~collections.namedtuple` :class:`.AdamParameters` containing the
      parameters needed by the Optimizer. The type of this parameter is used to select one of the GD variants.
    
    :param optimizee_bounding_func:
      This is a function that takes an individual as argument and returns another individual that is
      within bounds (The bounds are defined by the function itself). If not provided, the individuals
      are not bounded.
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters, optimizee_bounding_func=optimizee_bounding_func)
        self.recorder_parameters = parameters
        self.optimizee_bounding_func = optimizee_bounding_func
        
        traj.f_add_parameter('learning_rate', parameters.learning_rate, comment='Value of learning rate')
        traj.f_add_parameter('exploration_step_size', parameters.exploration_step_size, 
                             comment='Standard deviation of the random steps')
        traj.f_add_parameter('n_random_steps', parameters.n_random_steps, 
                             comment='Amount of random steps taken for calculating the gradient')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('n_inner_params', parameters.n_inner_params, comment='Number of parameters internally explored per individual')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('seed', np.uint32(parameters.seed), comment='Optimizer random seed')

        _, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)
        self.random_state = np.random.RandomState(seed=traj.par.seed)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the gradient descent algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        # Attempt with homogeneous distribution of points in space.
        # Multigradient descent starts with a homogeneous distribution of all parameters in space.
        # individual_dicts = self.optimizee_create_individual() # [{delay: a1, coupling: b1}, {delay: a1, coupling: b2},...]

        self.current_individual = np.array(
            dict_to_list(self.optimizee_create_individual()))  # [one random sample]

        # Depending on the algorithm used, initialize the necessary variables
        self.updateFunction = None
        if type(parameters) is MultiClassicGDParameters:
            self.init_classic_gd(parameters, traj)
        elif type(parameters) is MultiStochasticGDParameters:
            self.init_stochastic_gd(parameters, traj)
        elif type(parameters) is MultiAdamParameters:
            self.init_adam(parameters, traj)
        elif type(parameters) is MultiRMSPropParameters:
            self.init_rmsprop(parameters, traj)
        else:
            raise Exception('Class of the provided "parameters" argument is not among the supported types')

        # Added a generation-wise parameter logging
        traj.results.f_add_result_group('generation_params',
                                        comment='This contains the optimizer parameters that are'
                                                ' common across a generation')

        # Explore the neighbourhood in the parameter space of current individual
         new_individual_list = [
            list_to_dict(self.current_individual + 
                         self.random_state.normal(0.0, parameters.exploration_step_size, self.current_individual.size),
                         self.optimizee_individual_dict_spec)
            for i in range((parameters.n_random_steps*traj.n_inner_params)-1)
        ]
        # In multigradient we create this by hand the first time. Here we should have 1024 parameters / individuals
        # Attempt with homogeneous distribution of points in space.
        # new_individual_list = [np.array(dict_to_list(self.optimizee_create_individual(individual_dicts[i]))) for i in individual_dicts] # [[a1,b1],[a1,b2],...]
        # print(new_individual_list)

        # Also add the current individual to determine it's fitness
        #new_individual_list.append(list_to_dict(self.current_individual, self.optimizee_individual_dict_spec))
            
        if optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]

        self.grouped_params_dict = get_grouped_dict(new_individual_list)

        # Storing the fitness of the current individual
        self.current_fitness = -np.Inf
        self.g = 0
        new_individual_list = self.compress_individual(new_individual_list, traj.n_inner_params)
        self.eval_pop = new_individual_list
        self._expand_trajectory(traj)

    def get_params(self):
        """
        Get parameters used for recorder
        :return: Dictionary containing recorder parameters
        """
        return self.recorder_parameters._asdict()

    def expand_individual(self, c_population, inner_params):
        """
        Expands a multi-individual into many individuals, each with their own parameter combination.
        This expansion is necessary in order for the evolutionary algorithm to be applied correctly on the parameter
        space being explored.

        :param  c_population: the population of multi-individuals to be expanded
        :param inner_params: a description of the parameters which describe each of the individuals within
        the multi-individual.

        """
        individual_exp = [{} for i in range(len(c_population)*inner_params)]
        for ind_id, elem in enumerate(c_population):
            for key in self.grouped_params_dict.keys():
                parameters = elem[key]
                for ix, e in enumerate(parameters):
                    individual_exp[ind_id*inner_params+ix][key] = float(e)
        return individual_exp

    def compress_individual(self, e_population, inner_params):
        """
        Compresses a set of individuals into a set of multi-individuals.
        This is required after the optimization algorithm has been applied in order to return the description
        of the individuals to the way it can be correctly interpreted during execution.

        :param  e_population: the population of individuals to be compressed
        :param inner_params: a description of the parameters which describe each of the individuals within
        the multi-individual.

        """
        e_population_reform = []
        for s in range(int(len(e_population) / inner_params)):
           tmp_dict = {}
           for key in self.grouped_params_dict.keys():
               tmp = []
               for p in range(inner_params):
                   tmp.append(e_population[s*inner_params + p][key])
               tmp_dict[key] = tmp
           e_population_reform.append(tmp_dict)
        return e_population_reform

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        # print('g counter:', self.g)
        old_eval_pop = self.eval_pop.copy()
        old_eval_pop_expanded = self.expand_individual(old_eval_pop,traj.n_inner_params)
        self.eval_pop.clear()

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))
        
        assert len(fitnesses_results) == traj.n_random_steps

        # We need to collect the directions of the random steps along with the fitness evaluated there
        fitnesses = np.zeros((traj.n_random_steps*traj.n_inner_params))
        dx = np.zeros((traj.n_random_steps*traj.n_inner_params, 2))
        weighted_fitness_list = []
        fitnesses_results_exp = []
        for (id, elem) in fitnesses_results:
            for ix, e in enumerate(elem):
                fitnesses_results_exp.append((ix, float(e)))
        # print(fitnesses_results_exp)
        for i, (run_index, fitness) in enumerate(fitnesses_results_exp):
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation
            traj.v_idx = run_index
            ind_index = i#traj.par.ind_idx

            individual = old_eval_pop_expanded[i]
            traj.f_add_result('$set.$.individual', individual)
            traj.f_add_result('$set.$.fitness', fitness)
            # print(fitness)
            weighted_fitness = np.dot(fitness, self.optimizee_fitness_weights)
            weighted_fitness_list.append(weighted_fitness)

            # The last element of the list is the evaluation of the individual obtained via gradient descent
            if i == len(fitnesses_results) - 1:
                self.current_fitness = weighted_fitness
            else:
                fitnesses[i] = weighted_fitness
                dx[i] = np.array(dict_to_list(individual)) - self.current_individual
        traj.v_idx = -1  # set the trajectory back to default

        # Performs descending arg-sort of weighted fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))
        old_eval_pop_as_array = np.array([dict_to_list(x) for x in old_eval_pop])

        # Sorting the data according to fitness
        sorted_population = old_eval_pop_as_array[fitness_sorting_indices]
        sorted_fitness = np.asarray(weighted_fitness_list)[fitness_sorting_indices]

        logger.info("-- End of generation %d --", self.g)
        logger.info("  Evaluated %d individuals", len(fitnesses_results))
        logger.info('  Average Fitness: %.4f', np.mean(sorted_fitness))
        logger.info("  Current fitness is %.2f", self.current_fitness)
        logger.info('  Best Fitness: %.4f', sorted_fitness[0])
        logger.info("  Best individual is %s", sorted_population[0])

        generation_result_dict = {
            'generation': self.g,
            'current_fitness': self.current_fitness,
            'best_fitness_in_run': sorted_fitness[0],
            'average_fitness_in_run': np.mean(sorted_fitness),
        }

        generation_name = 'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        logger.info("-- End of iteration {}, current fitness is {} --".format(self.g, self.current_fitness))

        if self.g < traj.n_iteration - 1 and traj.stop_criterion > self.current_fitness:
            # Create new individual using the appropriate gradient descent
            self.update_function(traj, np.dot(np.linalg.pinv(dx), fitnesses - self.current_fitness))
            current_individual_dict = list_to_dict(self.current_individual, self.optimizee_individual_dict_spec)
            if self.optimizee_bounding_func is not None:
                current_individual_dict = self.optimizee_bounding_func(current_individual_dict)
            self.current_individual = np.array(dict_to_list(current_individual_dict))

            # Explore the neighbourhood in the parameter space of the current individual
            new_individual_list = [
                list_to_dict(self.current_individual +
                             self.random_state.normal(0.0, traj.exploration_step_size, self.current_individual.size),
                             self.optimizee_individual_dict_spec)
                for _ in range((traj.n_random_steps*traj.n_inner_params)-1)
            ]
            if self.optimizee_bounding_func is not None:
                new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]
            new_individual_list.append(current_individual_dict)

            new_individual_list_reform = self.compress_individual(new_individual_list, traj.n_inner_params)
            fitnesses_results.clear()
            self.eval_pop = new_individual_list_reform
            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        best_last_indiv_dict = list_to_dict(self.current_individual.tolist(),
                                            self.optimizee_individual_dict_spec)

        traj.f_add_result('final_individual', best_last_indiv_dict)
        traj.f_add_result('final_fitness', self.current_fitness)
        traj.f_add_result('n_iteration', self.g + 1)

        logger.info("The last individual was %s with fitness %s", self.current_individual, self.current_fitness)
        logger.info("-- End of (successful) gradient descent --")

    def init_classic_gd(self, parameters, traj):
        """
        Classic Gradient Descent specific initializiation.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.
        :return:
        """
        self.update_function = self.classic_gd_update
    
    def init_rmsprop(self, parameters, traj):
        """
        RMSProp specific initializiation.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.
        :return:
        """

        self.update_function = self.rmsprop_update

        traj.f_add_parameter('momentum_decay', parameters.momentum_decay, 
                             comment='Decay of the historic momentum at each gradient descent step')

        self.delta = 10**(-6)  # used to for numerical stabilization
        self.so_moment = np.zeros(len(self.current_individual))  # second order moment

    def init_adam(self, parameters, traj):
        """
        ADAM specific initializiation.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.
        :return:
        """

        self.update_function = self.adam_update

        traj.f_add_parameter('first_order_decay', parameters.first_order_decay, 
                             comment='Decay of the first order momentum')
        traj.f_add_parameter('second_order_decay', parameters.second_order_decay, 
                             comment='Decay of the second order momentum')

        self.delta = 10**(-8)  # used for numerical stablization
        self.fo_moment = np.zeros(len(self.current_individual))  # first order moment
        self.so_moment = np.zeros(len(self.current_individual))  # second order moment

    def init_stochastic_gd(self, parameters, traj):
        """
        Stochastic Gradient Descent specific initializiation.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory on which the parameters should get stored.
        :return:
        """

        self.update_function = self.stochastic_gd_update

        traj.f_add_parameter('stochastic_deviation', parameters.stochastic_deviation, 
                             comment='Standard deviation of the random vector added to the gradient')
        traj.f_add_parameter('stochastic_decay', parameters.stochastic_decay, comment='Decay of the random vector')

    def classic_gd_update(self, traj, gradient):
        """
        Updates the current individual using the classic Gradient Descent algorithm.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm
        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual
        :return:
        """
        self.current_individual += traj.learning_rate * gradient

    def rmsprop_update(self, traj, gradient):
        """
        Updates the current individual using the RMSProp algorithm.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm
        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual
        :return:
        """

        self.so_moment = (traj.momentum_decay * self.so_moment + 
                          (1 - traj.momentum_decay) * np.multiply(gradient, gradient))
        self.current_individual += np.multiply(traj.learning_rate / (np.sqrt(self.so_moment + self.delta)),
                                               gradient)

    def adam_update(self, traj, gradient):
        """
        Updates the current individual using the ADAM algorithm.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm
        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual
        :return:
        """

        self.fo_moment = (traj.first_order_decay * self.fo_moment +
                          (1 - traj.first_order_decay) * gradient)
        self.so_moment = (traj.second_order_decay * self.so_moment +
                          (1 - traj.second_order_decay) * np.multiply(gradient, gradient))
        fo_moment_corrected = self.fo_moment / (1 - traj.first_order_decay ** (self.g + 1))
        so_moment_corrected = self.so_moment / (1 - traj.second_order_decay ** (self.g + 1))

        self.current_individual += traj.learning_rate * fo_moment_corrected / \
                                    (np.sqrt(so_moment_corrected) + self.delta)

    def stochastic_gd_update(self, traj, gradient):
        """
        Updates the current individual using a stochastic version of the gradient descent algorithm.
        :param ~pypet.trajectory.Trajectory traj: The :mod:'pypet' trajectory which contains the parameters 
            required by the update algorithm
        :param ~numpy.ndarray gradient: The gradient of the fitness curve, evaluated at the current individual
        :return:
        """

        gradient += (self.random_state.normal(0.0, traj.stochastic_deviation, self.current_individual.size) * 
                     traj.stochastic_decay**(self.g + 1))
        self.current_individual += traj.learning_rate * gradient

