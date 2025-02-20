
import logging
from collections import namedtuple

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from l2l import dict_to_list
from l2l import list_to_dict
from l2l.optimizers.optimizer import Optimizer
from l2l import get_grouped_dict

logger = logging.getLogger("optimizers.gradientdescent")

MultiClassicGDParameters = namedtuple(
    'ClassicGDParameters',
    ['learning_rate', 'exploration_step_size', 'n_random_steps', 'n_iteration', 'stop_criterion', 'seed'])
MultiClassicGDParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
"""

MultiStochasticGDParameters = namedtuple(
    'StochasticGDParameters',
    ['learning_rate', 'stochastic_deviation', 'stochastic_decay', 'exploration_step_size', 'n_random_steps', 'n_iteration',
     'stop_criterion', 'seed'])
MultiStochasticGDParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param stochastic_deviation: The standard deviation of the random vector used to perturbate the gradient
:param stochastic_decay: The decay of the influence of the random vector that is added to the gradient
    (set to 0 to disable stochastic perturbation)
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
"""

MultiAdamParameters = namedtuple(
    'AdamParameters',
    ['learning_rate', 'exploration_step_size', 'n_random_steps', 'first_order_decay', 'second_order_decay', 'n_iteration',
     'stop_criterion', 'seed'])
MultiAdamParameters.__doc__ = """
:param learning_rate: The rate of learning per step of gradient descent
:param exploration_step_size: The standard deviation of random steps used for finite difference gradient
:param n_random_steps: The amount of random steps used to estimate gradient
:param first_order_decay: Specifies the amount of decay of the historic first order momentum per gradient descent step
:param second_order_decay: Specifies the amount of decay of the historic second order momentum per gradient descent step
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value

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
"""


class MultiGradientDescentOptimizer(Optimizer):
    """
    Class for a generic gradient descent solver.
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

        print("give me some sanity !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the gradient descent algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        # Attempt with homogeneous distribution of points in space.
        # Multigradient descent starts with a homogeneous distribution of all parameters in space.
        # individual_dicts = self.optimizee_create_individual() # [{delay: a1, coupling: b1}, {delay: a1, coupling: b2},...]
        # self.current_individual = self.optimizee_create_individual()

        oci = self.optimizee_create_individual()
        # print('CIraw', oci)
        self.current_individual = np.array(
            dict_to_list(oci))  # [one random sample]
        # print('CId2l', self.current_individual)

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
        # print(new_individual_list)

        # Also add the current individual to determine it's fitness
        new_individual_list.append(list_to_dict(self.current_individual, self.optimizee_individual_dict_spec))

        if optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]

        self.grouped_params_dict = get_grouped_dict(new_individual_list)

        # Storing the fitness of the current individual
        self.current_fitness = -np.Inf
        self.g =  traj.individual.generation
        new_individual_list = self.compress_individual(new_individual_list, traj.n_inner_params)
        self.eval_pop = new_individual_list
        #print('evalpop', self.eval_pop)
        #print('traj', traj)
        self._expand_trajectory(traj)

    def get_params(self):
        """
        Get parameters used for recorder
        :return: Dictionary containing recorder parameters
        """
        return self.recorder_parameters._asdict()

    def expand_individual(self, c_population, inner_params):
        individual_exp = [{} for i in range(len(c_population)*inner_params)]
        for ind_id, elem in enumerate(c_population):
            #print('c_pop', c_population)
            #print('sgpdk', self.grouped_params_dict.keys())
            for key in self.grouped_params_dict.keys():
                parameters = elem[key]
                #print('parei', parameters)
                for ix, e in enumerate(parameters):
                    #print('eeeee', e)
                    #print('eeee1', e[0])
                    individual_exp[ind_id*inner_params+ix][key] = float(e)
        #print('expa', individual_exp)
        return individual_exp

    def compress_individual(self, e_population, inner_params):
        e_population_reform = []
        for s in range(int(len(e_population) / inner_params)):
           tmp_dict = {}
           for key in self.grouped_params_dict.keys():
               tmp = []
               for p in range(inner_params):
                   tmp.append(e_population[s*inner_params + p][key])
               tmp_dict[key] = tmp
           e_population_reform.append(tmp_dict)
        #print('comp', e_population_reform)
        return e_population_reform

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        # print('g counter:', self.g)
        old_eval_pop = self.eval_pop.copy()
        old_eval_pop_expanded = self.expand_individual(old_eval_pop,traj.n_inner_params)
        #print("oldshape", len(fitnesses_results))
        #print("finnessres", fitnesses_results)
        # print('oldepop', old_eval_pop, '')
        #print('oldepop0', old_eval_pop_expanded[0])
        #print('oldepop3', old_eval_pop_expanded[3])
        self.eval_pop.clear()

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))

        assert len(fitnesses_results) == traj.n_random_steps

        # We need to collect the directions of the random steps along with the fitness evaluated there
        fitnesses = np.zeros((traj.n_random_steps*traj.n_inner_params))
        #print('lentrajparams',len(traj.individual.params))
        dx = np.zeros((traj.n_random_steps*traj.n_inner_params, len(traj.individual.params) ))
        #print('trajind', traj.individual)
        # dx = np.zeros((len(fitnesses_results), len(traj.individual))))
        # dx = np.zeros((16, 4))
        weighted_fitness_list = []
        fitnesses_results_exp = []
        for (id, elem) in fitnesses_results:
            for ix, e in enumerate(elem):
                fitnesses_results_exp.append((ix, float(e)))
        # print('frex', fitnesses_results)
        for i, (run_index, fitness) in enumerate(fitnesses_results_exp):
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation
            traj.v_idx = run_index
            # print('runindex', run_index)
            ind_index = i#traj.par.ind_idx
            #print('iiiii', i)
            #if i > len(fitnesses_results):
            #    break
            individual = old_eval_pop_expanded[i]
            #print('indinloop', individual)
            traj.f_add_result('$set.$.individual', individual)
            traj.f_add_result('$set.$.fitness', fitness)
            #print('fitnessinloop', fitness)
            weighted_fitness = np.dot(fitness, self.optimizee_fitness_weights)
            weighted_fitness_list.append(weighted_fitness)

            # print('individual', dict_to_list(individual))

            # mv
            # take best fitness from last iteration
            indictlist = np.array(dict_to_list(individual))
            # TODO should current individual be from last generation or best or mean from current?
            dx[i] = indictlist - self.current_individual
            # print('indictlist', indictlist.shape, '', indictlist)
            # curind = np.array(self.current_individual, 'f')
            # print('curind', curind.shape, '', curind)

            fitnesses[i] = weighted_fitness

            # The last element of the list is the evaluation of the individual obtained via gradient descent
            # if i == len(fitnesses_results_exp) - 1:
            #     #print('ilast', i)
            #     self.current_fitness = weighted_fitness
            #     print('currentind', self.current_individual)
            # else:
            #     fitnesses[i] = weighted_fitness
            #     #print('fitnessi', fitnesses[i])
            #     #print('fressize', fitnesses_results)
            #     #print('selfcurI', self.current_individual)
            #     #print('dct2lstI', dict_to_list(individual))
            #     dx[i] = np.array(dict_to_list(individual)) - self.current_individual
            #     #print('dxidxidxid', dx[i])
        traj.v_idx = -1  # set the trajectory back to default

        #print('wfl',weighted_fitness_list)
        # Performs descending arg-sort of weighted fitness
        fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list, axis=0)))
        #print('fsi', fitness_sorting_indices)
        old_eval_pop_as_array = np.array([dict_to_list(x) for x in old_eval_pop])
        # print("innerparams, lenindivparams", traj.n_inner_params, len(traj.individual.params))
        # print('oepa', old_eval_pop_as_array, '', old_eval_pop_as_array.shape)
        # old_eval_pop_as_array = old_eval_pop_as_array[0].reshape(traj.n_inner_params, len(traj.individual.params))
        old_eval_pop_as_array = old_eval_pop_as_array[0].reshape(len(traj.individual.params), traj.n_inner_params).T
        # print('oepashapafter', old_eval_pop_as_array)
        fitness_sorting_indices = np.array(fitness_sorting_indices).squeeze()

        # print('fitness_sorting_indices', fitness_sorting_indices.shape, ' ', )
        # Sorting the data according to fitness and taking the highest index
        best_index = fitness_sorting_indices[0]
        logger.info(f'old {old_eval_pop_as_array.shape} idx {best_index}')
        best_individual = old_eval_pop_as_array[best_index]
        # print('sortedpop', sorted_population, '', sorted_population.shape)
        avrg_fitness = np.asarray(np.mean(weighted_fitness_list))
        best_fitness = weighted_fitness_list[best_index]

        # print('fitness_sorting_indices', fitness_sorting_indices, ' ',)
        # print('weighted_fitness_list', weighted_fitness_list, ' ',)
        # print('best_index', best_index, ' ',)

        logger.info("-- End of generation %d --", self.g)
        # logger.info("  Evaluated %d individuals", len(fitnesses_results))
        logger.info("  Evaluated %d individuals", len(traj.individual.params))
        logger.info('  Average Fitness: %.4f', avrg_fitness)
        logger.info("  Current fitness is %.2f", self.current_fitness)
        logger.info('  Best Fitness: %.4f', best_fitness)
        logger.info("  Best individual is %s", best_individual)

        generation_result_dict = {
            'generation': self.g,
            'current_fitness': self.current_fitness,
            'best_fitness_in_run': best_fitness,
            'average_fitness_in_run': avrg_fitness,
        }

        # mv. current fitness is set as the best or mean of this.generation (TODO should it be previous?)
        self.current_fitness = best_fitness

        generation_name = 'generation_{}'.format(self.g)
        traj.results.generation_params.f_add_result_group(generation_name)
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        logger.info("-- End of iteration {}, current fitness is {} --".format(self.g, self.current_fitness))

        if self.g < traj.n_iteration -1 and traj.stop_criterion > self.current_fitness:
            # Create new individual using the appropriate gradient descent

            # print('fitn-scfit', fitnesses-self.current_fitness)
            #print('np.linalg.pinv(dx)', np.linalg.pinv(dx))
            #print('dotlinalg', np.dot(np.linalg.pinv(dx), fitnesses - self.current_fitness))

            # print('nilr', new_individual_list_reform)
            # scf = np.array(self.current_fitness, 'f')
            # print('scurfitnes', scf.shape, ' ', scf)
            # fts = np.array(fitnesses, 'f')
            # print('fitnesses', fts.shape, ' ', fts)
            # dxx = np.array(dx, 'f')
            # print('dxdxdxdx', dxx.shape, ' ')
            # dpv = np.array(np.linalg.pinv(dx), 'f')
            # print('pinv', dpv.shape, ' ', dpv)

            self.update_function(traj, np.dot(np.linalg.pinv(dx), fitnesses - self.current_fitness))
            # print('fitnesses ', fitnesses)
            current_individual_dict = list_to_dict(self.current_individual, self.optimizee_individual_dict_spec)
            if self.optimizee_bounding_func is not None:
                current_individual_dict = self.optimizee_bounding_func(current_individual_dict)
            self.current_individual = np.array(dict_to_list(current_individual_dict))

            # Explore the neighbourhood in the parameter space of the current individual
            # newstepsize = ((1-best_fitness)/best_fitness) * traj.exploration_step_size
            # traj.exploration_step_size = newstepsize
            # print('newstepsize', newstepsize)
            newstepsize = traj.exploration_step_size

            new_individual_list = [
                list_to_dict(self.current_individual +
                             self.random_state.normal(0.0, newstepsize, self.current_individual.size),
                             self.optimizee_individual_dict_spec)
                for _ in range((traj.n_random_steps*traj.n_inner_params)-1)
            ]
            if self.optimizee_bounding_func is not None:
                new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]
            new_individual_list.append(current_individual_dict)

            new_individual_list_reform = self.compress_individual(new_individual_list, traj.n_inner_params)

            #print("newshape", new_individual_list_reform)
            #for s in range(len(new_individual_list)/traj.n_inner_params):
            #    tmp = []
            #    tmp_dict = {}
            #    for key in self.grouped_params_dict.keys():
            #        for p in range(traj.n_inner_params):
            #            tmp.append(new_individual_list[s*traj.n_inner_params+p][key])
            #        tmp_dict[key] = tmp
            #    new_individual_list_reform.append(tmp_dict)
            #print(new_individual_list_reform)
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
        # print('ciupdate', self.current_individual)

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
