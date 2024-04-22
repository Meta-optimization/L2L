"""
Optimizer for Simulation-based Inference based on the sbi python package
"""

import logging

import sbi
import torch
import numpy as np

from l2l.utils.tools import cartesian_product
from l2l import dict_to_list, list_to_dict, get_grouped_dict
from l2l.optimizers.optimizer import Optimizer

from collections import namedtuple
from os.path import join, isdir
import os
import dill

logger = logging.getLogger("optimizers.sbi")

SBIOptimizerParameters = namedtuple('SBIParameters',
                                        ['pop_size', 'n_iteration', 'seed', 'inference_method', 'save_path', 'x_obs', 'restrict_prior', 'tensorboard'],
                                    defaults=(None, None, 0, False))
SBIOptimizerParameters.__doc__ = """
:param seed: Random seed
"""

class SBIOptimizer(Optimizer):
    """
    Implements simulation-based inference based on the sbi python package.

    This is the base class for the Optimizers i.e. the outer loop algorithms. These algorithms generate parameters, \
    give them to the inner loop to be evaluated, and with the resulting fitness modify the parameters in some way.

    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store the parameters of the specific runs.
        The parameters should be initialized based on the values in :param parameters:

    :param optimizee_create_individual: A function which when called returns one instance of parameter (or "individual")

    :param optimizee_fitness_weights: The weights which should be multiplied with the fitness returned from the
        :class:`~l2l.optimizees.optimizee.Optimizee` -- one for each element of the fitness (fitness can be
        multi-dimensional). If some element is negative, the Optimizer minimizes that element of fitness instead of
        maximizing. By default, the `Optimizer` maximizes all fitness dimensions.

    :param parameters: A named tuple containing the parameters for the Optimizer class

    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 optimizee_bounding_func,
                 parameters):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters, optimizee_bounding_func=optimizee_bounding_func)

        # check parameters and add to trajectory
        traj.f_add_parameter('pop_size', parameters.pop_size, comment='Population size')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iterations to run')
        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')
        traj.f_add_parameter('inference_method', parameters.inference_method, comment='SBI method to use for inference')
        traj.f_add_parameter('save_path', parameters.save_path, comment='Path for saving the models')
        traj.f_add_parameter('x_obs', parameters.x_obs, comment='Observation to use for multi-round inference')
        traj.f_add_parameter('restrict_prior', parameters.restrict_prior, comment='Number of generations to run with the restriction estimator. Value 0 indicates to use no restriction at all.')
        traj.f_add_parameter('tensorboard', parameters.tensorboard, comment='Whether to use TensorBoard or not')

        if traj.save_path and not os.path.isdir(traj.save_path):
                raise ValueError(f'Path {traj.save_path} (save_path) does not exist.')

        if traj.x_obs is None and traj.n_iteration > traj.restrict_prior+1:
            raise ValueError('You have to define an observation x_obs if you are doing multi-round inference.')

        summary_writer = None
        if traj.tensorboard:
            summary_writer = torch.utils.tensorboard.writer.SummaryWriter() # TODO anderen Path angeben?

        # initialize models
        if traj.is_loaded: # TODO
            raise NotImplementedError()
            # logger.info('Loading previous models')
            # last_idx = ?
            # self.prior = self._load_obj(join(traj.save_path, f'restricted_prior_{last_idx}.pkl'))
            # self.restriction_estimator = self._load_obj(join(traj.save_path, f'restriction_estimator_{last_idx}.pkl'))

            # self.g = last_idx
        else:
            logger.info('Creating new models')
            ind_dict, self.prior = optimizee_create_individual()
            #_, self.dict_spec = dict_to_list(ind_dict, True)
            if traj.restrict_prior > 0:
                self.restriction_estimator = sbi.utils.RestrictionEstimator(prior=self.prior)
            self.inference = traj.inference_method(prior=self.prior, summary_writer=summary_writer)

            self.g = 0  # the current generation

        logger.info('Initializing individuals')
        #samples = self.prior.sample((traj.pop_size,))
        self.eval_pop, self.samples = self.optimizee_create_individual(traj.pop_size)
        #self.eval_pop = samples
        #self.eval_pop = [list_to_dict(sample, self.dict_spec) for sample in samples]
        print(self.eval_pop)
        #self.eval_pop = [{'parameters': sample} for sample in samples] # TODO parameter vector?

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        This is the key function of this class. Given a set of :obj:`fitnesses_results`,  and the :obj:`traj`, it uses
        the fitness to decide on the next set of parameters to be evaluated. Then it fills the :attr:`.Optimizer.eval_pop` with the
        list of parameters it wants evaluated at the next simulation cycle, increments :attr:`.Optimizer.g` and calls
        :meth:`._expand_trajectory`

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :param list fitnesses_results: This is a list of fitness results that contain tuples run index and the fitness.
            It is of the form `[(run_idx, run), ...]`

        """
        logger.info('Gathering simulation results')
        # x = torch.Tensor([traj.current_results[i][1] for i in range(traj.pop_size)])
        # individuals = traj.individuals[self.g]
        # theta = torch.stack([individuals[i].parameters for i in range(traj.pop_size)]) # TODO best way?
        theta = self.samples # TODO aufpassen bei invalid, falls ersetzt wird
        x = torch.zeros((len(theta), len(fitnesses_results[0]))) # TODO change zeros to empty

        for i, (run_index, fitness) in enumerate(fitnesses_results):
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx
            x[ind_index] = torch.Tensor(fitness[1]) # only need simulation results

        print('theta', theta)
        print('x', x)

        # check if there are any valid simulations
        mask = torch.isnan(x).any(dim=1)
        if mask.all():
            raise ValueError(f'There was no valid simulation in generation {self.g}. Please check your prior and your simulation.')

        if traj.save_path:
            logger.info('Saving data')
            tmp_path = join(traj.save_path, f'gen{self.g}')
            os.mkdir(tmp_path)
            torch.save(x[~mask], join(tmp_path, f'x_{self.g}.pt'))
            torch.save(theta[~mask], join(tmp_path, f'theta_{self.g}.pt'))
            if mask.any():
                torch.save(theta[mask], join(tmp_path, f'invalid_theta_{self.g}.pt'))

        # check if it is time for running the inference
        inference_round = (self.g >= traj.restrict_prior)
        evaluation_round = False

        #########################
        # restriction estimator #
        #########################

        if not inference_round and mask.any():
            logger.info('Fitting the restriction estimator')
            self.restriction_estimator.append_simulations(theta, x)
            self.restriction_estimator.train()
            self.prior = self.restriction_estimator.restrict_prior()

            if traj.save_path:
                logger.info('Saving the restriction estimator and the restricted prior to disk')
                save_dict = {'restriction_estimator': self.restriction_estimator,
                             'restricted_prior': self.prior,
                             'validation_log_probs': self.restriction_estimator._validation_log_probs}
                self._save_obj(save_dict, join(tmp_path, f'restriction_{self.g}.dill'))

                # TODO sampeln und analysieren, Lernkurven? nur bei bestimmtem Flag?
        elif inference_round:
            logger.info('No need to fit the restriction estimator (inference round)')
        else:
            logger.info('No need to fit the restriction estimator (no invalid simulation data)')

        ####################
        # inference method #
        ####################

        self.inference = self.inference.append_simulations(theta[~mask], x[~mask], proposal=self.prior) # TODO lieber alle?
        if inference_round:
            logger.info('Running the inference method')
            self.density_estimator = self.inference.train()
            self.posterior = self.inference.build_posterior(self.density_estimator)
            self.prior = self.posterior.set_default_x(traj.x_obs) # for multi-round inference

            if traj.save_path:
                logger.info('Saving the inference model data and posterior to disk')
                save_dict = {'inference': self.inference,
                             'density_estimator': self.density_estimator,
                             'posterior': self.posterior,
                             'summary': self.inference.summary}
                self._save_obj(save_dict, join(tmp_path, f'inference_{self.g}.dill'))

                # TODO sampeln und analysieren, Lernkurven? nur bei bestimmtem Flag?

        ##############
        # evaluation #
        ##############

        if evaluation_round:
            logger.info('Evaluating')

            # TODO Daten speichern/Plot erstellen bzgl. Posterior Predictive Check
            # TODO bei amortized: Simulation-based Calibration?

        # TODO schon eine generation vorher müssen die richtigen Parameter für die Evaluation gewählt werden
        if not self.g+1 == traj.n_iteration: # skip last generation
            logger.info('Sampling the new population')
            self.eval_pop, self.samples = self.optimizee_create_individual(traj.pop_size, prior=self.prior)
            self.g += 1
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        """
        pass

    def _save_obj(self, obj, path):
        """
        This function saves an object using dill.
        """
        with open(path, 'wb') as handle:
            dill.dump(obj, handle)

    def _load_obj(self, path):
        """
        This function laods an object from file using dill.
        """
        with open(path, "rb") as handle:
            obj = dill.load(handle)
        return obj

    def _expand_trajectory(self, traj):
        """
        Add as many explored runs as individuals that need to be evaluated. Furthermore, add the individuals as explored
        parameters.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return:
        """

        grouped_params_dict = get_grouped_dict(self.eval_pop)
        grouped_params_dict = {'individual.' + key: val for key, val in grouped_params_dict.items()}

        final_params_dict = {'generation': [self.g],
                             'ind_idx': range(len(self.eval_pop))}
        final_params_dict.update(grouped_params_dict)

        # We need to convert them to lists or write our own custom IndividualParameter ;-)
        # Note the second argument to `cartesian_product`: This is for only having the cartesian product
        # between ``generation x (ind_idx AND individual)``, so that every individual has just one
        # unique index within a generation.
        traj.f_expand(cartesian_product(final_params_dict,
                                        [('ind_idx',) + tuple(grouped_params_dict.keys()), 'generation']))