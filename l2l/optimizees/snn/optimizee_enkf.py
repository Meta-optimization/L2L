from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter
from l2l.optimizers.kalmanfilter.data import fetch
from l2l.optimizees.snn.reservoir_nest3 import Reservoir
from l2l.optimizers.crossentropy.distribution import Gaussian
from scipy.special import softmax
from scipy.stats import rv_histogram

import json
import numpy as np
import os
import pathlib
import pandas as pd
import subprocess
import time


# TODO:
'''
- weights are saved as `{individual_index}_{simulation_index}_weights_{type}.csv`
- the key in the dictionary is `{individual_index}_{simulation_index}_weights_{type}`
- model outs are saved as `{individual_index}_{simulation_index}_model_out.npy`
- if the reservoir conn structure changes the conn dictionary needs to be adapted
- save weights and cov mat
- save train set and train labels
'''

EnKFOptimizeeParameters = namedtuple(
    'EnKFOptimizeeParameters', ['path',    # path to the result files
                                'ensemble_size',
                                'record_spiking_firingrate',
                                'save_plot',
                                'n_batches',       # n images to be shown
                                'n_test_batch',    # n test images to be shown
                                'stop_criterion',
                                'scale_weights', 'sample',
                                'best_n', 'worst_n', 'pick_method',
                                'data_loader_method', 'shuffle', 'n_slice',
                                'kwargs'],
    defaults=(True, False, 0.25, 0.25, 'random', 'separated', False, 4,
              {'pick_probability': 0.7, 'loc': 0, 'scale': 0.1})
)

EnKFOptimizeeParameters.__doc__ = """
:param gamma: float, A small value, multiplied with the eye matrix
:param maxit: int, Epochs to run inside the Kalman Filter
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
    Corresponds to number of ensembles
:param n_batches: int, Number of mini-batches to use for training
param n_repeat_batch: int, How often the same image batch should be shown
:param online: bool, Indicates if only one data point will used
    Default: False
:param scale_weights: bool, scales weights between [0, 1]
:param sampling: bool, If sampling of best individuals should be done
:param best_n: float, Percentage for best `n` individual. In combination with
    `sampling`. Default: 0.25
:param worst_n: float, Percentage for worst `n` individual. In combination with
    `sampling`. Default: 0.25
:param pick_method: str, How the best individuals should be taken. `random` or
    `best_first` must be set. If `pick_probability` is taken then a key
    word argument `pick_probability` with a float value is needed. `gaussian` 
    creates a multivariate normal distribution using the best individuals 
    which will replace the worst individuals. (see also :param kwargs) 
    In combination with `sampling`.
    Default: 'random'.
:param data_loader_method: str, Method on how to load the data set. 
    - `random`: Just randomly pick `n_batch` images from the whole data set.
    - 'separated': Separates the data set according to the test labels,
        e.g. [1,1,2,0,0,2] -> [[0,0],[1,1],[2,2]]. 
        See also l2l.optimizers.kalmanfilter.get_separated_data
:param shuffle: bool, if `True` shuffles the data and labels. In combination 
    with `data_loader_method`.  
        Default is False. 
        See also l2l.optimizers.kalmanfilter.get_separated_data
:param n_slice: int, How many slices are going to be taken.
        E.g. if target labels are `[0,1,2]` and `n_slice` 4 then the
        function will return `3 * 4 = 12` labels and corresponding data back. 
        The slice starts at the generation number. Used in combination with
        `shuffle=True`. Default is 4.
        See also l2l.optimizers.kalmanfilter.get_separated_data
:param kwargs: dict, key word arguments if `sampling` is True.
    - `pick_probability` - float, probability to pick the first best individual
      Default: 0.7
    - loc - float, mean of the gaussian normal, can be specified if 
      `pick_method` is `random` or `pick_probability`
      Default: 0.
    - scale - float, std scale of the gaussian normal, can be specified if 
      `pick_method` is `random` or `pick_probability` 
      Default: 0.1
:param seed: The random seed used to sample and fit the distribution. 
    Uses a random generator seeded with this seed.
:param path: String, Root path for the file saving and loading the connections
:param stop_criterion: float, When the current fitness is smaller or equal the 
    `stop_criterion` the optimization in the outer loop ends
"""


class EnKFOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.parameters = parameters
        self.fp = pathlib.Path(__file__).parent.absolute()
        config_path = os.path.join(str(self.fp), 'config.json')
        print(f"Config path: {config_path}")
        with open(config_path) as jsonfile:
            self.config = json.load(jsonfile)
        # Lists for labels
        self.target_labels = []
        self.random_ids = []
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx
        self.recording_firingrate = self.parameters.record_spiking_firingrate

        # MNIST DATA HANDLING
        self.target_label = ['0', '1', '2']
        self.other_label = ['3', '4', '5', '6', '7', '8', '9']
        self.train_set = None
        self.train_labels = None
        self.other_set = None
        self.other_labels = None
        self.test_set = None
        self.test_labels = None
        self.test_set_other = None
        self.test_labels_other = None
        self.optimizee_labels = None
        self.reservoir = Reservoir()
        self.rng = np.random.default_rng(self.gen_idx)
        self.data_loader_method = self.parameters.data_loader_method

        # dictionary for weights
        self.dict_weights = {}
        # Dictionary for connection lengths
        self.dict_conns = {}
        # batch file
        self.batchfile = 'batchfile.sh'
        self.types = ['eeo', 'eio', 'ieo', 'iio']
        self.test_gen = 10  # generation number for testing
        self.iterations = 10  # number of iterations the inner loop will take
        self.iteration_idx = 0

        # Hyper-parameters
        # place holder values, will be overwritten in create_individual
        self.max_ens_size = 30
        self.max_gamma = 1.0
        self.max_reps = 10
        self.gamma = 0.01
        self.ensemble_size = int(self.parameters.ensemble_size)
        self.repetitions = 3


    def get_mnist_data(self, mnist_path='/p/project/haf/users/yegenoglu/L2L/bin/mnist784_dat/'):
        self.train_set, self.train_labels, self.test_set, self.test_labels = \
            fetch(path=mnist_path, labels=self.target_label)
        self.other_set, self.other_labels, self.test_set_other, self.test_labels_other = \
            fetch(path=mnist_path, labels=self.other_label)

    def get_separated_data(self, train_labels, train_set, target_labels,
                           shuffle=False, gen_id=0, n_slice=4):
        """
        Separates the data set according to the test labels,
        e.g. [1,1,2,0,0,2] -> [[0,0],[1,1],[2,2]].
        It returns the separated data set and labels.

        :param train_labels: array_like, labels of the data set
        :param train_set: array_like, the data set itself
        :param target_labels: array_like, the labels/digits which are going to
            be separated
        :param gen_id: int, the generation index. Used in combination with `
            shuffle=True`. Default is 0.
        :param n_slice: int, how many slices are going to be taken.
            E.g. if target labels are `[0,1,2]` and `n_slice` 4 then the
            function will return `3 * 4 = 12` labels and corresponding data
            back. The slice starts at `gen_id * n_slice`. Used in combination
            with `shuffle=True`. Default is 4.
        :param shuffle: bool, if `True` shuffles the data and labels.
            Default is False.
        """
        data_set = []
        data_labels = []
        for tl in target_labels:
            indices = np.array(train_labels) == tl
            data_set.append(np.array(train_set)[indices])
            data_labels.append(np.array(train_labels)[indices])
        if shuffle:
            sliced_data = []
            sliced_labels = []
            index0 = gen_id * n_slice
            index1 = gen_id * n_slice + n_slice
            for i in range(len(target_labels)):
                len_data = len(data_set[i])
                sliced_data.append(data_set[i][index0 % len_data: index1 % len_data])
                len_labels = len(data_labels[i])
                sliced_labels.append(data_labels[i][index0 % len_labels: index1 % len_labels])
            # shuffle now
            sliced_labels = np.array(sliced_labels).ravel()
            shuffled_index = np.arange(sliced_labels.size)
            self.rng.shuffle(shuffled_index)
            # reshape so that the shuffling is possible according the index
            sliced_data = np.reshape(sliced_data, (shuffled_index.size, -1))
            return sliced_data[shuffled_index], sliced_labels[shuffled_index].astype(int)
        else:
            return data_set, data_labels

    @staticmethod
    def randomize_labels(labels, size):
        """
        Randomizes given labels `labels` with size `size`.

        :param labels: list of strings with labels
        :param size: int, size of how many labels should be returned
        :return list of randomized labels
        :return list of random numbers used to randomize the `labels` list
        """
        rnd = np.random.randint(low=0, high=len(labels), size=size)
        return [int(labels[i]) for i in rnd], rnd

    def sample_from_individuals(self, individuals, fitness, model_output,
                                best_n=0.25, worst_n=0.25,
                                pick_method='random',
                                **kwargs):
        """
        Samples from the best `n` individuals via different methods.

        :param individuals: array_like
            Input data, the individuals
        :param fitness: array_like
            Fitness array
        :param best_n: float
            Percentage of best individuals to sample from
        :param model_output, array like, model outputs of the best indiviudals
            will be used to replace the model outputs of the worst individuals
            in the same manner as the sampling
        :param worst_n:
            Percentage of worst individuals to replaced by sampled individuals
        :param pick_method: str
            Either picks the best individual randomly 'random' or it picks the
            iterates through the best individuals and picks with a certain
            probability `best_first` the first best individual
            `best_first`. In the latter case must be used with the key word
            argument `pick_probability`.  `gaussian` creates a multivariate
            normal using the mean and covariance of the best individuals to
            replace the worst individuals.
            Default: 'random'
        :param kwargs:
            'pick_probability': float
                Probability of picking the first best individual. Must be used
                when `pick_method` is set to `pick_probability`.
            'loc': float, mean of the gaussian normal, can be specified if
               `pick_method` is `random` or `pick_probability`
               Default: 0.
            'scale': float, std scale of the gaussian normal, can be specified if
                `pick_method` is `random` or `pick_probability`
                 Default: 0.1
        :return: array_like
            New array of sampled individuals.
        """
        # best fitness should be here ~ 1 (which means correct choice)
        # sort them from best to worst via the index of fitness
        # get indices
        indices = np.argsort(fitness)[::-1]
        sorted_individuals = np.array(individuals)[indices]
        # get best n individuals from the front
        best_individuals = sorted_individuals[:int(len(individuals) * best_n)]
        # get worst n individuals from the back
        worst_individuals = sorted_individuals[
            len(individuals) - int(len(individuals) * worst_n):]
        # sort model outputs
        sorted_model_output = model_output[indices]
        for wi in range(len(worst_individuals)):
            if pick_method == 'random':
                # pick a random number for the best individuals add noise
                rnd_indx = self.rng.integers(len(best_individuals))
                ind = best_individuals[rnd_indx]
                # add gaussian noise
                noise = self.rng.normal(loc=kwargs['loc'],
                                        scale=kwargs['scale'],
                                        size=len(ind))
                worst_individuals[wi] = ind + noise
                model_output[wi] = sorted_model_output[rnd_indx]
            elif pick_method == 'best_first':
                for bidx, bi in enumerate(best_individuals):
                    pp = kwargs['pick_probability']
                    rnd_pp = self.rng.random()
                    if pp >= rnd_pp:
                        # add gaussian noise
                        noise = self.rng.normal(loc=kwargs['loc'],
                                                scale=kwargs['scale'],
                                                size=len(bi))
                        worst_individuals[wi] = bi + noise
                        model_output[wi] = sorted_model_output[bidx]
                        break
            else:
                sampled = self._sample(best_individuals, pick_method)
                worst_individuals = sampled
                rnd_int = self.rng.integers(
                    0, len(best_individuals), size=len(best_individuals))
                model_output[len(sorted_individuals) -
                             len(worst_individuals):] = sorted_model_output[rnd_int]
                break
        sorted_individuals[len(sorted_individuals) -
                           len(worst_individuals):] = worst_individuals
        return sorted_individuals, model_output

    def _sample(self, individuals, method='gaussian'):
        if method == 'gaussian':
            dist = Gaussian()
            dist.init_random_state(self.rng.bit_generator)
            dist.fit(individuals)
            sampled = dist.sample(len(individuals))
        elif method == 'rv_histogram':
            sampled = [rv_histogram(h) for h in individuals]
        else:
            raise KeyError('Sampling method {} not known'.format(method))
        sampled = np.asarray(sampled)
        return sampled

    def create_batchfile(self, csv_path):
        with open(os.path.join(csv_path, self.batchfile), 'w') as f:
            reservoir_path = os.path.join(str(self.fp), 'reservoir_nest3.py')
            account = 'icei-hbp-2022-0007'
            tasks_node = 30
            cpu_task = 4
            print(f"Reservoir path: {reservoir_path}")
            batch = "#!/bin/bash \n" \
                    "#SBATCH --nodes=1 \n" \
                    f"#SBATCH --ntasks-per-node={tasks_node} \n" \
                    f"#SBATCH --cpus-per-task={cpu_task} \n" \
                    "#SBATCH --time=01:00:00 \n" \
                    "#SBATCH --partition=batch \n" \
                    f"#SBATCH --account={account} \n" \
                    "#SBATCH --gres=gpu:0 \n" \
                    "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} \n" \
                    "for ((x=0; x<$1; x++)) \n" \
                    "do \n" \
                    f"  srun -n 1 -c $2 python {reservoir_path} $3 $4 $5 $6 -si $x &\n" \
                    "done \n" \
                    "wait"
            f.write(batch)

    def create_ensembles(self):
        # check first if the connections exists and delete
        for t in self.types:
            conn = os.path.join(self.parameters.path, f'{t}_connections.csv')
            # if os.path.isfile(conn):
            #     os.remove(conn)
        # connect network
        self.execute_subprocess(self.parameters.path, simulation='--create')
        while True:
            if all([os.path.isfile(
                    os.path.join(self.parameters.path, f'{t}_connections.csv'))
                    for t in self.types]):
                break
            else:
                time.sleep(3)
        print('Creating the weights')
        mu = self.config['mu']
        sigma = self.config['sigma']
        size_eeo, size_eio, size_ieo, size_iio = self.get_connections_sizes(self.parameters.path)
        self.dict_conns['eeo'] = size_eeo
        self.dict_conns['eio'] = size_eio
        self.dict_conns['ieo'] = size_ieo
        self.dict_conns['iio'] = size_iio
        for i in range(self.ensemble_size):
            self.dict_weights[
                f'{self.ind_idx}_{i}_weights_eeo'] = np.random.normal(mu, sigma,
                                                                      size_eeo)
            self.dict_weights[
                f'{self.ind_idx}_{i}_weights_eio'] = np.random.normal(mu, sigma,
                                                                      size_eio)
            self.dict_weights[
                f'{self.ind_idx}_{i}_weights_ieo'] = np.random.normal(-mu,
                                                                      sigma,
                                                                      size_ieo)
            self.dict_weights[
                f'{self.ind_idx}_{i}_weights_iio'] = np.random.normal(-mu,
                                                                      sigma,
                                                                      size_iio)
            # finally, save weights
            self.save_weights(self.parameters.path, i,
                              iteration_idx=self.iteration_idx,
                              generation_idx=self.gen_idx)

    def execute_subprocess(self, csv_path, index='0', simulation='--create'):
        threads = int(self.config['threads'])
        batchfile_path = os.path.join(self.parameters.path, self.batchfile)
        print(f"Batch file path: {batchfile_path}")
        if simulation == '--create' or simulation == '-c':
            print('Create phase')
            sub = subprocess.run(
                ['sbatch', f'{batchfile_path}',
                 f'{1}', f'{threads}',
                 f'{simulation}',
                 f'--path={csv_path}',
                 f'--generation={self.gen_idx}',
                 f'--record_spiking_firingrate={self.recording_firingrate}',
                 ],
                check=True)
        else:
            print('Simulate phase')
            sub = subprocess.run([
                'sbatch', f'{batchfile_path}',
                f'{self.ensemble_size}', f'{threads}',
                f'{simulation}',
                f'--index={index}',
                f'--generation={self.gen_idx}',
                f'--path={csv_path}',
                f'--record_spiking_firingrate={self.parameters.record_spiking_firingrate}',
            ],
                                 check=True)
        sub.check_returncode()

    def create_individual(self):
        # normalize hyper-parameters before optimization 
        gamma = np.abs(self.rng.normal(0., 0.5))
        gamma = self.min_max_normalize(gamma, 0.009, self.max_gamma)
        ensemble_size = self.rng.integers(10, self.max_ens_size)
        ensemble_size = self.min_max_normalize(ensemble_size, 9, self.max_ens_size)
        repetitions = self.rng.integers(1, self.max_reps)
        repetitions = self.min_max_normalize(repetitions, 1, self.max_reps)
        print(f'Parameters in create individual ens {ensemble_size} reps {repetitions} gamma {gamma}')
        return {'gamma': gamma,
                'ensemble_size': ensemble_size,
                'repetitions': repetitions,
                # TODO sampling values
                }

    def bounding_func(self, individual):
        print(f"In bounding before clipping, ensemble: {individual['ensemble_size']}, "
              f"repetitions: {individual['repetitions']}, gamma: {individual['gamma']}")
        self.ensemble_size = np.clip(individual['ensemble_size'], 1e-4, 1.0)
        self.repetitions = np.clip(individual['repetitions'], 1e-4, 1.0)
        self.gamma = np.clip(individual['gamma'], 1e-4, 1.0)
        # self.gamma = self.min_max_normalize(self.gamma, 0.009, 0.9999)
        # self.ensemble_size = self.min_max_normalize(self.ensemble_size, 9, 30)
        # self.repetitions = self.min_max_normalize(self.repetitions, 1, 10)
        individual = {'gamma': self.gamma,
                      'ensemble_size': self.ensemble_size,
                      'repetitions': self.repetitions,
                      }
        print(f'In bounding, ensemble: {self.ensemble_size}, repetitions: {self.repetitions}, gamma: {self.gamma}')
        return individual

    def simulate(self, traj):
        # get indices
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx
        self.gamma = traj.individual.gamma
        self.ensemble_size = traj.individual.ensemble_size
        self.repetitions = traj.individual.repetitions
        if self.gen_idx >= 0:
            print(f'Parameters before re-normalization ens {self.ensemble_size} reps {self.repetitions} gamma {self.gamma}')
            # re-normalize the normalized parameters
            # ensembles
            self.ensemble_size = self.min_max_normalize(self.ensemble_size, min_val=9,
                                                        max_val=self.max_ens_size,
                                                        renormalize=True)
            # repetitions
            self.repetitions = self.min_max_normalize(self.repetitions, min_val=1,
                                                      max_val=self.max_reps,
                                                      renormalize=True)
            # gamma
            self.gamma = self.min_max_normalize(self.gamma, min_val=0.009,
                                                max_val=self.max_gamma,
                                                renormalize=True)
            print(f'Parameters after re-normalization ens {self.ensemble_size} '
                  f'reps {self.repetitions} gamma {self.gamma}')
            # self.ensemble_size = np.clip(self.ensemble_size, 9, 30)
            # self.repetitions = np.clip(self.repetitions, 1, 10)
            # self.gamma = np.clip(self.gamma, 0.01, 1.)

        # optimizer may make parameters as floats,
        # convert them to integers
        self.repetitions = int(self.repetitions)
        self.ensemble_size = int(self.ensemble_size)
        print(f'Hyperparameters in generation {self.gen_idx}, Individual {self.ind_idx}')
        print(f'Gamma: {self.gamma}, Repetitions: {self.repetitions}, Ensemble size: {self.ensemble_size}')
        if self.gen_idx == 0:
            # create the batch file if not existent
            if not os.path.isfile(
                    os.path.join(self.parameters.path, self.batchfile)):
                self.create_batchfile(csv_path=self.parameters.path)
            self.create_ensembles()
        # load the latest weights and construct the connection dictionary
        if self.gen_idx > 0:
            for i in range(self.ensemble_size):
                self.load_weights(csv_path=self.parameters.path,
                                  simulation_idx=i)
                size_eeo, size_eio, size_ieo, size_iio = self.get_connections_sizes(self.parameters.path)
                self.dict_conns['eeo'] = size_eeo
                self.dict_conns['eio'] = size_eio
                self.dict_conns['ieo'] = size_ieo
                self.dict_conns['iio'] = size_iio
        # get new data
        self.get_mnist_data()

        # Prepare for simulation
        n_output_clusters = self.config['n_output_clusters']
        fitnesses = None
        results = None
        # gamma needs to be an I-matrix multiplied by a scalar
        gamma = np.eye(n_output_clusters) * self.gamma

        # Training
        if self.gen_idx % self.test_gen != 0 or self.gen_idx == 0:
            enkf = EnsembleKalmanFilter(maxit=1,
                                        online=False,
                                        n_batches=traj.n_batches
                                        # len(self.optimizee_labels)
                                        )
            # Show i times the batch of images
            for i in range(self.iterations):
                # get new data for every repetition
                self.get_new_data(index=i)
                for r in range(self.repetitions):
                    enkf.n_batches = len(self.optimizee_labels)
                    for j in range(self.ensemble_size):
                        model_out = os.path.join(self.parameters.path,
                                                 f'{self.ind_idx}_{j}_model_out.npy')
                        # TODO: is this needed?
                        if os.path.isfile(model_out):
                            os.remove(model_out)
                    # save weights before simulation
                    # self.save_weights(csv_path=self.parameters.path,
                    #                   simulation_idx=j)
                    print('now execute simulate subprocess', flush=True)
                    self.execute_subprocess(csv_path=self.parameters.path,
                                            index=self.ind_idx, simulation='--simulate')
                    while True:
                        if all([os.path.isfile(os.path.join(self.parameters.path,
                                                            f'{self.ind_idx}_{idx}_model_out.npy'))
                                for idx in range(self.ensemble_size)]):
                            break
                        else:
                            time.sleep(3)

                    # obtain the individual model outputs
                    model_out = [np.load(os.path.join(
                        self.parameters.path, f'{self.ind_idx}_{idx}_model_out.npy'),
                        allow_pickle=True).squeeze()
                        for idx in range(self.ensemble_size)]
                    # apply softmax on model outs
                    # model outs shape: (ensemble_size x labels x n_output_clusters)
                    model_out = softmax(model_out, -1)
                    # get fitness
                    fitnesses = self.get_fitness(
                        n_output_clusters=n_output_clusters, model_outs=model_out)

                    # EnKF fit
                    # concatenate weights
                    weights = [np.concatenate(
                        (self.dict_weights[f'{self.ind_idx}_{i}_weights_eeo'],
                         self.dict_weights[f'{self.ind_idx}_{i}_weights_eio'],
                         self.dict_weights[f'{self.ind_idx}_{i}_weights_ieo'],
                         self.dict_weights[f'{self.ind_idx}_{i}_weights_iio']))
                        for i in range(self.ensemble_size)]
                    ens = np.array(weights)
                    if self.parameters.scale_weights:
                        ens = ens / np.abs(ens).max()
                    if self.parameters.sample:
                        ens, model_out = self.sample_from_individuals(
                            individuals=ens,
                            model_output=model_out,
                            fitness=fitnesses,
                            pick_method=self.parameters.pick_method,
                            best_n=self.parameters.best_n,
                            worst_n=self.parameters.worst_n,
                            # / (self.g % traj.n_repeat_batch /2 + 1),
                            **self.parameters.kwargs)
                    # reshape the model outs
                    model_out = model_out.reshape((self.ensemble_size,
                                                  n_output_clusters,
                                                  len(self.optimizee_labels)))
                    enkf.fit(ensemble=np.array(ens),
                             model_output=model_out,
                             ensemble_size=self.ensemble_size,
                             observations=np.array(self.optimizee_labels),
                             gamma=gamma)
                    results = enkf.ensemble.cpu().numpy()
                    # apply scaling
                    if self.parameters.scale_weights:
                        results = results * np.abs(weights).max()
                    # save new results into weights dictionary
                    for j in range(self.ensemble_size):
                        self.restructure_weight_dict(w=results,
                                                     simulation_index=j)
                    # save new weights after optimization step is done
                    for j in range(self.ensemble_size):
                        self.save_weights(csv_path=self.parameters.path,
                                          simulation_idx=j,
                                          iteration_idx=self.iteration_idx,
                                          generation_idx=self.gen_idx)
                    self.iteration_idx = i

            # store results after repetitions
            # if self.gen_idx % 9 == 0 and self.gen_idx > 0:
            self.save_results(fitness=fitnesses, cov_mat=enkf.cov_mat,
                              weights=results, test=False)

        # Testing 
        elif self.gen_idx % self.test_gen == 0 and self.gen_idx > 0:
            if self.test_labels:
                testset = self.test_set[:len(self.optimizee_labels)]
                testlabels = [int(t) for t in self.test_labels[:len(self.optimizee_labels)]]
                print(f'Testing in generation {self.gen_idx}, individual {self.ind_idx}')
                print(f'Testing for labels: {testlabels}')
                # save test set and test labels
                self.save_data_set(file_path=self.parameters.path,
                                   trainset=testset,
                                   targets=testlabels,
                                   generation=self.gen_idx)

            self.execute_subprocess(csv_path=self.parameters.path,
                                    index=self.ind_idx, simulation='--simulate')
            while True:
                if all([os.path.isfile(os.path.join(self.parameters.path,
                                                    f'{self.ind_idx}_{idx}_model_out.npy'))
                        for idx in range(self.ensemble_size)]):
                    break
                else:
                    time.sleep(3)
            model_outs = [np.load(os.path.join(
                self.parameters.path, f'{self.ind_idx}_{idx}_model_out.npy'),
                                 allow_pickle=True).squeeze()
                         for idx in range(self.ensemble_size)]
            # apply softmax on model outs
            # ensemble_size x labels x n_output_clusters
            model_outs = softmax(model_outs, -1)
            fitnesses = self.get_fitness(n_output_clusters=n_output_clusters,
                                         model_outs=model_outs)
            # Save test results
            self.save_results(fitness=fitnesses, weights=None, cov_mat=None,
                              test=True)
        self.dict_weights.clear()
        if fitnesses.size == 0:
            raise AttributeError(f'No fitness obtained - '
                                 f'Got instead : {fitnesses}')

        # # normalize hyper-parameters before optimization
        # self.gamma = self.min_max_normalize(self.gamma, 0.009, 0.9999)
        # self.ensemble_size = self.min_max_normalize(self.ensemble_size, 9, 30)
        # self.repetitions = self.min_max_normalize(self.repetitions, 1, 10)
        print(f'Parameters before optimization ens {self.ensemble_size} '
              f'reps {self.repetitions} gamma {self.gamma}')
        return np.mean(fitnesses),

    def save_weights(self, csv_path, simulation_idx, generation_idx,
                     iteration_idx, failed=False):
        """
        Saves the weights and connections in a csv file
        """
        # if iteration 0 skip to write any weights unless generation_idx=0
        # however write the new individuals
        if iteration_idx == 0 and generation_idx > 0 and not failed:
            return
        elif iteration_idx == 0 and generation_idx > 0 and failed:
            for typ in self.types:
                # add new columns of weights
                key = f'{self.ind_idx}_{simulation_idx}_weights_{typ}'
                # check if file with weights exists already, skip
                if os.path.exists(os.path.join(csv_path, key, '.csv')):
                    return
                # if file does not exist, save in conns dict and write
                else:
                    conns = pd.read_csv(os.path.join(csv_path,
                                                     f'{typ}_connections.csv'))
                    conns['weights'] = self.dict_weights.get(key)
                    # write file with connections and new weights
                    conns.to_csv(os.path.join(csv_path, f'{key}.csv'))
        # For the other cases normally continue
        # Read the connections
        for typ in self.types:
            conns = pd.read_csv(os.path.join(csv_path,
                                             '{}_connections.csv'.format(typ)))
            # add new columns of weights
            key = f'{self.ind_idx}_{simulation_idx}_weights_{typ}'
            conns['weights'] = self.dict_weights.get(key)
            # remove if file with weights exists already
            if os.path.exists(os.path.join(csv_path, key, '.csv')):
                os.remove(os.path.join(csv_path, key))
            # write file with connections and new weights
            conns.to_csv(os.path.join(csv_path, f'{key}.csv'))

    def save_results(self, fitness, cov_mat, weights, test=False):
        dir_name = f"results_{os.environ['SLURM_JOB_ID']}"
        results_path = os.path.join(self.parameters.path, dir_name)
        if not os.path.exists(results_path):
            try:
                os.mkdir(results_path)
            except FileExistsError as fe:
                print(fe)
        key = f"generation_{self.gen_idx}/individual_{self.ind_idx}"
        if test:
            print(f'Saving test results in {results_path}')
            np.savez_compressed(os.path.join(results_path,
                                             f'{self.gen_idx}_{self.ind_idx}_test_results.npz'),
                                fitness=fitness)
            # with h5py.File(os.path.join(results_path, "test_results.hdf5"), "a") as f:
            #     f.create_dataset(f"{key}/fitness", data=fitness)
        else:
            print(f'Saving training results in {results_path}')
            np.savez_compressed(os.path.join(results_path,
                                             f'{self.gen_idx}_{self.ind_idx}_results.npz'),
                                fitness=fitness, Cpp=cov_mat['Cpp'], Cup=cov_mat['Cup'],
                                weights=weights)
            # with h5py.File(os.path.join(results_path, "train_results.hdf5"), "a") as f:
            #     f.create_dataset(f"{key}/fitness", data=fitness)
            #     f.create_dataset(f"{key}/cov_mat/CPP", data=cov_mat['Cpp'])
            #     f.create_dataset(f"{key}/cov_mat/CUP", data=cov_mat['Cup'])
            #     f.create_dataset(f"{key}/weights", data=weights)

    def load_weights(self, csv_path, simulation_idx):
        """
        Load weights and saves them in a class member dictionary `dict_weights`
        """
        failed = False
        for typ in self.types:
            key = f'{self.ind_idx}_{simulation_idx}_weights_{typ}'
            try:
                conns = pd.read_csv(os.path.join(csv_path, key+'.csv'))
                # will be cleared later for new generation
                self.dict_weights[key] = conns['weights'].values
            except FileNotFoundError:
                failed = True
                # create new ensemble by randomly picking from the ensembles
                # and perturbing it
                print(f'File {key} could not be found. Creating new ensemble',
                      flush=True)

                # create a function to split strings via "_"
                # and take 2. argument
                def func(x, idx=1):
                    return int(x.split('_')[idx])
                files = [f for f in os.listdir(csv_path) if
                         f.endswith('.csv') and f.startswith(
                             f'{self.ind_idx}')]
                # sometimes there may be empty lists, if in case use a
                # random weight csv
                if len(files) == 0:
                    files = [f for f in os.listdir(csv_path) if
                             f.endswith('.csv')]
                random_filename = self.rng.choice(files)
                if len(files) == 0:
                    ind_idx = func(random_filename, 0)
                else:
                    ind_idx = self.ind_idx
                # get simulation index from the random filename
                sim_ind = func(random_filename)
                tmp_key = f'{ind_idx}_{sim_ind}_weights_{typ}'
                if tmp_key in self.dict_weights:
                    weights = self.dict_weights[tmp_key]
                else:
                    print(f'Not in dictionary, loading key {tmp_key}.csv '
                          f'directly from file')
                    # cast pandas to numpy array
                    weights = pd.read_csv(os.path.join(csv_path, tmp_key +'.csv'))
                noise = self.rng.normal(loc=self.parameters.kwargs['loc'],
                                        scale=self.parameters.kwargs['scale'],
                                        size=len(weights))
                new_ensemble = weights + noise
                self.dict_weights[key] = new_ensemble
            # only save if exception occurs
            if failed:
                # save weight
                self.save_weights(csv_path, simulation_idx,
                                  iteration_idx=self.iteration_idx,
                                  generation_idx=self.gen_idx,
                                  failed=failed)

    def restructure_weight_dict(self, w, simulation_index):
        def masked(x, mu=1000, sigma=200):
            mask = x > mu
            x[mask] = np.random.normal(mu, sigma, np.count_nonzero(mask))
            mask = x < -mu
            x[mask] = np.random.normal(-mu, sigma, np.count_nonzero(mask))
            return x

        length = 0
        for typ in self.types:
            key = f'{self.ind_idx}_{simulation_index}_weights_{typ}'
            weights = w[simulation_index][length:self.dict_conns[typ] + length]
            # bound the weights
            weights = masked(weights)
            self.dict_weights[key] = weights
            length += self.dict_conns[typ]

    @staticmethod
    def save_data_set(file_path, trainset, targets, generation):
        """
        Saves the dataset and label as a numpy file per generation.
        If the file already exists the data is not created.
        """
        filename = f"{generation}_dataset.npy"
        if not os.path.exists(filename):
            np.save(os.path.join(file_path, filename),
                    {'train_set': trainset, 'targets': targets},
                    allow_pickle=True)

    def get_connections_sizes(self, csv_path):
        sizes = []
        for typ in self.types:
            con = pd.read_csv(os.path.join(
                csv_path, '{}_connections.csv'.format(typ)))
            sizes.append(len(con['source']))
        return tuple(sizes)

    @staticmethod
    def apply_softmax(model_outs, n_output_clusters):
        mos = []
        argmax = []
        # obtain the individual model outputs and apply softmax on them
        for model_out in model_outs:
            softm = softmax([np.mean(model_out[j]) for j in
                             range(n_output_clusters)])
            argmax.append(np.argmax(softm))
            mos.append(softm)
        return mos, argmax

    def get_fitness(self, n_output_clusters, model_outs):
        fitnesses = []
        argmax = np.argmax(model_outs, -1)  # ensemble_size x labels
        for i, target in enumerate(self.optimizee_labels):
            # one hot encoding
            label = np.eye(n_output_clusters)[target]
            pred = np.eye(n_output_clusters)[argmax[:, i]]
            # MSE of 1 is worst 0 is best, that's why 1 - fitness for L2L
            fitness = 1 - self._calculate_fitness(label, pred, "MSE", axis=1)
            fitnesses.append(fitness)
            print('Fitness {} for target {}, argmax {}'.format(
                fitness, target, argmax[:, i]))
        return np.mean(fitnesses, 0)

    @staticmethod
    def _calculate_fitness(label, prediction, costf='MSE', axis=None):
        if costf == 'MSE':
            return ((label - prediction) ** 2).mean(axis)

    @staticmethod
    def min_max_normalize(x, min_val, max_val, renormalize=False):
        if renormalize:
            return x*(max_val - min_val) + min_val
        else:
            return (x - min_val) / (max_val - min_val)

    def clean_files(self, csv_path):
        """
        Cleans csv and npy files produced by the optimizee
        """
        for f in os.listdir(csv_path):
            try:
                if f.endswith(('.csv', '.npy')) and not f.startswith(
                        tuple(self.types)) and not f.startswith('results'):
                    if os.path.isfile(os.path.join(csv_path, f)):
                        print(f'Removing {f}')
                        os.remove(os.path.join(csv_path, f))
            except OSError as ose:
                print(ose)

    def get_new_data(self, index=0):
        if self.train_labels:
            trainset = None
            if self.data_loader_method == 'random':
                self.optimizee_labels, self.random_ids = self.randomize_labels(
                    self.train_labels, size=self.parameters.n_batches)
                trainset = [self.train_set[r] for r in self.random_ids]
            elif self.data_loader_method == 'separated':
                self.optimizee_data, self.optimizee_labels = self.get_separated_data(
                    self.train_labels, self.train_set, self.target_label,
                    gen_id = self.gen_idx * self.repetitions + index,
                    shuffle=self.parameters.shuffle,
                    n_slice=self.parameters.n_slice)
                trainset = self.optimizee_data
                if not self.parameters.shuffle:
                    rand_ind = self.rng.integers(0, len(self.target_label), 1)[0]
                    self.optimizee_labels, self.random_ids = self.randomize_labels(
                        self.optimizee_labels[rand_ind], size=self.parameters.n_batches)
                    trainset = [self.optimizee_data[rand_ind][r] for r in
                                self.random_ids]
        else:
            raise AttributeError('Train Labels are not set, please check.')
        # all individuals/simulations are getting the same batch of data
        self.save_data_set(file_path=self.parameters.path,
                           trainset=trainset,
                           targets=self.optimizee_labels,
                           generation=self.gen_idx)


def randomize_labels(labels, size):
    rnd = np.random.randint(low=0, high=len(labels), size=size)
    return [int(labels[i]) for i in rnd], rnd
