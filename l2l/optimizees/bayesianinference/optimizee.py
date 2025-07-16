from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .prior import prior, labels, x_obs
import torch
import numpy as np
import os.path
import os
import time

SBIOptimizeeParameters = namedtuple('SBIOptimizeeParameters', [])

import dill

def save_obj(obj, path):
    with open(path, 'wb') as handle:
        dill.dump(obj, handle)

def load_obj(path):
    with open(path, "rb") as handle:
        obj = dill.load(handle)
    return obj

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
        self.nrec = 500
        self.scale = 10
        self.COUNT = 0

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
        import nest # import mpi4py before or after?
        from mpi4py import MPI

        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

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
                                   nrec=self.nrec,
                                   ev_name='test',
                                   seed=self.generation*1234+1
                                   )
        events = net.run_simulation()

        if type(events) == float:
            return (np.nan, (np.nan, np.nan, np.nan, np.nan))

        # MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        use_mpi = False
        if use_mpi:

            events = comm.gather(events, root=0)

            if rank == 0:
                obs = self.calculate_observation(events)
                average_rate = obs[0]
                desired_rate = x_obs[0]
                fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
                print(f'gen {self.generation} ind {self.ind_idx} rate {average_rate} fitness {fitness}')
                print(obs)
                if np.isnan(obs).any():
                    fitness = np.nan
                return (fitness, obs)

        else:
            events_path = f'/p/scratch/cslns/todt1/L2L/results/events/events_{self.generation}_{self.ind_idx}.dill'
            if rank == 0:
                while not os.path.isfile(events_path):
                    time.sleep(1)
                events2 = load_obj(events_path)
                events = [events, events2]
                os.remove(events_path)

                obs = self.calculate_observation(events)
                average_rate = obs[0]
                desired_rate = x_obs[0]
                fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
                print(f'gen {self.generation} ind {self.ind_idx} rate {average_rate} fitness {fitness}')
                print(obs)
                if np.isnan(obs).any():
                    fitness = np.nan
                return (fitness, obs)
            else:
                save_obj(events, events_path)

    def calculate_observation(self, events_list):
        from nest.raster_plot import extract_events

        t_start = 300
        t_stop = 10300
        rates = []
        intervals = []
        ff_times = []

        for events in events_list:
            senders = events['senders']
            times = events['times']
            data = np.stack([senders, times], axis=1)

            spike_trains = self.extract_time(data, t_start, t_stop)

            rates += [self.mean_firing_rate(st) for st in spike_trains]
            intervals += [self.inter_spike_intervals(st) for st in spike_trains]
            if len(intervals[-1]) > 0:
                ff_times.append(extract_events(data, time=[t_start, t_stop])[:,1])
        if len(ff_times) > 0:
            ff_times = np.concatenate(ff_times)

        rate_mean = np.mean(rates)
        rate_std = np.std(rates)
        if len(intervals) > 0:
            isi_v = np.mean([np.std(i)/np.mean(i) for i in intervals])
        else:
            isi_v = 0

        # fano factor
        if len(ff_times) > 0:
            bin_width = 10
            bins = list(range(t_start, t_stop+1, bin_width))
            hist, ret_bins = np.histogram(ff_times, bins=bins)
            #hist = 1000.*hist/(bin_width*500) # rate
            h_mean = np.mean(hist)
            if abs(h_mean) < 1e-3:
                ff = 0
            else:
                ff = np.var(hist)/np.mean(hist)
        else:
            ff = 0

        if np.isnan(isi_v):
            isi_v = 0
        if np.isnan(ff):
            ff = 0

        return [rate_mean, rate_std, isi_v, ff]

    def inter_spike_intervals(self, times):
        if len(times) < 2:
            return []
        return times[1:]-times[:-1]

    def mean_firing_rate(self, times, simtime=10_000):
        return 1000.*len(times)/(1.*simtime)

    def extract_time(self, data, t_start, t_stop, verbose=False):
        from nest.raster_plot import extract_events
        spike_trains = []
        for sender in set(data[:,0]):
            tmp = extract_events(data, time=[t_start, t_stop], sel=[sender])
            if len(tmp) == 0:
                spike_trains.append([])
                continue
            spike_trains.append(tmp[:,1])
        return spike_trains

    def bounding_func(self, individual):
        """
        placeholder
        """
        return individual
