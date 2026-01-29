"""
Optimizee combining Arbor and SBI Optimizee
"""

import numpy as np
from .prior import prior, labels, x_obs

from l2l.optimizees.optimizee import Optimizee

class ArborOptimizee(Optimizee):
    """
    ...
    """

    def __init__(self, traj):
        """
        ...
        """
        super().__init__(traj)

    def create_individual(self, n=1, prior=prior):
        """
        ...
        """
        samples = prior.sample((n,))
        pop = [dict(zip(labels, sample)) for sample in samples]
        if n == 1:
            return pop[0], prior # TODO okay?
        return pop, samples

    def bounding_func(self, individual):
        """
        ...
        """
        return individual # prior is already bounded

    def get_features(self, ts, um):
        # resting potential
        u_rest = um[30_000] # at t=150 ms

        # spike identification
        spike_idx = []
        for i, u in enumerate(um[1:], 1):
            if u > -20 and um[i-1] <= -20:
                spike_idx.append(i)
        spike_idx = np.array(spike_idx)
        if len(spike_idx) > 0:
            spike_ts = ts[spike_idx]
        else:
            spike_ts = []

        # initial spike time
        if len(spike_ts) > 0:
            first_spike_t = spike_ts[0]*1e-3
        else:
            first_spike_t = 0

        # spike frequency/average firing rate
        avg_rate = len(spike_ts)/(ts[-1]-200)*1e3

        # average inter-spike interval and trough height
        if len(spike_ts) > 1:
            avg_isi = np.mean(spike_ts[1:]-spike_ts[:-1])*1e-3
            through_height = np.mean(um[(spike_idx[1:]+spike_idx[:-1])//2])
        else:
            avg_isi = 0
            through_height = 0

        # action potential height and width
        if len(spike_ts) > 0:
            ap_height = np.mean([np.max(um[i:i+1000]) for i in spike_idx]) # i:i+1000 means next 5 ms

            val = (u_rest+ap_height)//2
            start = 0
            tmp = []
            for i, u in enumerate(um[1:], 1):
                if u > val and um[i-1] <= val:
                    start = ts[i]
                    continue
                if um[i-1] > val and u <= val:
                    tmp.append(ts[i]-start)
            ap_width = np.mean(tmp)
        else:
            ap_height = 0
            ap_width = 0

        return (u_rest, first_spike_t, avg_rate, avg_isi, through_height, ap_height, ap_width)

    def simulate(self, traj):
        """
        ...
        """
        from .interface import load_ref, load_params, ArborParRunner

        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        ref = load_ref('/p/home/jusers/todt1/jusuf/slns/L2L/l2l/optimizees/arbor/ref.csv') # Vergleichskurve ("observation")
        par = load_params('/p/home/jusers/todt1/jusuf/slns/L2L/l2l/optimizees/arbor/fit.json')
        opt = ArborParRunner()

        # replace parameters
        for label in labels:
            par[label] = float(traj.individual[label])

        ts, um = opt.run(par)

        # summary statistics
        obs = self.get_features(ts, um)

        fitness = -np.mean(np.square(um-ref['U/mV'][:-1])) # mse

        return (fitness, obs)
