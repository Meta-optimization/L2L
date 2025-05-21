import os
import numpy as np
import math
import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .helpers import create_config, get_distance, get_labels_from_sample, fix_sample_one_hot, is_valid_one_hot
from dwave.cloud.config import load_config
from dwave.cloud.client import Client
from sklearn.metrics import calinski_harabasz_score
import dimod
import itertools
from dwave.system import EmbeddingComposite, DWaveSampler,LeapHybridSampler

HybridClusteringOptimizeeParameters = namedtuple(
    'HybridClusteringOptimizeeParameters', ['APIToken', 'config_path', 'alpha', 'gamma', 'delta', 'one_hot_strength',
                                      'points', 'num_clusters', 'result_path'])


class HybridClusteringOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.points = parameters.points
        self.num_clusters = parameters.num_clusters
        self.alpha = parameters.alpha
        self.gamma = parameters.gamma
        self.delta = parameters.delta
        self.one_hot_strength = parameters.one_hot_strength
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generatio
        self.config_path = os.path.join(parameters.config_path, "dwave.conf")
        os.makedirs(parameters.result_path, exist_ok=True)
        self.result_path = os.path.join(parameters.result_path, "result.txt")
        if not os.path.exists(self.config_path):
            create_config(parameters.APIToken, parameters.path)

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {"alpha": self.alpha, "gamma": self.gamma, 
                      "delta": self.delta, "one_hot_strength": self.one_hot_strength}
        return individual
    

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'alpha': np.clip(individual['alpha'], a_min=0, a_max=50),
                'gamma': np.clip(individual['gamma'], a_min=0, a_max=50),
                'delta': np.clip(individual['delta'], a_min=0, a_max=50),
                'one_hot_strength': np.clip(individual['one_hot_strength'], a_min=0, a_max=50*len(self.points))}

    def simulate(self, traj):
        """
        Does Clustering
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        config = load_config(self.config_path)
        print(config)

        num_points = len(self.points)
        max_distance = max(get_distance(a, b) for a, b in itertools.combinations(self.points, 2))

        # Define variables for each point and cluster
        variables = {(i, c): f"x_{i}_{c}" for i in range(num_points) for c in range(self.num_clusters)}
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype='BINARY')

        ## One-hot constraints: ensure each point is assigned to exactly one cluster
        for i in range(num_points):
            vars_i = [variables[(i, c)] for c in range(self.num_clusters)]
            for v in vars_i:
                bqm.add_variable(v, -1*traj.individual.one_hot_strength)  # Bias for assignment
            for v1, v2 in itertools.combinations(vars_i, 2):
                bqm.add_interaction(v1, v2, 2*traj.individual.one_hot_strength)  # Penalize multiple assignments to the same point

        # Attraction: points close together should be assigned to the same cluster
        # Encourage nearby points to be in the same cluster
        for (i, p0), (j, p1) in itertools.combinations(enumerate(self.points), 2):
            d = get_distance(p0, p1) / max_distance
            same_cluster_weight = -math.cos(traj.individual.alpha * d * math.pi)

            for c in range(self.num_clusters):
                var1 = variables[(i, c)]
                var2 = variables[(j, c)]
                # Encourage same cluster for close points
                bqm.add_interaction(var1, var2, same_cluster_weight)

            # Encourage far-apart points to be in different clusters
            d_far = math.sqrt(get_distance(p0, p1) / max_distance)
            different_cluster_weight = -math.tanh(traj.individual.gamma * d_far) * traj.individual.delta

            for c1 in range(self.num_clusters):
                for c2 in range(self.num_clusters):
                    if c1 != c2:
                        var1 = variables[(i, c1)]
                        var2 = variables[(j, c2)]
                        bqm.add_interaction(var1, var2, different_cluster_weight)

        try:
            # code that uses client
            client = Client.from_config(config_file=self.config_path)
            sampler = LeapHybridSampler()

            start = time.perf_counter()
            sampleset = sampler.sample(bqm,
                                    label='Hybrid-L2L')
            end = time.perf_counter()

            wall_time_ms = (end - start) * 1000
            #qpu_access_time_ms = sampleset.info['timing']['qpu_access_time'] / 1000
            #queue_time_ms = wall_time_ms - qpu_access_time_ms

            best_sample = None

            for sample, energy in zip(sampleset.samples(), sampleset.record['energy']):
                if is_valid_one_hot(sample, num_points, self.num_clusters, variables):
                    best_sample = sample
                    break

            if best_sample is None:
                print("⚠️ Kein gültiges Sample gefunden. Wende Postprocessing an.")
                best_sample = fix_sample_one_hot(sampleset.first.sample, num_points, self.num_clusters, variables)
            client.close()
        except Exception as e:
            with open(self.result_path, "a", encoding="utf-8") as f:
                import traceback
                f.write("an error occuerd")
                traceback.print_exc(file=f)

        #safe results
        labels = get_labels_from_sample(best_sample, len(self.points), self.num_clusters)
        with open(self.result_path, "a", encoding="utf-8") as f:
            f.write(f"Sampling time: {wall_time_ms:.2f} ms \n")
            #f.write(f"QPU access time: {qpu_access_time_ms:.2f} ms \n")
            #f.write(f"Estimated queueing/host overhead: {queue_time_ms:.2f} ms \n")
            f.write(f'Generation: {self.generation}, Individual: {self.ind_idx} \n')
            f.write(f'best sample: {best_sample} \n')
            f.write(f'points: {self.points} \n')
            f.write(f'labels: {labels} \n\n')

        #fitness = len(solvers)
        return (1/calinski_harabasz_score(self.points, labels), )
    


    
    
    




