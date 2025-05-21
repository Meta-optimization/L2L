import os
import numpy as np
import math
import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .helpers import create_config
from dwave.cloud.config import load_config
from dwave.cloud.client import Client
from sklearn.metrics import calinski_harabasz_score
import dimod
import itertools
from dwave.system import EmbeddingComposite, DWaveSampler

CommunityOptimizeeParameters = namedtuple(
    'CommunityOptimizeeParameters', ['APIToken', 'config_path', 'result_path'])


class CommunityOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.bound = [0, 2000]
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
        individual = {}
        return individual
    

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {}

    def simulate(self, traj):
        """
        Does Clustering
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        config = load_config(self.config_path)
        print(config)


        try:
            client = Client.from_config(config_file=self.config_path)
            # code that uses client
            solvers = client.get_solvers()
            solver = solvers[5]

            start = time.perf_counter()
            sampler = EmbeddingComposite(DWaveSampler(solver=solver.id))
            end = time.perf_counter()
            embedding_time_ms = (end-start)*1000

            start = time.perf_counter()
            sampleset = ...
            end = time.perf_counter()

            wall_time_ms = (end - start) * 1000
            qpu_access_time_ms = sampleset.info['timing']['qpu_access_time'] / 1000
            queue_time_ms = wall_time_ms - qpu_access_time_ms

            best_sample = None

            client.close()
        except:
            print("error")

        #safe results
        labels = get_labels_from_sample(best_sample, len(self.points), self.num_clusters)
        with open(self.result_path, "a", encoding="utf-8") as f:
            f.write(f"Embedding time: {embedding_time_ms:.2f} ms \n")
            f.write(f"Sampling time: {wall_time_ms:.2f} ms \n")
            f.write(f"QPU access time: {qpu_access_time_ms:.2f} ms \n")
            f.write(f"Estimated queueing/host overhead: {queue_time_ms:.2f} ms \n")
            f.write(f'Generation: {self.generation}, Individual: {self.ind_idx} \n')
            f.write(f'points: {self.points} \n')
            f.write(f'labels: {labels} \n\n')

        #fitness = len(solvers)
        return (1/1, ) 
    


    
    
    




