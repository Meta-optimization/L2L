import os
import numpy as np

import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .helpers import create_config
from dwave.cloud.config import load_config
from dwave.cloud.client import Client
from dimod import DiscreteQuadraticModel
import itertools
from dwave.system import LeapHybridDQMSampler
import networkx as nx

HybridCommunityOptimizeeParameters = namedtuple(
    'HybridCommunityOptimizeeParameters', ['APIToken', 'config_path', 'num_partitions', 'Graph','result_path'])


class HybridCommunityOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.num_partitions = parameters.num_partitions
        self.G = parameters.Graph
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
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
        individual = {'num_partitions': self.num_partitions}
        return individual
    

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'num_partitons': np.clip(individual['num_partitions'], a_min=1, a_max=50)}

    def simulate(self, traj):
        """
        Performs community detection on the graph using the D-Wave Quantum Computer.

        Args:
        - traj: The trajectory object containing the individual's parameters.

        Returns:
        - A tuple containing the fitness value (1/modularity) of the clustering.
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        # Load the configuration from the specified path
        config = load_config(self.config_path)
        print(config) 

        # Define the range of partitions for the clustering
        partitions = range(int(traj.individual.num_partitions))


        B = nx.modularity_matrix(self.G)

        # Initialize the Discrete Quadratic Model (DQM) object
        dqm = DiscreteQuadraticModel()

        # Add variables to the DQM for each node in the graph
        for i in self.G.nodes():
            dqm.add_variable(int(traj.individual.num_partitions), label=i)

        # Set the quadratic terms for the DQM
        for i in self.G.nodes():
            for j in self.G.nodes():
                if i == j:
                    continue  # Skip the linear term in QUBO/Ising formulation
                dqm.set_quadratic(i, j, {(c, c): ((-1)*B[i, j]) for c in partitions})

        try:
            # Sample a DQM using LeapHybridDQMSampler and retrieve the best solution
            client = Client.from_config(config_file=self.config_path)
            sampler = LeapHybridDQMSampler()
            sampleset = sampler.sample_dqm(dqm, time_limit=10, label='community detection')
            run_time = (sampleset.info['run_time'])*0.001
            best_sample = sampleset.first.sample
            client.close()

        except Exception as e:
            with open(self.result_path, "a", encoding="utf-8") as f:
                import traceback
                f.write("An error occurred")
                traceback.print_exc(file=f)

        # Count the nodes in each partition
        counts = np.zeros(int(traj.individual.num_partitions))

        # Create communities as a parameter for the evaluation function
        communities = []
        for k in partitions:
            comm = []
            for i in best_sample:
                if best_sample[i] == k:
                    comm.append(i)
            communities.append(set(comm))

        # Compute the modularity of the clustering
        modularity = nx.community.modularity(self.G, communities)

        # Write the results to the result file
        with open(self.result_path, "a", encoding="utf-8") as f:
            f.write(f"Sampling time: {run_time:.2f} ms \n")
            f.write(f'Generation: {self.generation}, Individual: {self.ind_idx} \n')
            f.write(f'Best sample: {best_sample} \n')
            f.write(f'Communities: {communities} \n')
            f.write(f'Modularity: {modularity} \n\n')

        # Compute the fitness value (1/modularity) for the clustering
        # Note: The fitness value is the inverse of the modularity because the
        # optimization algorithm is minimizing the fitness value, but higher
        # modularity values are better.
        if modularity > 0:
            fitness = 1/modularity
        else:
            fitness = 100 

        return (fitness, )
    


    
    
    




