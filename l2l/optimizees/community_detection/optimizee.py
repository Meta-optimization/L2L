import os
import numpy as np
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .helpers import create_config
from dwave.cloud.client import Client
import networkx as nx
from dimod import BinaryQuadraticModel
from itertools import combinations
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import time


CommunityOptimizeeParameters = namedtuple(
    'CommunityOptimizeeParameters', ['APIToken', 'config_path', 'num_partitions', 'Graph','result_path'])


class CommunityOptimizee(Optimizee):
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
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        partitions = range(int(traj.individual.num_partitions))
        B = nx.modularity_matrix(self.G)
        bqm = BinaryQuadraticModel('BINARY')
        one_hot_strength = 3.0

        for i in self.G.nodes():
            for c in partitions:
                var = f"{i}__{c}"
                bqm.add_variable(var, 0.0) 


        for i in self.G.nodes():
            for j in self.G.nodes():
                if i == j:
                    continue
                for c in partitions:
                    vi = f"{i}__{c}"
                    vj = f"{j}__{c}"
                    bqm.add_interaction(vi, vj, -B[i, j])

        for i in self.G.nodes():
            vars_i = [f"{i}__{c}" for c in partitions]
            for v in vars_i:
                bqm.add_variable(v, bqm.get_linear(v) + one_hot_strength)
            for u, v in combinations(vars_i, 2):
                bqm.add_interaction(u, v, 2 * one_hot_strength)
            for v in vars_i:
                bqm.add_variable(v, bqm.get_linear(v) - 2 * one_hot_strength)
            bqm.offset += one_hot_strength

        try:
            client = Client.from_config(config_file=self.config_path)
            """sampler = EmbeddingComposite(DWaveSampler())
            
            
            start = time.time()
            sampleset = sampler.sample(bqm, num_reads=100, label="Community Detection via BQM")
            run_time = (time.time() - start) * 1000  # ms
            
            best_sample = sampleset.first.sample
            energy = sampleset.first.energy"""
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(bqm,
                                    label='Hybrid-BQM-Community')
            best_sample = sampleset.first.sample
            client.close()


            assignment = {}
            for var, value in best_sample.items():
                if value == 1:
                    node, part = var.split("__")
                    assignment[int(node)] = int(part)


            communities = []
            for c in partitions:
                comm = {node for node, cluster in assignment.items() if cluster == c}
                if comm:
                    communities.append(comm)

            #Postprocessing
            for node in self.G.nodes():
                if node not in assignment:

                    max_community = None
                    max_edges = 0
                    for c, comm in enumerate(communities):
                        edges = sum(1 for neighbor in self.G.neighbors(node) if neighbor in comm)
                        if edges > max_edges:
                            max_edges = edges
                    assignment[node] = max_community
                    communities[max_community].add(node)


            modularity = nx.community.modularity(self.G, communities)

            with open(self.result_path, "a", encoding="utf-8") as f:
                #f.write(f"Sampling time: {run_time:.2f} ms \n")
                f.write(f'Generation: {self.generation}, Individual: {self.ind_idx} \n')
                f.write(f'Best sample: {best_sample} \n')
                f.write(f'Communities: {communities} \n')
                f.write(f'Modularity: {modularity} \n\n')

            if modularity > 0:
                fitness = 1 / modularity
            else:
                fitness = 100

        except Exception as e:
            with open(self.result_path, "a", encoding="utf-8") as f:
                import traceback
                f.write("An error occurred\n")
                traceback.print_exc(file=f)
            fitness = 100

        return (fitness,)

    


    
    
    




