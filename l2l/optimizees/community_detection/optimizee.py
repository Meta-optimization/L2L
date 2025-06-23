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
from collections import defaultdict


CommunityOptimizeeParameters = namedtuple(
    'CommunityOptimizeeParameters', ['APIToken', 'config_path', 'num_partitions', 'one_hot_strength', 'num_reads', 'Graph','result_path'])


class CommunityOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.num_partitions = parameters.num_partitions
        self.one_hot_strength = parameters.one_hot_strength
        self.num_reads = parameters.num_reads
        self.G = parameters.Graph
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.config_path = os.path.join(parameters.config_path, "dwave.conf")
        os.makedirs(parameters.result_path, exist_ok=True)
        self.result_path = os.path.join(parameters.result_path, "result.txt")
        if not os.path.exists(self.config_path):
            create_config(parameters.APIToken, parameters.config_path)

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {'num_partitions': self.num_partitions, "one_hot_strength": self.one_hot_strength, 
                      'num_reads': self.num_reads}
        return individual
    

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'num_reads' :np.clip(individual['num_reads'], a_min=1, a_max=1000),
                'one_hot_strength': np.clip(individual['one_hot_strength'], a_min=1, a_max=50*len(self.G.nodes)),
                #num partitions is not possible to calculate like this 
                'num_partitions': np.clip(individual['num_partitions'], a_min=2, a_max=6)} #a_max = len(self.G.nodes)

    def simulate(self, traj):
        # Extract metadata from trajectory object
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        # Define partitions (clusters) and initialize variables
        partitions = range(int(traj.individual.num_partitions))
        
        # Compute the modularity matrix of the graph
        B = nx.modularity_matrix(self.G)
        
        # Initialize a binary quadratic model (BQM)
        bqm = BinaryQuadraticModel('BINARY')
        
        # One-hot encoding constraint strength
        one_hot_strength = traj.individual.one_hot_strength

        # Add variables for each node in each partition (one-hot encoding)
        for i in self.G.nodes():
            for c in partitions:
                var = f"{i}__{c}"
                bqm.add_variable(var, 0.0)

        # Add modularity-based interactions (encodes community structure objective)
        for i in self.G.nodes():
            for j in self.G.nodes():
                if i == j:
                    continue
                for c in partitions:
                    vi = f"{i}__{c}"
                    vj = f"{j}__{c}"
                    # Negative interaction to maximize modularity
                    bqm.add_interaction(vi, vj, -B[i, j])

        # Apply one-hot constraints: each node must belong to exactly one partition
        for i in self.G.nodes():
            vars_i = [f"{i}__{c}" for c in partitions]
            
            # Add a small positive bias to all variables
            for v in vars_i:
                bqm.add_variable(v, bqm.get_linear(v) + one_hot_strength)
            
            # Penalize assigning a node to multiple clusters
            for u, v in combinations(vars_i, 2):
                bqm.add_interaction(u, v, 2 * one_hot_strength)
            
            # Subtract linear term to center the penalty
            for v in vars_i:
                bqm.add_variable(v, bqm.get_linear(v) - 2 * one_hot_strength)
            
            # Add constant offset for the one-hot penalty
            bqm.offset += one_hot_strength

        try:
            # Load D-Wave quantum sampler client
            client = Client.from_config(config_file=self.config_path)

            # Use embedding composite sampler for running on quantum hardware
            start = time.time()
            sampler = EmbeddingComposite(DWaveSampler())
            embedding_time = (time.time() - start)

            # Submit the BQM for sampling and time the process
            start = time.time()
            sampleset = sampler.sample(bqm, num_reads=int(traj.individual.num_reads),label="Community Detection via BQM")
            wall_time = (time.time() - start) 
            
            best_sample = sampleset.first.sample
            """sampler = LeapHybridSampler()
            sampleset = sampler.sample(bqm,
                                    label='Hybrid-BQM-Community')
            best_sample = sampleset.first.sample"""
            client.close()

            #Robust Decoding
            
            raw_assignments = defaultdict(list)

            # Collect all 1-valued variables per node
            for var, value in best_sample.items():
                if value == 1:
                    node_str, part_str = var.split("__")
                    node = int(node_str)
                    part = int(part_str)
                    raw_assignments[node].append(part)

            #  Clean and fix assignments
            assignment = {}
            for node in self.G.nodes():
                assigned_parts = raw_assignments.get(node, [])

                if len(assigned_parts) == 1:
                    assignment[node] = assigned_parts[0]
                else:
                    # Invalid case: node unassigned or over-assigned
                    max_community = None
                    max_edges = -1
                    for c in partitions:
                        edges = sum(1 for neighbor in self.G.neighbors(node)
                                    if assignment.get(neighbor) == c)
                        if edges > max_edges:
                            max_edges = edges
                            max_community = c
                    if max_community is None:
                        max_community = 0  # fallback
                    assignment[node] = max_community

            #Build community sets
            communities = []
            for c in partitions:
                comm = {node for node, cluster in assignment.items() if cluster == c}
                if comm:
                    communities.append(comm)

            modularity = nx.community.modularity(self.G, communities)

            with open(self.result_path, "a", encoding="utf-8") as f:
                f.write(f"Embedding time: {embedding_time} s \n")
                f.write(f"Sampling time: {wall_time} s \n")
                f.write(f"QPU access time: {sampleset.info['timing']['qpu_access_time'] / 1000} ms \n")
                f.write(f'Generation: {self.generation}, Individual: {self.ind_idx} \n')
                f.write(f'Best sample: {best_sample} \n')
                f.write(f'Communities: {communities} \n')
                f.write(f'Modularity: {modularity} \n\n')

            # Compute fitness score: better modularity means lower fitness
            if modularity > 0:
                fitness = 1 / modularity
            else:
                fitness = 100  # Penalize very poor solutions

        except Exception as e:
            with open(self.result_path, "a", encoding="utf-8") as f:
                import traceback
                f.write("An error occurred\n")
                traceback.print_exc(file=f)
            fitness = 100

        return (fitness,)

    


    
    
    




