import os
import numpy as np
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .helpers import create_config, result_csv
from dwave.cloud.client import Client
import networkx as nx
from dimod import BinaryQuadraticModel
from itertools import combinations
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import time
from collections import defaultdict


CommunityOptimizeeParameters = namedtuple(
    'CommunityOptimizeeParameters', ['APIToken', 'config_path', 'seed', 'Graph', 'weight','result_path'])


class CommunityOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        seed = np.uint32(parameters.seed)
        self.random_state = np.random.RandomState(seed=seed)

        self.G = parameters.Graph
        self.weight = parameters.weight

        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        self.config_path = os.path.join(parameters.config_path, "dwave.conf")
        os.makedirs(parameters.result_path, exist_ok=True)
        self.result_path = os.path.join(parameters.result_path, "result.csv")
        if not os.path.exists(self.config_path):
            create_config(parameters.APIToken, parameters.config_path)

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {'num_partitions': self.random_state.uniform(2,6), 
                      "one_hot_strength": self.random_state.uniform(0.01,10), 
                      'num_reads': self.random_state.uniform(50,1000), 
                      'annealing_time': self.random_state.uniform(10,100)}
        return individual
    

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'num_partitions': np.clip(individual['num_partitions'], a_min=2, a_max=6),
                'one_hot_strength': np.clip(individual['one_hot_strength'], a_min=0.01, a_max=10),
                'num_reads' :np.clip(individual['num_reads'], a_min=50, a_max=1000),
                'annealing_time': np.clip(individual['annealing_time'], a_min=10, a_max=100)} 

    def simulate(self, traj):
        # Extract metadata from trajectory object
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        # Define partitions (clusters) and initialize variables
        partitions = range(int(np.round(traj.individual.num_partitions)))
        
        # Compute the modularity matrix of the graph
        B = nx.modularity_matrix(self.G, weight=self.weight)
        
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
            sampleset = sampler.sample(bqm, num_reads=int(np.round(traj.individual.num_reads)),
                                       annealing_time = int(np.round(traj.individual.annealing_time)),
                                       label="Community Detection via BQM")
            wall_time = (time.time() - start) 
            
            best_sample = sampleset.first.sample
            client.close()

            # Iterate over samples in the sampleset
            valid_sample = None
            for sample in sampleset.samples():
                is_valid = True
                for node in self.G.nodes():
                    # Count how many partitions this node is assigned to (i.e., how many variables for this node are 1)
                    assigned = sum(sample[f"{node}__{c}"] for c in partitions)
                    if assigned != 1:
                        is_valid = False
                        break  # No need to continue if one node already violates the constraint
                if is_valid:
                    valid_sample = sample
                    break
            
            sample = valid_sample if valid_sample is not None else best_sample
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
                        max_community = np.random.randint(0, partitions)  # fallback
                    assignment[node] =  max_community

            #Build community sets
            communities = []
            for c in partitions:
                comm = {node for node, cluster in assignment.items() if cluster == c}
                if comm:
                    communities.append(comm)

            modularity = nx.community.modularity(self.G, communities)

            result_csv(path=self.result_path, embedding_time=embedding_time, wall_time=wall_time, 
                       qpu_time=sampleset.info['timing']['qpu_access_time']/1000, generation=self.generation, 
                       ind_idx=self.ind_idx, best_sample=best_sample, communities=communities, modularity=modularity)

        except Exception as e:
            with open(self.result_path, "a", encoding="utf-8") as f:
                import traceback
                f.write("An error occurred\n")
                traceback.print_exc(file=f)
            modularity = 0

        return (modularity,)

    


    
    
    




