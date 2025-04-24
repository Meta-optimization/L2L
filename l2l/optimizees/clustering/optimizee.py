import os
import numpy as np
import math
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .helpers import create_config, get_distance, get_max_distance
from dwave.cloud.config import load_config
from dwave.cloud.client import Client
import dwavebinarycsp
from dwave.system import EmbeddingComposite, DWaveSampler

ClusteringOptimizeeParameters = namedtuple(
    'ClusteringOptimizeeParameters', ['APIToken', 'path', 'num_reads', 'coordinates'])


class ClusteringOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.num_reads = parameters.num_reads
        self.coordinates = parameters.coordinates
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        self.bound = [0, 2000]
        self.config_path = os.path.join(parameters.path, "dwave.conf")
        if not os.path.exists(self.config_path):
            create_config(parameters.APIToken, parameters.path)

    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        individual = {'num_reads': self.num_reads}
        return individual
    

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return {'coords': np.clip(individual['num_reads'], a_min=self.bound[0], a_max=self.bound[1])}

    def simulate(self, traj):
        """
        Does Clustering
        """
        config = load_config(self.config_path)
        print(config)

        max_distance = max(get_max_distance(self.coordinates), 1)

        # Build constraints
        csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

        # Apply constraint: coordinate can only be in one colour group
        choose_one_group = {(0, 0, 1), (0, 1, 0), (1, 0, 0)}
        for coord in self.coordinates:
            csp.add_constraint(choose_one_group, (coord.r, coord.g, coord.b))

        # Build initial BQM
        bqm = dwavebinarycsp.stitch(csp)

        # Edit BQM to bias for close together points to share the same color
        for i, coord0 in enumerate(self.coordinates[:-1]):
            for coord1 in self.coordinates[i+1:]:
                # Set up weight
                d = get_distance(coord0, coord1) / max_distance  # rescale distance
                weight = -math.cos(d*math.pi)

                # Apply weights to BQM
                bqm.add_interaction(coord0.r, coord1.r, weight)
                bqm.add_interaction(coord0.g, coord1.g, weight)
                bqm.add_interaction(coord0.b, coord1.b, weight)

        # Edit BQM to bias for far away points to have different colors
        for i, coord0 in enumerate(self.coordinates[:-1]):
            for coord1 in self.coordinates[i+1:]:
                # Set up weight
                # Note: rescaled and applied square root so that far off distances
                #   are all weighted approximately the same
                d = math.sqrt(get_distance(coord0, coord1) / max_distance)
                weight = -math.tanh(d) * 0.1

                # Apply weights to BQM
                bqm.add_interaction(coord0.r, coord1.b, weight)
                bqm.add_interaction(coord0.r, coord1.g, weight)
                bqm.add_interaction(coord0.b, coord1.r, weight)
                bqm.add_interaction(coord0.b, coord1.g, weight)
                bqm.add_interaction(coord0.g, coord1.r, weight)
                bqm.add_interaction(coord0.g, coord1.b, weight)

        try:
            client = Client.from_config(config_file=self.config_path)
            # code that uses client
            solvers = client.get_solvers()
            solver = solvers[5]
            print(solver)
            sampler = EmbeddingComposite(DWaveSampler(solver=solver.id))
            sampleset = sampler.sample(bqm,
                                    chain_strength=4,
                                    num_reads=1000,
                                    label='Example - Clustering')

            best_sample = sampleset.first.sample
            client.close()
        except:
            print("error")

        fitness = len(solvers)
        return (fitness,) 
    


    
    
    




