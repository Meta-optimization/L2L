import os
import configparser
import numpy as np
from collections import defaultdict
import random

def create_config(API_token, path):
    if not os.path.exists(path):
        os.makedirs(path)
    config = configparser.ConfigParser()
    config["defaults"] = {'token': API_token}
    with open(path + "/dwave.conf", "w") as configfile:
        config.write(configfile)

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_labels_from_sample(sample, num_points, num_clusters):
    y = [-1] * num_points
    for i in range(num_points):
        for c in range(num_clusters):
            var_name = f"x_{i}_{c}"
            if sample.get(var_name, 0) == 1:
                y[i] = c
                break
    return y

def is_valid_one_hot(sample, num_points, num_clusters, variables):
    for i in range(num_points):
        if sum(sample[variables[(i, c)]] for c in range(num_clusters)) != 1:
            return False
    return True

def fix_sample_one_hot(sample, num_points, num_clusters, variables):
    """
    Fixes a sample to ensure that each point is assigned to exactly one cluster
    (i.e., satisfies one-hot encoding constraints).

    :param sample: dict, e.g., {"x_0_0": 1, "x_0_1": 0, ...}
    :param num_points: total number of data points
    :param num_clusters: total number of clusters
    :param variables: dict mapping (i, c) → "x_i_c"
    :return: corrected sample (dict)
    """
    corrected_sample = sample.copy()

    for i in range(num_points):
        # Get all variables corresponding to point i
        cluster_vars = [variables[(i, c)] for c in range(num_clusters)]

        # Check which cluster variables are active (set to 1)
        active = [var for var in cluster_vars if corrected_sample.get(var, 0) == 1]

        if len(active) == 1:
            # Already valid one-hot — nothing to change
            continue
        elif len(active) > 1:
            # Multiple active — choose one randomly
            chosen = random.choice(active)
        else:
            # No active cluster — pick the one with the highest value (or just default to 0s)
            values = [(var, corrected_sample.get(var, 0)) for var in cluster_vars]
            chosen = max(values, key=lambda x: x[1])[0]  # default: highest "confidence" or fallback to 0

        # Set all cluster variables to 0, except for the chosen one
        for var in cluster_vars:
            corrected_sample[var] = 1 if var == chosen else 0

    return corrected_sample