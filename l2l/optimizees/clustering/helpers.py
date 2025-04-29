import os
import configparser
import numpy as np
from collections import defaultdict

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