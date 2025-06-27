import os
import configparser
import csv

def create_config(API_token, path):
    """
    Creates a configuration file for DWave API connection.

    Args:
    - API_token (str): The token for authentication with DWave API.
    - path (str): The directory path where the configuration file will be saved.

    Returns:
    - None
    """

    # Check if the provided path exists, if not create the directory
    if not os.path.exists(path):
        os.makedirs(path)

    #Set token at the configuration
    config = configparser.ConfigParser()
    config["defaults"] = {'token': API_token}

    # Open a new file in write mode to save the configuration
    with open(path + "/dwave.conf", "w") as configfile:
        config.write(configfile)

def result_csv(path,embedding_time,wall_time, qpu_time, generation, ind_idx, best_sample, communities, modularity):
    # Define header once (if file doesn't exist yet)
    write_header = not os.path.exists(path)

    with open(path, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "Embedding time (s)",
                "Solving time (s)",
                "QPU access time (ms)",
                "Generation",
                "Individual",
                "Best sample",
                "Communities",
                "Modularity"
            ])

        writer.writerow([
            embedding_time,
            wall_time,
            qpu_time,
            generation,
            ind_idx,
            best_sample,
            communities,
            modularity
        ])


def result_csv_hybrid(path,wall_time, generation, ind_idx, best_sample, communities, modularity):
    # Define header once (if file doesn't exist yet)
    write_header = not os.path.exists(path)

    with open(path, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "Solving time (s)",
                "Generation",
                "Individual",
                "Best sample",
                "Communities",
                "Modularity"
            ])

        writer.writerow([
            wall_time,
            generation,
            ind_idx,
            best_sample,
            communities,
            modularity
        ])

