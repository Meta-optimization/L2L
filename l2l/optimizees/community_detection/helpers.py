import os
import configparser

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

