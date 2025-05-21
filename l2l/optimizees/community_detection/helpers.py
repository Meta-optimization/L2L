import os
import configparser

def create_config(API_token, path):
    if not os.path.exists(path):
        os.makedirs(path)
    config = configparser.ConfigParser()
    config["defaults"] = {'token': API_token}
    with open(path + "/dwave.conf", "w") as configfile:
        config.write(configfile)

