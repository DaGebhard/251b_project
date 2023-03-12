"""
This file implements the get_model function used in main.py
"""
from stable_baselines3 import PPO


def get_model(env, config):
    """
    :param env: a gym environment
    :param config: a dictionary holding the configuration specified in the json-file
    :return model: a gym model for the given environment, following the specifications given in config
    """
    type = config['model']['type']
    if type == "PPO":
        policy = config['model']['policy']
        model = PPO(policy, env)
    return model
