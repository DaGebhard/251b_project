"""
This file implements the get_environment function called by main.py
"""
import flappy_bird_gym


def get_environment(config):
    """
    :param config: a dictionary holding the configuration specified in the json-file
    :return env: a gym environment, following the specifications given in config
    """
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")
    return env
