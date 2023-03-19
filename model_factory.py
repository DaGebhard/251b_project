"""
This file implements the get_model function used in main.py
"""
from stable_baselines3 import PPO, DQN
import os


def get_number_of_trained_epochs(agent_name):
    """
    :param agent_name: string containing the name of the agent
    :return number_of_previous_epochs: the number of epochs the model was trained for before
    """
    number_of_previous_epochs = 0
    for file in os.listdir(os.path.join("agents", agent_name)):
        if agent_name in file:
            epoch_number = int(file[file.rfind('-') + 1:-4])
            if epoch_number > number_of_previous_epochs:
                number_of_previous_epochs = epoch_number
    return number_of_previous_epochs

def get_model(env, config, agent_name):
    """
    :param env: a gym environment
    :param config: a dictionary holding the configuration specified in the json-file
    :param agent_name: the name of an agent to be loaded
    :return model: a gym model for the given environment, following the specifications given in config
    :return number_of_previous_epochs: the number of previously trained epochs for loaded model; 0 for new model
    """
    type = config['model']['type']
    if agent_name is not None:
        if agent_name[agent_name.rfind('-') + 1:].isdigit():
            number_of_previous_epochs = int(agent_name[agent_name.rfind('-') + 1:])
            if get_number_of_trained_epochs(agent_name[:agent_name.rfind('-')]) \
                    > number_of_previous_epochs + config['training']['epochs']:
                choice = input(
                    "You are about to retrain an agent from a given number of epochs which is lower than the\n"
                    "one the model was trained on previously. This will overwrite states of the agents.\n"
                    "Additionally, the total number of epochs after training will be lower than the number\n"
                    "of total epochs the model was trained on before. This might lead to unintended behavior\n"
                    "because there will be a cutoff point in the saved agents. Please consider if you really\n"
                    "want to do this. Confirm with [y] or cancel with any other input.\n")
                if choice != 'y':
                    exit()
            elif get_number_of_trained_epochs(agent_name[:agent_name.rfind('-')]) > number_of_previous_epochs:
                choice = input(
                    "You are about to retrain an agent from a given number of epochs which is lower than the\n"
                    "one the model was trained on previously. This will overwrite states of the agents.\n"
                    "Please consider if you really want to do this. Confirm with [y] or cancel with any other input.\n")
                if choice != 'y':
                    exit()
            agent_name = os.path.join(agent_name[:agent_name.rfind('-')], agent_name)
        else:
            number_of_previous_epochs = get_number_of_trained_epochs(agent_name)
            agent_name = os.path.join(agent_name, agent_name + "-{}".format(number_of_previous_epochs))

        number_of_previous_epochs += 1
        if type == "PPO":
            model = PPO.load(os.path.join("agents", agent_name), env)
        elif type == "DQN":
            model = DQN.load(os.path.join("agents", agent_name), env)
    else:
        number_of_previous_epochs = 0
        if type == "PPO":
            policy = config['model']['policy']
            model = PPO(policy, env, verbose=1, gamma=0.999)
        elif type == "DQN":
            policy = config['model']['policy']
            model = DQN(policy, env, buffer_size=10000, verbose=1, gamma=config['model']['gamma'],
                        exploration_final_eps=config['model']['exploration final epsilon'],
                        exploration_fraction=config['model']['exploration fraction'])
    return model, number_of_previous_epochs
