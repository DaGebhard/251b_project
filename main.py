"""
This file implements the training functionality. It gets passed the name of a configuration as an argument.
Usage:
1) train new agent based on config: python main.py --config default
2) load most recent version of an agent: python main.py --load PPO-MlpPolicy
3) load specific version of an agent: python main.py --load PPO-MlpPolicy-0 (this overrides previous training and
might lead to intended behavior if less than already available epochs are retrained)
"""
import os
import json
from model_factory import get_model
from environment_factory import get_environment
import argparse
import shutil
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSE151B Project')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='configuration to train a new agent on')
    group.add_argument('--load', type=str, help='name of the model to load')
    args = parser.parse_args()

    if args.config is not None:
        config_name = args.config
        agent_name = None
    elif args.load is not None:
        agent_name = args.load
        if agent_name[agent_name.rfind('-') + 1:].isdigit():
            config_name = os.path.join("agents", agent_name[:agent_name.rfind('-')], "config")
        else:
            config_name = os.path.join("agents", agent_name, "config")
    else:
        raise ValueError("Either --config or --load argument has to be passed")

    with open(config_name + ".json") as json_file:
        config = json.load(json_file)
    if args.config is not None:
        os.mkdir(os.path.join("agents", config['model']['name']))
        shutil.copy2(config_name + ".json", os.path.join("agents", config['model']['name'], "config.json"))

    log_dir = os.path.join("agents", config['model']['name'])
    env = get_environment(config)
    env = Monitor(env, log_dir)
    model, number_of_previous_epochs = get_model(env, config, agent_name)
    if args.load is not None:
        model.set_env(env)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=config['training']['timesteps_per_epoch'], callback=callback)
    plot_results([log_dir], config['training']['timesteps_per_epoch'], results_plotter.X_TIMESTEPS, env)
    plt.savefig(os.path.join(log_dir, "stats.png"))
    plt.show()
    # for epoch in range(config['training']['epochs']):
    #     model.learn(config['training']['timesteps_per_epoch'],  progress_bar=True, reset_num_timesteps=False)
    #     model.save(os.path.join("agents",config['model']['name'], config['model']['name'] + "-{}".format(
    #         epoch + number_of_previous_epochs)))
