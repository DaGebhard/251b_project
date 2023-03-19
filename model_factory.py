"""
This file implements the get_model function used in main.py
"""
from stable_baselines3 import PPO
from stable_baselines3.dqn import DQN
import os
from typing import Optional, Tuple

import numpy as np
from gym import spaces

from stable_baselines3.common.noise import ActionNoise


class CustomDQN(DQN):
    def __init__(self, policy, env, learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, batch_size=32,
                 tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, replay_buffer_class=None,
                 replay_buffer_kwargs=None, optimize_memory_usage=False, target_update_interval=10000,
                 exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10,
                 tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True):
        super().__init__(policy=policy, env=env, learning_rate=learning_rate, buffer_size=buffer_size,
                         learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                         train_freq=train_freq, gradient_steps=gradient_steps, replay_buffer_class=replay_buffer_class,
                         replay_buffer_kwargs=replay_buffer_kwargs, optimize_memory_usage=optimize_memory_usage,
                         target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,
                         exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps,
                         max_grad_norm=max_grad_norm, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs,
                         verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([np.random.choice([0, 1], p=[0.92, 0.08]).astype(int) for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
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
            model = CustomDQN.load(os.path.join("agents", agent_name), env)
    else:
        number_of_previous_epochs = 0
        if type == "PPO":
            policy = config['model']['policy']
            model = PPO(policy, env, verbose=1, gamma=0.999)
        elif type == "DQN":
            policy = config['model']['policy']
            model = CustomDQN(policy, env, buffer_size=1000, verbose=1, gamma=0.999, exploration_final_eps=0.005)
    return model, number_of_previous_epochs
