"""
This file implements the functionality to test an agent (type + name) passed to it.
Usage:
python test.py PPO MlpPolicy-0
"""
from stable_baselines3 import PPO, DQN
import sys
import time
import os
from environment_factory import get_environment
import json


if __name__ == "__main__":
    if len(sys.argv) > 2:
        agent_type = sys.argv[1]
        agent_name = sys.argv[2]
    else:
        raise ValueError("Wrong arguments. Have to be of format type name")

    if agent_type == "PPO":
        model = PPO.load(os.path.join("agents", agent_name[:agent_name.rfind('-')], agent_name))
    elif agent_type == "DQN":
        model = DQN.load(os.path.join("agents", agent_name[:agent_name.rfind('-')], agent_name))
    else:
        raise ValueError("Unknown agent type {}.".format(agent_type))
    config_name = os.path.join("agents", agent_name[:agent_name.rfind('-')], "config")
    with open(config_name + ".json") as json_file:
        config = json.load(json_file)
    env = get_environment(config)

    obs = env.reset()
    env.render()
    input("Press Key to continue")
    total_reward = 0
    while True:
        action = model.predict(obs, deterministic=True)
        print(action)
        # Processing:
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Rendering the game:
        env.render()
        time.sleep(1 / 30)  # FPS

        # Checking if the player is still alive
        if done:
            print("total reward:", total_reward)
            print(info)
            break

    env.close()