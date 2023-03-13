"""
This file implements the functionality to test an agent (type + name) passed to it.
Usage:
python test.py PPO MlpPolicy-0
"""
from stable_baselines3 import PPO
import sys
import time


if __name__ == "__main__":
    if len(sys.argv) > 2:
        agent_type = sys.argv[1]
        agent_name = sys.argv[2]
    else:
        raise ValueError("Wrong arguments. Have to be of format type name")

    if agent_type == "PPO":
        model = PPO.load("agents/" + agent_name)
    else:
        raise ValueError("Unknown agent type.")
    env = model.get_env()

    obs = env.reset()
    input("Press Key to continue")
    while True:
        action = model.predict(obs, deterministic=True)

        # Processing:
        obs, reward, done, info = env.step(action)

        # Rendering the game:
        env.render()
        time.sleep(1 / 30)  # FPS

        # Checking if the player is still alive
        if done:
            break

    env.close()