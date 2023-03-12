import time
import flappy_bird_gym
from stable_baselines3 import PPO


env = flappy_bird_gym.make("FlappyBird-rgb-v0")
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1_000)

# display current capability
obs = env.reset()
input("Press Key to continue")
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = model.predict(obs)

    # Processing:
    obs, reward, done, info = env.step(action)

    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS

    # Checking if the player is still alive
    if done:
        break

env.close()