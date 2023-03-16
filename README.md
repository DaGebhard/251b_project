# 251b_project
Environment: https://github.com/Talendar/flappy-bird-gym

Stable baselines documentation: https://stable-baselines3.readthedocs.io/en/master/index.html

To-Dos:
- create agent based on ResNet
- get it to run on datahub (sound problems)

Requirements:

    pip install stable-baselines3[extra] 
    pip install flappy-bird-gym

Changes to be made on the flappy-bird-gym (description for datahub, apply the same changes on local machine):

1: turn audio off (prevents error on datahub)

    cd ~/.local/lib/python3.9/site-packages/flappy_bird_gym/envs
    vi renderer.py


Press ``i``, overwrite ``audio_on=True`` with ``audio_on=False`` in the init function. Press ``esc``, type ``:w``, and
finally type ``:q`` to save your changes.

2: update observation space of rgb environment (this allows the usage of CNN policy)

    cd ~/.local/lib/python3.9/site-packages/flappy_bird_gym/envs
    vi flappy_bird_env_rgb.py

Press ``i``. In the init function, replace ``self.observation_space = gym.spaces.Box(0, 255, [*screen_size, 3])`` with
``self.observation_space = gym.spaces.Box(0, 255, [*screen_size, 3], dtype=np.uint8)`` by adding ``, dtype=np.uint8``.
Press ``esc``, type ``:w``, and finally type ``:q`` to save your changes.
