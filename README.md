# 251b_project
Environment: https://github.com/Talendar/flappy-bird-gym

Stable baselines documentation: https://stable-baselines3.readthedocs.io/en/master/index.html

We want a json file for the configuration (number of training steps, how frequently it saves, model, policy etc.), a model factory given the environemnt creating a model based on the configuration, a main function executing the training and saving the agent from time to time, and a test function which given a saved agent executes the same.