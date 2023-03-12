"""
This file implements the training functionality. It gets passed the name of a configuration as an argument.
"""
import sys
import json
from model_factory import get_model
from environment_factory import get_environment


if __name__ == "__main__":
    config_name = 'default'

    if len(sys.argv) > 1:
        config_name = sys.argv[1]

    with open(config_name + ".json") as json_file:
        config = json.load(json_file)

    env = get_environment(config)
    model = get_model(env, config)
    for epoch in range(config['training']['epochs']):
        model.learn(config['training']['timesteps_per_epoch'])
        model.save("agents/" + config['model']['name'] + "-{}".format(epoch))
