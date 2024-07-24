'''
Deep Reinforcement Learning for mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

Main program
'''


# Import libraries
import gymnasium as gym

# Import algorithm
from ray.rllib.algorithms import DQNConfig, DQN

# Import tuner
from ray import air
from ray import tune

# Import environment
from envs.MiniGreenhouse import MiniGreenhouse

'''
TO-DO:

https://docs.ray.io/en/latest/tune/key-concepts.html

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        metric="score",
        mode="min",
        search_alg=BayesOptSearch(random_search_steps=4),
    ),
    run_config=train.RunConfig(
        stop={"training_iteration": 20},
    ),
    param_space=config,
)
results = tuner.fit()

best_result = results.get_best_result()  # Get best result object
best_config = best_result.config  # Get best trial's hyperparameters
best_logdir = best_result.path  # Get best trial's result directory
best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
best_metrics = best_result.metrics  # Get best trial's last results
best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

'''

# Setup
config = DQNConfig()

config = config.training(
    num_atoms=tune.grid_search([1,]))
config = config.environment(env=MiniGreenhouse)

# Loop
tuner = tune.Tuner(
    "DQN",
    run_config=air.RunConfig(stop={"training_iteration":1}),
    param_space=config.to_dict()
)

# Save model
results = tuner.fit()
