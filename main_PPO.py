# Import libraries
import sys
import os
import time

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Import environment
from envs.MiniGreenhouse import MiniGreenhouse

# RL Configuration and Training
config = (
    PPOConfig().environment(
        env=MiniGreenhouse,
        # Config dict to be passed to our custom env's constructor.
        env_config={},
    )
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=1)
)

# Construct the PPO algorithm object from the config
algo = config.build()

# Train the model for a number of iterations
for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. rewards={results['env_runners']['episode_return_mean']}")

# call `save()` to create a checkpoint.
save_result = algo.save('model/drl-model-mini-greenhouse')

path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)