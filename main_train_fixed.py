# Import RLlib algorithms
# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse2 import MiniGreenhouse2

# Import support libraries
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Use PPOConfig to configure and build the PPO algorithm
config = (
    PPOConfig()
    .environment(
        env=MiniGreenhouse2,
        env_config={
            "flag_run": False,
            "first_day": 1,
            "season_length": 1/72,
            "max_steps": 4
        })
    #.env_runners(num_envs_per_env_runner=2)
    .training(train_batch_size=10, sgd_minibatch_size=2)
)

# Build.
algo = config.build()

# Train the model for a number of iterations
# 1 iteration mean 1 hour because 1 iteration has 4 time-steps that equal to 1 hour in real-time
iterations = 480

for i in tqdm(range(iterations)):
    results = algo.train()
    
    reward = results['env_runners']['episode_return_mean']
    
    # Print the rewards
    print(f"Iter: {i}; avg. rewards={reward}")

# Save the model checkpoint
save_result = algo.save('model/model-minigreenhouse-6')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)

# Remove unnecessary variables
os.remove('controls.mat') # controls file
os.remove('drl-env.mat')  # simulation file
os.remove('indoor.mat')   # indoor measurements
os.remove('fruit.mat')    # fruit growth
#os.remove('outdoor.mat')  # outdoor measurements
