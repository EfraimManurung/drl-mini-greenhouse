# Import RLlib algorithms
# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse2 import MiniGreenhouse2

# Import support libraries
import math
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
            "first_day": 6,
            "season_length": 1/72,
            "max_steps": 4
        })
    # .env_runners(num_envs_per_env_runner=2)
    .training(train_batch_size=2, sgd_minibatch_size=1)
)

# Build.
algo = config.build()

# Initialize lists to store reward values
avg_rewards_list = []

# Give initial values for reward
sum_rewards = 0

# Train the model for a number of iterations
iterations = 8

for i in tqdm(range(iterations)):
    results = algo.train()
    
    reward = results['env_runners']['episode_return_mean']
    
    # Print the rewards
    print(f"Iter: {i}; avg. rewards={reward}")
    
    if math.isnan(reward):
        reward = 0
        sum_rewards += reward
        avg_rewards_list.append(sum_rewards)
    else:
        sum_rewards += reward
        avg_rewards_list.append(sum_rewards)
    
print(f"Average rewards for {iterations} iterations : ", avg_rewards_list)
    
# Save the model checkpoint
save_result = algo.save('model/model-minigreenhouse-3')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)

# Plotting the results
plt.plot(range(iterations), avg_rewards_list, marker='o')
plt.xlabel('Iterations [-]')
plt.ylabel('Sum of Rewards [-]')
plt.title('Sum of Rewards over Iterations')
plt.grid(True)
plt.show()

# Remove unnecessary variables
os.remove('controls.mat') # controls file
os.remove('drl-env.mat')  # simulation file
os.remove('indoor.mat')   # indoor measurements
os.remove('fruit.mat')    # fruit growth
#os.remove('outdoor.mat')  # outdoor measurements
