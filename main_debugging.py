# Import RLlib algorithms
# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse2 import MiniGreenhouse2

# Use PPOConfig to configure and build the PPO algorithm
config = (
    PPOConfig()
    .environment(env=MiniGreenhouse2)
    .rollouts(num_envs_per_worker=2)
    .training(train_batch_size=4, sgd_minibatch_size=1)
)

# Build.
algo = config.build()

# Train.
# print(algo.train())
# Train the model for a number of iterations
for i in range(2):
    results = algo.train()
    # print(f"Iter: {i}; avg. rewards={results['episode_reward_mean']}")
    # print(f"Iter: {i}; results={results}")
    
    # Access 'episode_reward_mean' in 'env_runners'
    if 'env_runners' in results and 'episode_reward_mean' in results['env_runners']:
        print(f"Iter: {i}; avg. rewards={results['env_runners']['episode_reward_mean']}")
    else:
        print(f"Iter: {i}; 'episode_reward_mean' not found in results")

# Save the model checkpoint
save_result = algo.save('model/model-minigreenhouse')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
