# Import RLlib algorithms
# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse2 import MiniGreenhouse2

# Use PPOConfig to configure and build the PPO algorithm
config = (
    PPOConfig()
    .environment(env=MiniGreenhouse2,)
    # .env_runners(num_envs_per_env_runner=2)
    .training(train_batch_size=2, sgd_minibatch_size=1)
)

# Build.
algo = config.build()

# Train the model for a number of iterations
for i in range(10):
    results = algo.train()
    
    # Print the rewards
    print(f"Iter: {i}; avg. rewards={results['env_runners']['episode_return_mean']}")
    
# Save the model checkpoint
save_result = algo.save('model/model-minigreenhouse')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
