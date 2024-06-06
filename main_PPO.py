from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from envs.MiniGreenhouse2 import MiniGreenhouse2

# Register your custom environment
register_env("my_env", lambda config: MiniGreenhouse2(config))

# Use PPOConfig to configure and build the PPO algorithm
config = (
    PPOConfig()
    .environment("my_env")
    .rollouts(num_envs_per_worker=1)
)

# Construct the PPO algorithm object from the config
algo = config.build()

# Train the model for a number of iterations
for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. rewards={results['episode_reward_mean']}")

# Save the model checkpoint
save_result = algo.save('model/model-minigreenhouse')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
