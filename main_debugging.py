from envs.MiniGreenhouse2 import MiniGreenhouse2

# Configure.
from ray.rllib.algorithms.ppo import PPOConfig
config = PPOConfig().environment(env=MiniGreenhouse2).training(train_batch_size=10, sgd_minibatch_size=5)

# Build.
algo = config.build()

# Train.
print(algo.train())
