# Configure.
from ray.rllib.algorithms.ppo import PPOConfig
from envs.MiniGreenhouse2 import MiniGreenhouse2

config = PPOConfig().environment(env=MiniGreenhouse2).training(train_batch_size=100, sgd_minibatch_size=10)


# Build.
algo = config.build()

# Train.
print(algo.train())