from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from envs.MiniGreenhouse2 import MiniGreenhouse2

YourExternalEnv = MiniGreenhouse2
register_env("my_env", lambda config: YourExternalEnv(config))

# Use PPOConfig to configure and build the PPO algorithm
config = PPOConfig().environment("my_env")
algo = config.build()

while True:
    print(algo.train())
