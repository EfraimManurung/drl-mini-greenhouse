from ray.rllib.algorithms.algorithm import Algorithm

# Import supporting libraries
import time

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse2 import MiniGreenhouse2

# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:

my_new_ppo = Algorithm.from_checkpoint('model/model-minigreenhouse')

env = MiniGreenhouse2({}, 6, True, 2)

# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

# Play one episode
while not terminated and not truncated:    
    # Make some delay, so it is easier to see
    time.sleep(1.0)
    
    # Compute a single action, given the current observation
    # from the environment.
    action = my_new_ppo.compute_single_action(obs)
    print("ACTION: ", action)
    
    # Apply the computed action in the environment.
    obs, reward, terminated, _, info = env.step(action)

    # sum up rewards for reporting purposes
    total_reward += reward

# Report results.
print(f"Played 1 episode; total-reward={total_reward}")
