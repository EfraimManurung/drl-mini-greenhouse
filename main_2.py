from envs.MiniGreenhouse2 import MiniGreenhouse2

env_config = {}

# def __init__(self, env_config, time_multiplier):
env = MiniGreenhouse2(env_config, 6.)

observation, _ = env.reset()
done = False

#env.done(10)

while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Done: {done}")
