from envs.MiniGreenhouse2 import MiniGreenhouse2

env_config = {}
env = MiniGreenhouse2(env_config)

observation, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Done: {done}")
