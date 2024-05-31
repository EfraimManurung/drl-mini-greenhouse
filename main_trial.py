from envs.MiniGreenhouse import MiniGreenhouse

env_config = {}
env = MiniGreenhouse(env_config)

# Test the environment
observation, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Done: {done}")