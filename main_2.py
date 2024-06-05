from envs.MiniGreenhouse import MiniGreenhouse

env_config = {}

# def __init__(self, env_config, time_multiplier):
env = MiniGreenhouse(env_config, 6.0, 4)

observation, _ = env.reset()
done = False

# Manually set the action values
# fixed_action = [0.0, 1.0, 1.0]  # Action values for fan, toplighting, and heating

#env.done(10)

while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Done: {done}")