# Import RLlib algorithms
# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse2 import MiniGreenhouse2

from tqdm import tqdm

# Configure the RLlib PPO algorithm
config = PPOConfig()
config.rollouts(num_rollout_workers=1)
config.resources(num_cpus_per_worker=1)
config.environment(
        env=MiniGreenhouse2,
        env_config={
            "flag_run": False,
            "first_day": 1,
            "season_length": 1/72,
            "max_steps": 4
        },
        render_env=False  # Whether to render the environment
    )

config.training(
        gamma=0.9,  # Discount factor
        lr=0.0001,  # Learning rate
        kl_coeff=0.3,  # KL divergence coefficient
        model={
            "fcnet_hiddens": [256, 256],  # Hidden layer configuration
            "fcnet_activation": "relu",  # Activation function
            "use_lstm": True,  # Whether to use LSTM
            "max_seq_len": 48,  # Maximum sequence length
        }
    )

# Build the algorithm object
algo = config.build()

# Train the algorithm
for episode in tqdm(range(10)):  # Train for 250 episodes
    result = algo.train()  # Perform training
    if episode % 5 == 0:  # Save a checkpoint every 5 episodes
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")