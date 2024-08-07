# Import RLlib algorithms
# Configure.
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse import MiniGreenhouse

# Import support libraries
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Configure the RLlib PPO algorithm
config = PPOConfig()
config.rollouts(num_rollout_workers=1)
config.resources(num_cpus_per_worker=1)
config.environment(
    env=MiniGreenhouse,
        env_config={
            "flag_run": False,
            "first_day": 1,
            "season_length": 1/72,
            "max_steps": 3
        })

config.training(
    gamma=0.9,  # Discount factor
        lr=0.0001, #lr = 0.1,  # Learning rate
        kl_coeff=0.3,  # KL divergence coefficient
        # model={
        #     "fcnet_hiddens": [256, 256, 256],  # Hidden layer configuration
        #     "fcnet_activation": "relu",  # Activation function
        #     "use_lstm": True,  # Whether to use LSTM
        #     "max_seq_len": 48,  # Maximum sequence length
        # }, 
        train_batch_size=2, 
        sgd_minibatch_size=1
)

# Build the algorithm object
algo = config.build()

# Train the algorithm
for episode in tqdm(range(300)):  # Train for 250 episodes
    result = algo.train()  # Perform training
    # if episode % 5 == 0:  # Save a checkpoint every 5 episodes
    #     checkpoint_dir = algo.save().checkpoint.path
    #     print(f"Checkpoint saved in directory {checkpoint_dir}")
        
# Save the model checkpoint
save_result = algo.save('model/model-minigreenhouse-config-3')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
    
# Remove unnecessary variables
os.remove('controls.mat') # controls file
os.remove('drl-env.mat')  # simulation file
os.remove('indoor.mat')   # indoor measurements
os.remove('fruit.mat')    # fruit growth
#os.remove('outdoor.mat')  # outdoor measurements
