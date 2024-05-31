'''
Deep Reinforcement Learning for mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

You can pass either a string name or a Python class to specify an environment. 
By default, strings will be interpreted as a gym environment name. 
Custom env classes passed directly to the algorithm must take asingle env_config parameters
in their constructor.

Components of an environment:
- Observation space
- Action space
- Rewards

In Python we will need to implement, at least:
- Constructor of MiniGreenhouse class
- reset()
- step()

In practice, we may also want other methods, such as render() 
to show the progress or other methods.

the MiniGreenhouse class requirements:
- convert .mat to JSON or CSV as the states (from observation)
- 

Table 1 Meaning of the state x(t), measurement y(t), control signal u(t) and disturbance d(t).
----------------------------------------------------------------------------------------------------------------------------------
 x1(t) Dry-weight (m2 / m-2)					 y1(t) Dry-weight (m2 / m-2) 
 x2(t) Indoor CO2 (ppm)							 y2(t) Indoor CO2 (ppm)
 x3(t) Indoor temperature (◦C)					 y3(t) Indoor temperature (◦C)
 x4(t) Indoor humidity (%)						 y4(t) Indoor humidity (%)
 x5(t) PAR Inside (W / m2)					     x5(t) PAR Inside (W / m2)
----------------------------------------------------------------------------------------------------------------------------------
 d1(t) Outside Radiation (W / m2)				 u1(t) Fan (-)
 d2(t) Outdoor CO2 (ppm)						 u2(t) Toplighting status (-)
 d3(t) Outdoor temperature (◦C)					 u3(t) Heating (-) 


Project sources:
- https://applied-rl-course.netlify.app/en/module3
- https://docs.ray.io/en/latest/rllib/rllib-env.html#configuring-environments

Other sources:
- 
'''

# Import libraries
import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np
import scipy.io as sio

class MiniGreenhouse(gym.Env):
    '''
    MiniGreenhouse environment, a custom environment based on the GreenLight model
    and real mini-greenhouse.
    '''
   
    def __init__(self, env_config):
        '''
        Initialize the MiniGreenhouse environment.
        
        Parameters:
        env_config(dict): Configuration dictionary for the environment.
        '''  
        
        # Load the data from the .mat file
        data = sio.loadmat(r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\deep-reinforcement-learning\matlab\datasets\drl-env\drl-env.mat')
        
        self.time = data['time'].flatten()
        self.temp_in = data['temp_in'].flatten()
        self.rh_in = data['rh_in'].flatten()
        self.co2_in = data['co2_in'].flatten()
        self.PAR_in = data['PAR_in'].flatten()
        self.fruit_dw = data['fruit_dw'].flatten()
        
        # Define the observation and action spaces
        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0]), high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
        self.action_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        
        # Initialize the state
        self.state = None
        self.current_step = 0
        self.reset()
        
    def reset(self, *, seed=None, options=None):
        '''
        Reset the environment to the initial state.
        
        Returns:
        int: The initial observation of the environment.
        '''
        
        self.current_step = 0
        self.state = np.array([
            self.co2_in[self.current_step],
            self.temp_in[self.current_step],
            self.rh_in[self.current_step],
            self.PAR_in[self.current_step],
            self.fruit_dw[self.current_step]
        ], dtype=np.float32)
        return self.state, {}
        
    def observation(self):
        '''
        Get the current observation of the environment.
        
        Returns:
        float: The index representing the states
        '''
        
        return np.array([
            self.co2_in[self.current_step],
            self.temp_in[self.current_step],
            self.rh_in[self.current_step],
            self.PAR_in[self.current_step],
            self.fruit_dw[self.current_step]
        ], dtype=np.float32)
        
    def reward(self):
        '''
        Get the reward for the current state.
        
        Returns:
        int: Reward, {some values} if the climate controls reaches
        the setpoints, otherwise 0.
        '''
        
        # Define a reward based on the conditions you want to optimize
        return 1.0  # Placeholder reward

    def done(self):
        '''
        Check if the episode is done.
        
        Returns:
        bool: True if the episode is done, otherwise False.
        '''
        # Episode is done if we have reached the end of the data
        return self.current_step >= len(self.time) - 1
    
    def step(self, action):
        '''
        Take an action in the environment.
        
        Parameters:
        
        Based on the u(t) controls
        
        action (float):
        -  u1(t) Fan (-)                       0-1 (1 is fully open) 
		-  u2(t) Toplighting status (-)        0/1 (1 is on)
	    -  u3(t) Heating (-)                   0/1 (1 is on)

        Returns: 
        tuple: A tuple containing the new observation, reward, done flag, and additional info.
        
        '''
        # Apply action to the environment
        # Placeholder logic: increment the current step and modify state based on action
        
        self.current_step += 1
        self.state = self.observation()
        
        # Calculate reward
        reward = self.reward()
        
        # Check if done
        done = self.done()
        
        truncated = False
        
        return self.observation(), self.reward(), self.done(), truncated, {}