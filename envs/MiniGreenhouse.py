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

import matlab.engine

import os

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
        
        # Start MATLAB engine
        self.eng = matlab.engine.start_matlab()

        # Define the path to your MATLAB script
        matlab_script_path = r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\drl-mini-greenhouse\matlab\DrlGlEnvironment.m'

        # Check if the file exists
        if os.path.isfile(matlab_script_path):
            print(f"Running MATLAB script: {matlab_script_path}")
            
            # Run the MATLAB script
            # Define the season length parameter
            # 1 hour is 1/24 = 0.041 in the greenlight matlab function
            # 1 / 144 = 10 minutes
            self.season_length = 1 / 144
            self.firstDay = 6
            
            # Initialize control variables to zero for 12 timesteps
            time_steps = np.linspace(0, 11 * 300, 2)  # 12 timesteps, each 300 seconds apart
            ventilation = np.zeros(2)
            lamps = np.zeros(2)
            heater = np.zeros(2)
            
            # Create control dictionary
            controls = {
                'time': time_steps.reshape(-1, 1),
                'ventilation': ventilation.reshape(-1, 1),
                'lamps': lamps.reshape(-1, 1),
                'heater': heater.reshape(-1, 1)
            }
            
            # Print dimensions for debugging
            print("Control dimensions at initialization:")
            for key, value in controls.items():
                print(f"{key}: {value.shape}")
        
            # Save control variables to .mat file
            controls_file = 'controls.mat'
            sio.savemat(controls_file, controls)

            # Call the MATLAB function with the parameter
            self.eng.DrlGlEnvironment(self.season_length, self.firstDay, controls_file, nargout=0)
        else:
            print(f"MATLAB script not found: {matlab_script_path}")

        # Stop MATLAB engine
        # eng.quit()
        
        # Load the data from the .mat file
        data = sio.loadmat("drl-env.mat")
        
        self.time = data['time'].flatten()
        self.co2_in = data['co2_in'].flatten()
        self.temp_in = data['temp_in'].flatten()
        self.rh_in = data['rh_in'].flatten()
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
        # return self.current_step >= len(self.time) - 1
        # return self.season_length >= 0.082
        return self.current_step >= 10 # 4 hours simulation
    
    def step(self, action):
        '''
        Take an action in the environment.
        
        Parameters:
        
        Based on the u(t) controls
        
        action (discrete integer):
        -  u1(t) Fan (-)                       0-1 (1 is fully open) 
		-  u2(t) Toplighting status (-)        0/1 (1 is on)
	    -  u3(t) Heating (-)                   0/1 (1 is on)

        Returns: 
        tuple: A tuple containing the new observation, reward, done flag, and additional info.
        
        '''
        # Apply action to the environment
        # Placeholder logic: increment the current step and modify state based on action
        
        # Convert actions to discrete values
        # To-do:
        # We need to make the action/controls is change every one hour
        fan = 1 if action[0] >= 0.5 else 0
        toplighting = 1 if action[1] >= 0.5 else 0
        heating = 1 if action[2] >= 0.5 else 0
        
        # Determine the number of remaining steps
        remaining_steps = len(self.time) - self.current_step
        
        # Check if there are enough time steps left
        if remaining_steps < 2:
            # If not enough steps remain, fill the remaining steps with zeros or terminate early
            time_steps = np.zeros(2)
            ventilation = np.zeros(2)
            lamps = np.zeros(2)
            heater = np.zeros(2)
            done = True  # Terminate the episode early
        else:
            # Initialize control arrays
            time_steps = self.time[self.current_step:self.current_step + 2]
            ventilation = np.full(2, fan)
            lamps = np.full(2, toplighting)
            heater = np.full(2, heating)
            done = False  # Continue the episode
        
        # # Check if there are enough time steps left
        # if self.current_step + 2 > len(self.time):
        #     raise ValueError("Not enough time steps remaining in self.time array")
            
        #  # Initialize control arrays
        # time_steps = self.time[self.current_step:self.current_step + 2]
        # ventilation = np.full(2, fan)
        # lamps = np.full(2, toplighting)
        # heater = np.full(2, heating)
        
        # Ensure all arrays have the same length
        assert len(time_steps) == len(ventilation) == len(lamps) == len(heater), "Array lengths are not consistent"

        # controls = {
        #     'time': np.array([current_time]),
        #     'ventilation': np.array([fan]),
        #     'lamps': np.array([toplighting]),
        #     'heater': np.array([heating])
        # }
        
        # Create control dictionary
        controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': ventilation.reshape(-1, 1),
            'lamps': lamps.reshape(-1, 1),
            'heater': heater.reshape(-1, 1)
        }
        
        # Print dimensions for debugging
        print("Control dimensions in step:")
        for key, value in controls.items():
            print(f"{key}: {value.shape}")
        
        # Save control variables to .mat file
        controls_file = 'controls.mat'
        sio.savemat(controls_file, controls)
        
        # Increment the current step
        self.current_step += 1
        print("CURRENT STEPS: ", self.current_step)

        # Update the season_length and firstDay
        # self.season_length += 0.041
        self.season_length = 1 / 144
        self.firstDay += 1 / 144
        
        # Update the MATLAB environment
        self.eng.DrlGlEnvironment(self.season_length, self.firstDay, controls_file, nargout=0)

        # Load the updated data from the .mat file
        data = sio.loadmat("drl-env.mat")
        
        self.time = data['time'].flatten()
        self.co2_in = data['co2_in'].flatten()
        self.temp_in = data['temp_in'].flatten()
        self.rh_in = data['rh_in'].flatten()
        self.PAR_in = data['PAR_in'].flatten()
        self.fruit_dw = data['fruit_dw'].flatten()

        self.state = self.observation()
        
        # Calculate reward
        reward = self.reward()
        
        # Check if done
        done = self.done()
        
        truncated = False
        
        return self.observation(), reward, done, truncated, {}

    # Ensure to properly close the MATLAB engine when the environment is no longer used
    def __del__(self):
        self.eng.quit()