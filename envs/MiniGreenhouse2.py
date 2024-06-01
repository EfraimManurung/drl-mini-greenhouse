import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import scipy.io as sio
import matlab.engine
import os

class MiniGreenhouse2(gym.Env):
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
        self.matlab_script_path = r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\drl-mini-greenhouse\matlab\DrlGlEnvironment.m'

        # Check if the file exists
        if os.path.isfile(self.matlab_script_path):
            print(f"Running MATLAB script: {self.matlab_script_path}")
            
            # Define the season length parameter
            self.season_length = 1 / 144  # 10 minutes
            self.firstDay = 6
            
            # Initialize control variables to zero for 2 timesteps
            self.init_controls()
            
            # Call the MATLAB function with the parameter
            self.run_matlab_script()
        else:
            print(f"MATLAB script not found: {self.matlab_script_path}")

        # Load the data from the .mat file
        self.load_mat_data()
    
        # Define the observation and action spaces
        self.define_spaces()
        
        # Initialize the state
        self.reset()

    def init_controls(self):
        '''
        Initialize control variables.
        '''
        time_steps = np.linspace(0, 300, 2)  # 10 minutes (600 seconds)
        self.controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': np.zeros(2).reshape(-1, 1),
            'lamps': np.zeros(2).reshape(-1, 1),
            'heater': np.zeros(2).reshape(-1, 1)
        }
        sio.savemat('controls.mat', self.controls)

    def run_matlab_script(self):
        '''
        Run the MATLAB script.
        '''
        self.eng.DrlGlEnvironment(self.season_length, self.firstDay, 'controls.mat', nargout=0)

    def load_mat_data(self):
        '''
        Load data from the .mat file.
        '''
        data = sio.loadmat("drl-env.mat")
        self.time = data['time'].flatten()
        self.co2_in = data['co2_in'].flatten()
        self.temp_in = data['temp_in'].flatten()
        self.rh_in = data['rh_in'].flatten()
        self.PAR_in = data['PAR_in'].flatten()
        self.fruit_dw = data['fruit_dw'].flatten()
        # Add debug information to verify data loading
        print(f"Loaded data lengths: time={len(self.time)}, co2_in={len(self.co2_in)}, temp_in={len(self.temp_in)}, rh_in={len(self.rh_in)}, PAR_in={len(self.PAR_in)}, fruit_dw={len(self.fruit_dw)}")

    def define_spaces(self):
        '''
        Define the observation and action spaces.
        '''
        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0]), high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
        self.action_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        '''
        Reset the environment to the initial state.
        
        Returns:
        int: The initial observation of the environment.
        '''
        self.current_step = 0
        self.state = self.observation()
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
        
        action (discrete integer):
        -  u1(t) Fan (-)                       0-1 (1 is fully open) 
        -  u2(t) Toplighting status (-)        0/1 (1 is on)
        -  u3(t) Heating (-)                   0/1 (1 is on)

        Returns: 
        tuple: A tuple containing the new observation, reward, done flag, and additional info.
        '''
        # Convert actions to discrete values
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
        
        # Ensure all arrays have the same length
        assert len(time_steps) == len(ventilation) == len(lamps) == len(heater), "Array lengths are not consistent"

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
        self.season_length = 1 / 144
        self.firstDay += 1 / 144
        
        # Update the MATLAB environment
        self.run_matlab_script()

        # Load the updated data from the .mat file
        self.load_mat_data()

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