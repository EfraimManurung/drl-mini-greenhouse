import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import scipy.io as sio
import matlab.engine
import os
import pandas as pd

class MiniGreenhouse2(gym.Env):
    '''
    MiniGreenhouse environment, a custom environment based on the GreenLight model
    and real mini-greenhouse.
    '''
   
    def __init__(self, env_config, _first_day = 6):
        '''
        Initialize the MiniGreenhouse environment.
        
        Parameters:
        env_config(dict): Configuration dictionary for the environment.
        '''  
        
        # Start MATLAB engine
        self.eng = matlab.engine.start_matlab()

        # Define the path to your MATLAB script
        self.matlab_script_path = r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\drl-mini-greenhouse\matlab\DrlGlEnvironment.m'

        # Initialize lists to store control values
        self.ventilation_list = []
        self.lamps_list = []
        self.heater_list = []
        
        # Check if the file exists
        if os.path.isfile(self.matlab_script_path):
            print(f"Running MATLAB script: {self.matlab_script_path}")
            
            # Define the season length parameter
            self.season_length = 2 / 144  # 15 minutes
            self.first_day = _first_day
            
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
        time_steps = np.linspace(300, 1200, 4)  # 10 minutes (600 seconds)
        self.controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': np.zeros(4).reshape(-1, 1),
            'lamps': np.zeros(4).reshape(-1, 1),
            'heater': np.zeros(4).reshape(-1, 1)
        }
        
         # Append controls to the lists twice
        self.ventilation_list.extend(self.controls['ventilation'].flatten())
        self.lamps_list.extend(self.controls['lamps'].flatten())
        self.heater_list.extend(self.controls['heater'].flatten())
    
        sio.savemat('controls.mat', self.controls)

    def run_matlab_script(self, indoor_file=None):
        '''
        Run the MATLAB script.
        '''
        if indoor_file is None:
            indoor_file = []

        self.eng.DrlGlEnvironment(self.season_length, self.first_day, 'controls.mat', indoor_file, nargout=0)

    def load_mat_data(self):
        '''
        Load data from the .mat file.
        '''
        data = sio.loadmat("drl-env.mat")
        new_time = data['time'].flatten()
        new_co2_in = data['co2_in'].flatten()
        new_temp_in = data['temp_in'].flatten()
        new_rh_in = data['rh_in'].flatten()
        new_PAR_in = data['PAR_in'].flatten()
        new_fruit_dw = data['fruit_dw'].flatten()
        
        # Check if attributes exist, if not initialize them
        if not hasattr(self, 'time'):
            self.time = new_time
            self.co2_in = new_co2_in
            self.temp_in = new_temp_in
            self.rh_in = new_rh_in
            self.PAR_in = new_PAR_in
            self.fruit_dw = new_fruit_dw
        else:
            self.time = np.concatenate((self.time, new_time))
            self.co2_in = np.concatenate((self.co2_in, new_co2_in))
            self.temp_in = np.concatenate((self.temp_in, new_temp_in))
            self.rh_in = np.concatenate((self.rh_in, new_rh_in))
            self.PAR_in = np.concatenate((self.PAR_in, new_PAR_in))
            self.fruit_dw = np.concatenate((self.fruit_dw, new_fruit_dw))
        
        # Add debug information to verify data loading
        print(f"Loaded data lengths: time={len(self.time)}, co2_in={len(self.co2_in)}, temp_in={len(self.temp_in)}, rh_in={len(self.rh_in)}, PAR_in={len(self.PAR_in)}, fruit_dw={len(self.fruit_dw)}, ventilation={len(self.ventilation_list)}, lamps={len(self.lamps_list)}, heater={len(self.heater_list)}")

    def print_all_data(self):
        '''
        Print all the appended data.
        '''
        print("")
        print("")
        print("-------------------------------------------------------------------------------------")
        print("Print all the appended data.")
        data = {
            'Time': self.time,
            'CO2 In': self.co2_in,
            'Temperature In': self.temp_in,
            'RH In': self.rh_in,
            'PAR In': self.PAR_in,
            'Fruit Dry Weight': self.fruit_dw,
            'Ventilation': self.ventilation_list,
            'Lamps': self.lamps_list,
            'Heater': self.heater_list
        }
        df = pd.DataFrame(data)
        print(df)

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
        if self.current_step >= 6:
            self.print_all_data()
            return True
        return False

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
        
        time_steps = np.linspace(300, 1200, 4)  # 10 minutes (600 seconds)
        ventilation = np.full(4, fan)
        lamps = np.full(4, toplighting)
        heater = np.full(4, heating)
        
        # Ensure all arrays have the same length
        assert len(time_steps) == len(ventilation) == len(lamps) == len(heater), "Array lengths are not consistent"

        # Append controls to the lists
        # self.ventilation_list.append(ventilation)
        # self.ventilation_list.append(ventilation)
        # self.lamps_list.append(lamps)
        # self.lamps_list.append(lamps)
        # self.heater_list.append(heater)
        # self.heater_list.append(heater)
        
        # Append controls to the lists twice
        self.ventilation_list.extend(ventilation)
        self.lamps_list.extend(lamps)
        self.heater_list.extend(heater)

        # Create control dictionary
        controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': ventilation.reshape(-1, 1),
            'lamps': lamps.reshape(-1, 1),
            'heater': heater.reshape(-1, 1)
        }
        
        # Save control variables to .mat file
        controls_file = 'controls.mat'
        sio.savemat(controls_file, controls)
        
        # Increment the current step
        self.current_step += 1
        print("CURRENT STEPS: ", self.current_step)

        # Update the season_length and first_day
        self.season_length = 2 / 144
        self.first_day += 2 / 144
        
        # Update the MATLAB environment with the 3 latest current state
        drl_indoor = {
            'time': self.time[-3:].astype(float).reshape(-1, 1),
            'temp_in': self.temp_in[-3:].astype(float).reshape(-1, 1),
            'rh_in': self.rh_in[-3:].astype(float).reshape(-1, 1),
            'co2_in': self.co2_in[-3:].astype(float).reshape(-1, 1)
        }
        
        # Save control variables to .mat file
        indoor_file = 'indoor.mat'
        sio.savemat(indoor_file, drl_indoor)
        
        self.run_matlab_script(indoor_file)

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
