import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import scipy.io as sio
import matlab.engine
import os
from utils.ServiceFunctions import ServiceFunctions

class MiniGreenhouse2(gym.Env):
    def __init__(self, env_config, _first_day=6):
        # Start MATLAB engine
        self.eng = matlab.engine.start_matlab()

        # Path to MATLAB script
        self.matlab_script_path = r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\drl-mini-greenhouse\matlab\DrlGlEnvironment.m'

        # Initialize lists to store control values
        self.ventilation_list = []
        self.lamps_list = []
        self.heater_list = []

        # Initialize ServiceFunctions
        self.service_functions = ServiceFunctions()

        # Check if MATLAB script exists
        if os.path.isfile(self.matlab_script_path):
            self.season_length = 1 / 72
            self.first_day = _first_day
            self.init_controls()
            self.run_matlab_script()
        else:
            print(f"MATLAB script not found: {self.matlab_script_path}")

        self.load_mat_data()

        # Define observation and action spaces
        self.observation_space = Box(
            low=np.array([393.72, 21.39, 50.36, 0.00, 0, 0, 0]), 
            high=np.array([1933.33, 24.53, 90.00, 5.85, np.inf, np.inf, np.inf]), 
            dtype=np.float32
        )
        
        self.action_space = Box(
            low=np.array([0, 0, 0]), 
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )

    def init_controls(self):
        time_steps = np.linspace(300, 1200, 4)
        self.controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': np.zeros(4).reshape(-1, 1),
            'lamps': np.zeros(4).reshape(-1, 1),
            'heater': np.zeros(4).reshape(-1, 1)
        }
        self.ventilation_list.extend(self.controls['ventilation'].flatten()[-3:])
        self.lamps_list.extend(self.controls['lamps'].flatten()[-3:])
        self.heater_list.extend(self.controls['heater'].flatten()[-3:])
        sio.savemat('controls.mat', self.controls)

    def run_matlab_script(self, indoor_file=None, fruit_file=None):
        indoor_file = indoor_file or []
        fruit_file = fruit_file or []
        self.eng.DrlGlEnvironment(self.season_length, self.first_day, 'controls.mat', indoor_file, fruit_file, nargout=0)

    def load_mat_data(self):
        data = sio.loadmat("drl-env.mat")
        new_time = data['time'].flatten()[-3:]
        new_co2_in = data['co2_in'].flatten()[-3:]
        new_temp_in = data['temp_in'].flatten()[-3:]
        new_rh_in = data['rh_in'].flatten()[-3:]
        new_PAR_in = data['PAR_in'].flatten()[-3:]
        new_fruit_leaf = data['fruit_leaf'].flatten()[-3:]
        new_fruit_stem = data['fruit_stem'].flatten()[-3:]
        new_fruit_dw = data['fruit_dw'].flatten()[-3:]

        if not hasattr(self, 'time'):
            self.time = new_time
            self.co2_in = new_co2_in
            self.temp_in = new_temp_in
            self.rh_in = new_rh_in
            self.PAR_in = new_PAR_in
            self.fruit_leaf = new_fruit_leaf
            self.fruit_stem = new_fruit_stem
            self.fruit_dw = new_fruit_dw
        else:
            self.time = np.concatenate((self.time, new_time))
            self.co2_in = np.concatenate((self.co2_in, new_co2_in))
            self.temp_in = np.concatenate((self.temp_in, new_temp_in))
            self.rh_in = np.concatenate((self.rh_in, new_rh_in))
            self.PAR_in = np.concatenate((self.PAR_in, new_PAR_in))
            self.fruit_leaf = np.concatenate((self.fruit_leaf, new_fruit_leaf))
            self.fruit_stem = np.concatenate((self.fruit_stem, new_fruit_stem))
            self.fruit_dw = np.concatenate((self.fruit_dw, new_fruit_dw))

    def reset(self, *, seed=None, options=None):
        self.load_mat_data()
        return self.observation(), {}

    def observation(self):
        obs = np.array([
            self.co2_in[-1], 
            self.temp_in[-1], 
            self.rh_in[-1], 
            self.PAR_in[-1], 
            self.fruit_leaf[-1], 
            self.fruit_stem[-1], 
            self.fruit_dw[-1]
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reward(self):
        return 1.0 if self.fruit_dw[-1] > 310.0 else -1.0

    def done(self):
        return self.reward() == 1.0

    def step(self, action):
        fan = 1 if action[0] >= 0.5 else 0
        toplighting = 1 if action[1] >= 0.5 else 0
        heating = 1 if action[2] >= 0.5 else 0

        time_steps = np.linspace(300, 1200, 4)
        ventilation = np.full(4, fan)
        lamps = np.full(4, toplighting)
        heater = np.full(4, heating)

        self.ventilation_list.extend(ventilation[-3:])
        self.lamps_list.extend(lamps[-3:])
        self.heater_list.extend(heater[-3:])

        controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': ventilation.reshape(-1, 1),
            'lamps': lamps.reshape(-1, 1),
            'heater': heater.reshape(-1, 1)
        }
        sio.savemat('controls.mat', controls)

        self.season_length = 1 / 72
        self.first_day += 1 / 72

        co2_density = self.service_functions.co2ppm_to_dens(self.temp_in[-3:], self.co2_in[-3:])
        vapor_density = self.service_functions.rh_to_vapor_density(self.temp_in[-3:], self.rh_in[-3:])
        vapor_pressure = self.service_functions.vapor_density_to_pressure(self.temp_in[-3:], vapor_density)

        drl_indoor = {
            'time': self.time[-3:].astype(float).reshape(-1, 1),
            'temp_in': self.temp_in[-3:].astype(float).reshape(-1, 1),
            'rh_in': vapor_pressure.reshape(-1, 1),
            'co2_in': co2_density.reshape(-1, 1)
        }
        sio.savemat('indoor.mat', drl_indoor)

        fruit_growth = {
            'time': self.time[-1:].astype(float).reshape(-1, 1),
            'fruit_leaf': self.fruit_leaf[-1:].astype(float).reshape(-1, 1),
            'fruit_stem': self.fruit_stem[-1:].astype(float).reshape(-1, 1),
            'fruit_dw': self.fruit_dw[-1:].astype(float).reshape(-1, 1)
        }
        sio.savemat('fruit.mat', fruit_growth)

        self.run_matlab_script('indoor.mat', 'fruit.mat')
        self.load_mat_data()

        truncated = False
        return self.observation(), self.reward(), self.done(), truncated, {}

    def __del__(self):
        self.eng.quit()
