import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json
import paho.mqtt.client as mqtt

class ServiceFunctions:
    def __init__(self):
        print("Service Functions initiated!")
        
        # Initiate the MQTT client
        self.client = mqtt.Client()
        
    def co2ppm_to_dens(self, _temp, _ppm):
        '''
        co2ppm_to_dens Convert CO2 molar concetration [ppm] to density [kg m^{-3}]
        
        Usage:
            co2_density = co2ppm_to_dens(temp, ppm) 
        Inputs:
           temp        given temperatures [°C] (numeric vector)
           ppm         CO2 concetration in air (ppm) (numeric vector)
           Inputs should have identical dimensions
         Outputs:
           co2Dens     CO2 concentration in air [mg m^{-3}] (numeric vector)
        
         Calculation based on ideal gas law pV=nRT, with pressure at 1 atm

        Based on the GreenLight model
        '''
        
        R = 8.3144598; # molar gas constant [J mol^{-1} K^{-1}]
        C2K = 273.15; # conversion from Celsius to Kelvin [K]
        M_CO2 = 44.01e-3; # molar mass of CO2 [kg mol^-{1}]
        P = 101325; # pressure (assumed to be 1 atm) [Pa]
        
        # Ensure inputs are numpy arrays for element-wise operations
        _temp = np.array(_temp)
        _ppm = np.array(_ppm)
        
        # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
        # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm   
        # co2Dens = P*10^-6*ppm*M_CO2./(R*(temp+C2K)); 
        _co2_density = P * 10**-6 * _ppm * M_CO2 / (R * (_temp + C2K)) * 1e6
        
        return _co2_density
    
    def rh_to_vapor_density(self, _temp, _rh):
        '''
        Convert relative humidity [%] to vapor density [kg{H2O} m^{-3}]
        
        Usage:
            vaporDens = rh2vaporDens(temp, rh)
        Inputs:
            temp        given temperatures [°C] (numeric vector)
            rh          relative humidity [%] between 0 and 100 (numeric vector)
            Inputs should have identical dimensions
        Outputs:
            vaporDens   absolute humidity [kg{H2O} m^{-3}] (numeric vector)
        
        Calculation based on 
            http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
        '''
        
        # constants
        R = 8.3144598  # molar gas constant [J mol^{-1} K^{-1}]
        C2K = 273.15  # conversion from Celsius to Kelvin [K]
        Mw = 18.01528e-3  # molar mass of water [kg mol^{-1}]
        
        # parameters used in the conversion
        p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        
        # Ensure inputs are numpy arrays for element-wise operations
        _temp = np.array(_temp)
        _rh = np.array(_rh)
        
        # Saturation vapor pressure of air in given temperature [Pa]
        satP = p[0] * np.exp(p[2] * _temp / (_temp + p[1]))
        
        # Partial pressure of vapor in air [Pa]
        pascals = (_rh / 100.0) * satP
        
        # convert to density using the ideal gas law pV=nRT => n=pV/RT 
        # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
        # number of kg in a m^3, where Mw is the molar mass of water.
        vaporDens = pascals * Mw / (R * (_temp + C2K))
        
        return vaporDens
    
    
    def vapor_density_to_pressure(self, _temp, _vaporDens):
        '''
        Convert vapor density [kg{H2O} m^{-3}] to vapor pressure [Pa]
        
        Usage:
            vaporPres = vaporDens2pres(temp, vaporDens)
        Inputs:
            temp        given temperatures [°C] (numeric vector)
            vaporDens   vapor density [kg{H2O} m^{-3}] (numeric vector)
            Inputs should have identical dimensions
        Outputs:
            vaporPres   vapor pressure [Pa] (numeric vector)
        
        Calculation based on 
            http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
        '''
        
        # parameters used in the conversion
        p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        
        # Ensure inputs are numpy arrays for element-wise operations
        _temp = np.array(_temp)
        _vaporDens = np.array(_vaporDens)
        
        # Convert relative humidity from vapor density
        _rh = _vaporDens / self.rh_to_vapor_density(_temp, 100)  # relative humidity [0-1]
        
        # Saturation vapor pressure of air in given temperature [Pa]
        _satP = p[0] * np.exp(p[2] * _temp / (_temp + p[1]))
        
        vaporPres = _satP * _rh
        
        return vaporPres
        
    
    def plot_all_data(self, time, co2_in, temp_in, rh_in, PAR_in, fruit_leaf, fruit_stem, fruit_dw, ventilation, lamps, heater, rewards):
        '''
        Plot all the appended data.
        
        Parameters:
        - time: List of time values
        - co2_in: List of CO2 values
        - temp_in: List of temperature values
        - rh_in: List of relative humidity values
        - par_in: List of PAR values
        - fruit_leaf: List of fruit leaf values
        - fruit_stem: List of fruit stem values
        - fruit_dw: List of fruit dry weight values
        - ventilation: List of ventilation control values
        - lamps: List of lamps control values
        - heater: List of heater control values
        '''
        
        # Create subplots with 4 rows and 3 columns
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
        
        # Data to be plotted along with their titles
        data = [
            (co2_in, 'CO2 In [ppm]'),
            (temp_in, 'Temperature In [°C]'),
            (rh_in, 'RH In [%]'),
            (PAR_in, 'PAR In [W/m2]'),
            (fruit_leaf, r'Fruit Leaf [mg (CH$_2$O) m$^{-2}$]'),
            (fruit_stem, r'Fruit Stem [mg (CH$_2$O) m$^{-2}$]'),
            (fruit_dw, r'Fruit Dry Weight [mg (CH$_2$O) m$^{-2}$]'),
            (ventilation, 'Ventilation [-]'),
            (lamps, 'Lamps [-]'),
            (heater, 'Heater [-]'),
            (rewards, 'Rewards [-]')
        ]
        
        # Plot each dataset in a subplot
        for i, (ax, (y_data, title)) in enumerate(zip(axes.flatten(), data)):
            ax.plot(time, y_data)  # Plot data
            # ax.set_title(title)  # Set the title of the subplot
            ax.set_xlabel('Time')  # Set the x-axis label
            ax.set_ylabel(title)  # Set the y-axis label
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.set_xticks(range(len(time)))  # Set the positions of the ticks on the x-axis
            ax.set_xticklabels(time, rotation=45, ha='right')  # Set the tick labels on the x-axis
        
        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
    def format_data_in_JSON(self, time, ventilation, lamps, heater):
        '''
        Convert data to JSON format and print it.
        
        Parameters:
        - time: List of time values
        - ventilation: List of ventilation control values
        - lamps: List of lamps control values
        - heater: List of heater control values
        '''
        
        def convert_to_native(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.int32, np.int64, np.float32, np.float64)):
                return value.item()
            else:
                return value

        data = {
            "time": [convert_to_native(v) for v in time],
            "ventilation": [convert_to_native(v) for v in ventilation],
            "lamps": [convert_to_native(v) for v in lamps],
            "heater": [convert_to_native(v) for v in heater]
        }

        json_data = json.dumps(data, indent=4)
        print("JSON DATA: ", json_data)
        return json_data
    
    def publish_mqtt_data(self, json_data, broker="localhost", port=1883, topic="greenhouse/drl-controls"):
        '''
        Publish JSON data to an MQTT broker.
        
        Parameters:
        - json_data: JSON formatted data to publish
        - broker: MQTT broker address
        - port: MQTT broker port
        - topic: MQTT topic to publish data to
        '''
        
        def on_connect(client, userdata, flags, rc):
            print("Connected with result code " + str(rc))
            client.publish(topic, str(json_data))
        
        self.client.on_connect = on_connect
        
        self.client.connect(broker, port, 60)
        self.client.loop_start()
