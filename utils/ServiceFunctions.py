import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ServiceFunctions:
    def __init__(self):
        print("Service Functions initiated!")
        
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
        
    
    def plot_all_data(self, time, co2_in, temp_in, rh_in, par_in, fruit_leaf, fruit_stem, fruit_dw, ventilation, lamps, heater):
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
        
        # Create a DataFrame from the data
        data = {
            'Time': time,
            'CO2 In': co2_in,
            'Temperature In': temp_in,
            'RH In': rh_in,
            'PAR In': par_in,
            'Fruit leaf': fruit_leaf,
            'Fruit stem': fruit_stem,
            'Fruit Dry Weight': fruit_dw,
            'Ventilation': ventilation,
            'Lamps': lamps,
            'Heater': heater
        }
        
        df = pd.DataFrame(data)

        # Create subplots
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))
        fig.tight_layout(pad=5.0)

        # Adjust layout to create more space between plots
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # Plot each variable
        plot_order = [
            ('CO2 In', 'CO2 [ppm]', co2_in),
            ('RH In', 'Relative Humidity [%]', rh_in),
            ('Temperature In', 'Temperature [°C]', temp_in),
            ('PAR In', 'PAR [W/m²]', par_in),
            ('Ventilation', 'Control Signal [-]', ventilation),
            ('Lamps', 'Control Signal [-]', lamps),
            ('Heater', 'Control Signal [-]', heater),
            ('Fruit Leaf', r'Dry-weight [mg (CH$_2$O) m$^{-2}$]', fruit_leaf),
            ('Fruit Stem', r'Dry-weight [mg (CH$_2$O) m$^{-2}$]', fruit_stem),
            ('Fruit Dry Weight', r'Dry-weight [mg (CH$_2$O) m$^{-2}$]', fruit_dw)
        ]

        # Plot the data in the specified order
        for i, (title, ylabel, data) in enumerate(plot_order):
            row, col = divmod(i, 2)
            axes[row, col].plot(time, data)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].set_ylabel(ylabel, fontsize=8)
            axes[row, col].set_xticklabels(time, rotation=45, ha='right')  # Rotate x-axis labels

        # Set x labels only for the bottom row
        axes[4, 0].set_xlabel('Time', fontsize=8)
        axes[4, 1].set_xlabel('Time', fontsize=8)

        # Show the plot
        plt.show()