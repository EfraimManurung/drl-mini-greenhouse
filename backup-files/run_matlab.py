import matlab.engine
import os

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the path to your MATLAB script
matlab_script_path = r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\drl-mini-greenhouse\matlab\DrlGlEnvironment.m'

# Check if the file exists
if os.path.isfile(matlab_script_path):
    print(f"Running MATLAB script: {matlab_script_path}")
    
    # Run the MATLAB script
    # Define the season length parameter
    season_length = 0.123

    # Call the MATLAB function with the parameter
    eng.DrlGlEnvironment(season_length, nargout=0)
else:
    print(f"MATLAB script not found: {matlab_script_path}")

# Stop MATLAB engine
eng.quit()