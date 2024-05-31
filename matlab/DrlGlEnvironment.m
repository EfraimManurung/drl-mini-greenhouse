% Use for the environment in the DRL environment
% Using createGreenLightModel
%
% Efraim Manurung, Information Technology Group
% Wageningen University
% efraim.efraimpartoginahotasi@wur.nl
% efraim.manurung@gmail.com
%
% Based on:
% David Katzin, Simon van Mourik, Frank Kempkes, and Eldert J. Van Henten. 2020. 
% "GreenLight - An Open Source Model for Greenhouses with Supplemental Lighting: Evaluation of Heat Requirements under LED and HPS Lamps.” 
% Biosystems Engineering 194: 61–81. https://doi.org/10.1016/j.biosystemseng.2020.03.010

tic; % start the timer
%% Set up the model
% Weather argument for createGreenLightModel
seasonLength = 0.041; % season length in days
firstDay = 6; % days since beginning of data 

% Choice of lamp
lampType = 'led'; 

% From IoT dataset
[outdoor_iot, indoor_iot, controls_iot, startTime] = loadMiniGreenhouseData2(firstDay, seasonLength);

% From DRL dataset
[controls_drl, startTime] = loadDrlDataset(firstDay, seasonLength);

% DynamicElements for the measured data
v.tAir = DynamicElement('v.tAir', [floor(indoor_iot(:,1)) indoor_iot(:,2)]);
v.rhAir = DynamicElement('v.rhAir', [floor(indoor_iot(:,1)) indoor_iot(:,3)]);
v.co2Air = DynamicElement('v.co2Air', [floor(indoor_iot(:,1)) indoor_iot(:,4)]);
v.iInside = DynamicElement('v.iInside', [floor(indoor_iot(:,1)) indoor_iot(:,5)]);

% number of seconds since beginning of year to startTime
secsInYear = seconds(startTime-datetime(year(startTime),1,1,0,0,0));

%indoor_iot(:,7) = skyTempRdam(indoor_iot(:,3), datenum(startTime)+indoor_iot(:,1)/86400); % add sky temperature
outdoor_iot(:,7) = outdoor_iot(:,3) - 10;
outdoor_iot(:,8) = soilTempNl(secsInYear+outdoor_iot(:,1)); % add soil temperature

%% Create an instance of createGreenLight with the default Vanthoor parameters

% Make two mini-greenhouse models

% led = createGreenLightModel('led', outdoor_iot, startTime, controls, indoor_iot);
drl_env = createGreenLightModel(lampType, outdoor_iot, startTime, controls_drl);

% Parameters for mini-greenhouse
setParamsMiniGreenhouse(drl_env);      % set greenhouse structure
setMiniGreenhouseLedParams(drl_env);   % set lamp params
%% Control parameters
% Read setGIParams.m about the explanation and default values of the control parameters
% setParam(led, 'rhMax', 50);        % upper bound on relative humidity  

% Set initial values for crop
% start with 3.12 plants/m2, assume they are each 2 g = 6240 mg/m2.
% Check the setGlinit.m for more information
% Default values    
drl_env.x.cLeaf.val = 0.7*6240;     
drl_env.x.cStem.val = 0.25*6240;    
drl_env.x.cFruit.val = 0.05*6240;   

% Default values
% led.x.cLeaf.val = 0.01;     
% led.x.cStem.val = 0.01;    
% led.x.cFruit.val = 0.01; 

%% Run simulation
solveFromFile(drl_env, 'ode15s');

% set data to a fixed step size (5 minutes)
drl_env = changeRes(drl_env, 300);

toc;
%% Get RRMSEs between simulation and measurements
% Check that the measured data and the simulations have the same size. 
% If one of them is bigger, some data points of the longer dataset will be
% discarded
mesLength = length(v.tAir.val(:,1)); % the length (array size) of the measurement data
simLength = length(drl_env.x.tAir.val(:,1)); % the length (array size) of the simulated data
compareLength = min(mesLength, simLength);

% Apply the multiplier to drl_env.a.rhIn values
multiplier_rh = 0.61; %0.85; %0.61; %0.83;
if exist('multiplier_rh', 'var') && ~isempty(multiplier_rh)
    drl_env.a.rhIn.val(:,2) = drl_env.a.rhIn.val(:,2) * multiplier_rh;
end

% Add more value for the rParGhLamp
% measured / simulated = 1.473 / 3.755 = 0.392
multiplier_irradiance = 0.39;
if exist('multiplier_irradiance', 'var') && ~isempty(multiplier_irradiance)
    drl_env.a.rParGhLamp.val(:,2) = drl_env.a.rParGhLamp.val(1:compareLength,2) * multiplier_irradiance;
end

% Added PAR from sun and lamp
sunLampIrradiance = (drl_env.a.rParGhSun.val(1:compareLength,2)+drl_env.a.rParGhLamp.val(1:compareLength,2));

% Calculate RRMSE
rrmseTair = (sqrt(mean((drl_env.x.tAir.val(1:compareLength,2)-v.tAir.val(1:compareLength,2)).^2))./mean(v.tAir.val(1:compareLength,2))) * 100;
rrmseRhair = (sqrt(mean((drl_env.a.rhIn.val(1:compareLength,2)-v.rhAir.val(1:compareLength,2)).^2))./mean(v.rhAir.val(1:compareLength,2))) * 100;
rrmseCo2air  = (sqrt(mean((drl_env.x.co2Air.val(1:compareLength,2)-v.co2Air.val(1:compareLength,2)).^2))./mean(v.co2Air.val(1:compareLength,2))) * 100;
rrmseIinside = (sqrt(mean((sunLampIrradiance - v.iInside.val(1:compareLength,2)).^2))./mean(v.iInside.val(1:compareLength,2))) * 100;

% Calculate RMSE
rmseTair = sqrt(mean((drl_env.x.tAir.val(1:compareLength,2) - v.tAir.val(1:compareLength,2)).^2));
rmseRhair = sqrt(mean((drl_env.a.rhIn.val(1:compareLength,2)-v.rhAir.val(1:compareLength,2)).^2));
rmseCo2air = sqrt(mean((drl_env.x.co2Air.val(1:compareLength,2) - v.co2Air.val(1:compareLength,2)).^2));
rmseIinside = sqrt(mean((sunLampIrradiance - v.iInside.val(1:compareLength,2)).^2));

% Calculate ME 
meTair = mean(drl_env.x.tAir.val(1:compareLength,2) - v.tAir.val(1:compareLength,2));
meRhair = mean(drl_env.a.rhIn.val(1:compareLength,2)- v.rhAir.val(1:compareLength,2));
meCo2air = mean(drl_env.x.co2Air.val(1:compareLength,2) - v.co2Air.val(1:compareLength,2));
meIinside = mean(sunLampIrradiance - v.iInside.val(1:compareLength,2));

% Save the output 
save exampleMiniGreenhouse

% Display the multiplier values
fprintf('\n');
if exist('multiplier_rh', 'var') && ~isempty(multiplier_rh)
    fprintf('Multiplier RH: %.2f\n', multiplier_rh);
end

if exist('multiplier_irradiance', 'var') && ~isempty(multiplier_irradiance)
    fprintf('Multiplier Irradiance: %.2f\n', multiplier_irradiance);
end

fprintf('Season Length: %d day(s) \n', seasonLength);
fprintf('---------------------------------------------\n');
fprintf('| Metric          | Value       | Unit       \n');
fprintf('---------------------------------------------\n');
fprintf('| RRMSE Tair      | %-12.2f| %%              \n', rrmseTair);
fprintf('| RRMSE Rhair     | %-12.2f| %%              \n', rrmseRhair);
%fprintf('| RRMSE Co2air    | %-12.2f| %%              \n', rrmseCo2air);
fprintf('| RRMSE IInside   | %-12.2f| %%              \n', rrmseIinside);
fprintf('| RMSE Tair       | %-12.2f| °C              \n', rmseTair);
fprintf('| RMSE Rhair      | %-12.2f| %%              \n', rmseRhair);
%fprintf('| RMSE Co2air     | %-12.2f| ppm             \n', rmseCo2air);
fprintf('| RMSE IInside    | %-12.2f| W m^{-2}        \n', rmseIinside);
fprintf('| ME Tair         | %-12.2f| °C              \n', meTair);
fprintf('| ME Rhair        | %-12.2f| %%              \n', meRhair);
%fprintf('| ME Co2air       | %-12.2f| ppm             \n', meCo2air);
fprintf('| ME Iinside      | %-12.2f| W m^{-2}        \n', meIinside);
fprintf('---------------------------------------------\n');

%% Extract the simulated data from the DRL environment
time = drl_env.x.tAir.val(:, 1);                    % Time
temp_in = drl_env.x.tAir.val(:, 2);       % Indoor temperature
rh_in = drl_env.a.rhIn.val(:, 2);          % Indoor humidity
co2_in = drl_env.a.co2InPpm.val(:, 2);           % Indoor co2
PAR_in = drl_env.a.rParGhSun.val(:, 2) + drl_env.a.rParGhLamp.val(:, 2); % PAR inside
fruit_dw = drl_env.x.cFruit.val(:, 2); % Fruit dry weight

% Save the extracted data to a .mat file
save('datasets/drl-env/drl-env.mat', 'time', 'temp_in', 'rh_in', 'co2_in', 'PAR_in', 'fruit_dw');

%% Clear the workspace
clear;
