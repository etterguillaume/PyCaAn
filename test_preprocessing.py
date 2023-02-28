#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2

#%% Import dependencies
import yaml
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
from functions.dataloaders import load_data
from functions.signal_processing import compute_velocity, compute_distance_time, interpolate_2D

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
#session_path = '../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
data = load_data(session_path)

#%% Preprocessing 
data['position'] = interpolate_2D(data['position'], data['behavTime'], data['caTime'])
data['velocity'], data['running_ts'] = compute_velocity(data['position'], data['caTime'], params['speed_threshold'])

#%%
elapsed_time, traveled_distance = compute_distance_time(data['position'], data['velocity'], data['caTime'], 2)

#%%
plt.figure()
plt.plot(data['velocity'])
plt.figure()
plt.plot(elapsed_time)
plt.figure()
plt.plot(traveled_distance)

#%%
plt.figure(figsize=(2,1))
plt.scatter(data['position'][:,0], data['position'][:,1], c=data['velocity'],cmap='magma', vmin=0,vmax=30)
plt.axis('square')
plt.axis('off')
plt.colorbar(label='velocity (cm.s$^{-1}$)')


plt.figure(figsize=(1.25,1.25))
plt.scatter(elapsed_time, traveled_distance, c=data['velocity'],cmap='magma', vmin=0,vmax=30)
plt.colorbar(label='velocity (cm.s$^{-1}$)')
plt.xlabel('elapsed time (s)')
plt.xticks([0,2,4,6,8,10])
plt.xlim([0,10])
plt.yticks([0,20,40,60,80,100])
plt.ylim([0,100])
plt.ylabel('distance traveled (cm)')

# %%
from numpy import diff
# %%
test=diff(data['position'][:,0])
test=np.append(test,0)
test[test>0] = 1
test[test<=0] = -1
# %%
plt.plot(test); plt.xlim([0,2000])
#%%

# %%
plt.scatter(data['caTime'],data['position'][:,0],c=test)
# %%
