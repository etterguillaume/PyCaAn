# %%
# Imports
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
import yaml
import numpy as np
import pingouin as pg
import pandas as pd
import os
import h5py
from tqdm import tqdm

# Custom functions
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data, smooth_1D, smooth_2D
# Params
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

# %%
# Pick example session/mouse
session_path='../../datasets/calcium_imaging/CA1/M990/M990_legoOF_20190114'
data = load_data(session_path)
data = preprocess_data(data, params)

# %%
# Correlated pixels
plt.imshow(data['corrProj'].T, cmap='viridis', vmin=0, vmax=1)
plt.axis('off')
# plt.title('Correlated pixels')
cax = plt.axes([.95, 0.15, 0.05, 0.7])
plt.colorbar(cax=cax, label='Pixel correlation')
plt.savefig(params['path_to_results']+"/figures/corrProj.pdf")

# %%
# Plot SFPs with color
import matplotlib
#cmap = matplotlib.cm.get_cmap('nipy_spectral')
cmap = matplotlib.cm.get_cmap('viridis')
#numNeurons=data['SFPs'].shape[0]
numNeurons=100
threshold=.2

colorSFPs = np.zeros((data['SFPs'].shape[1],data['SFPs'].shape[2],4))
SFPs = data['SFPs']/np.max(data['SFPs'])
for cell_i in range(numNeurons):
    color = cmap(cell_i/numNeurons)
    colorSFPs[SFPs[cell_i]>threshold] = color #data['SFPs'][cell_i]/np.max(data['SFPs'][cell_i])
    #colorSFPs[SFPs[cell_i]>threshold,3] = np.sqrt((SFPs[cell_i][SFPs[cell_i]>threshold])/np.max(SFPs[cell_i]))

plt.figure(figsize=(4,2))
plt.subplot(131)
plt.imshow(colorSFPs.transpose(1,0,2))
plt.axis('image')
plt.title(f'displaying {numNeurons}/{data["neuralData"].shape[1]} neurons')
cax = plt.axes([.15, 0.25, 0.25, 0.05])
cbar=plt.colorbar(cax=cax, label='Neuron #', orientation="horizontal", ticks=[0,1])
cbar.ax.set_xticklabels(['1', '200'])

plt.subplot(132)
for i in range(numNeurons):
    color = cmap(i/(numNeurons))
    plt.plot(data['caTime'],data['neuralData'][:,i]/5+i,
            c=color,
            linewidth=.3)

plt.xlim(0,30)
plt.xticks([0,15,30])
#plt.yticks([0,400],[0,200])
plt.ylim(0,numNeurons)
plt.xlabel('Time (s)')
plt.ylabel('Neuron #')

plt.subplot(133)
plt.imshow(data['binaryData'].T, cmap='GnBu', aspect='auto', interpolation='none')
plt.xlim(0,900)
plt.xticks([0,450,900],[0,15,30])
#plt.yticks([0,400],[0,200])
plt.ylim(0,numNeurons)
plt.xlabel('Time (s)')
#plt.ylabel('Neuron #')

# plt.subplot(133)
# binarized_traces = binarize_ca_traces(data['caTrace'],params['z_threshold'], params['sampling_frequency'])
# plt.imshow(binarized_traces.T,interpolation='none', aspect='auto', cmap='magma')
# plt.xlim(0,500)
# plt.yticks([])
# plt.ylim(0,numNeurons)

plt.tight_layout()
plt.savefig(params['path_to_results']+'/figures/SFPs_traces.pdf')

# %% Open all tuning curves
sessionList=os.listdir(os.path.join(params['path_to_results'],'tuning_data'))
data_list = [] # cell, animal,day,condition,AMI,pvalue
sessionList=os.listdir(os.path.join(params['path_to_results'],'tuning_data'))
for session in tqdm(sessionList):
    if os.path.exists(os.path.join(params['path_to_results'],'tuning_data',session,'info.yaml')): # TODO: Report any file missing
        info_file=open(os.path.join(params['path_to_results'],'tuning_data',session,'info.yaml'),'r')
        session_info = yaml.full_load(info_file)
        spatial_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'spatial_tuning.h5'), 'r')
        prospective_temporal_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'prospective_temporal_tuning.h5'), 'r')
        retrospective_temporal_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'retrospective_temporal_tuning.h5'), 'r')
        prospective_distance_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'prospective_distance_tuning.h5'), 'r')
        retrospective_distance_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'retrospective_distance_tuning.h5'), 'r')
        direction_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'direction_tuning.h5'), 'r')
        velocity_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'velocity_tuning.h5'), 'r')

        for i in range(session_info['numNeurons']):
            data_list.append( #This will create one list entry per cell
                {
                    # Basic conditions
                    'cell_ID':i,
                    'subject':session_info['subject'],
                    'region':session_info['region'],
                    'day':session_info['day'],
                    'condition':session_info['condition'],
                    'task':session_info['task'],
                    'darkness':session_info['darkness'],
                    'optoStim':session_info['optoStim'],
                    'rewards':session_info['rewards'],
                    # Info metrics
                    'spatial_info':spatial_file['AMI'][i],
                    'prospective_temporal_info':prospective_temporal_file['AMI'][i],
                    'prospective_distance_info':prospective_distance_file['AMI'][i],
                    'retrospective_temporal_info':retrospective_temporal_file['AMI'][i],
                    'retrospective_distance_info':retrospective_distance_file['AMI'][i],
                    'velocity_info':velocity_file['AMI'][i],
                    'heading_info':direction_file['AMI'][i],
                    'spatial_pvalue':spatial_file['p_value'][i],
                    'prospective_temporal_pvalue':prospective_temporal_file['p_value'][i],
                    'prospective_distance_pvalue':prospective_distance_file['p_value'][i],
                    'retrospective_temporal_pvalue':retrospective_temporal_file['p_value'][i],
                    'retrospective_distance_pvalue':retrospective_distance_file['p_value'][i],
                    'velocity_pvalue':velocity_file['p_value'][i],
                    'heading_pvalue':direction_file['p_value'][i]
                }
            )
        
        # Close files
        info_file.close()
        prospective_temporal_file.close()
        retrospective_temporal_file.close()
        spatial_file.close()
        prospective_distance_file.close()
        retrospective_distance_file.close()
        direction_file.close()
        velocity_file.close()

df = pd.DataFrame(data_list)

#%% Example place cell in open field
df_OF = df.query("region=='CA1' and task=='OF'")
pc_row=df_OF['spatial_info'].argmax()
session_info=df_OF.iloc[pc_row]
PC_cell_ID=session_info['cell_ID']
session=session_info['region']+'_'+session_info['subject']+'_'+session_info['task']+'_'+str(session_info['day'])
spatial_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'spatial_tuning.h5'), 'r')
spatial_file['tuning_curves']


PF=spatial_file['tuning_curves'][PC_cell_ID,:,:]
peak_prob=np.nanmax(PF)
PF[np.isnan(PF)]=0
PF=smooth_2D(PF, params)
plt.imshow(PF, interpolation='Bicubic')
plt.title(peak_prob.round(2))
#plt.colorbar()
plt.axis('off')

plt.savefig(params['path_to_results']+"/figures/example_placecell.pdf")
spatial_file.close()

#%% Example time cell in open field
#df_OF = df.query("region=='CA1' and task=='OF'")
tc_row=df_OF['retrospective_temporal_info'].argmax()
session_info=df_OF.iloc[tc_row]
TC_cell_ID=session_info['cell_ID']
session=session_info['region']+'_'+session_info['subject']+'_'+session_info['task']+'_'+str(session_info['day'])
retrospective_temporal_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'retrospective_temporal_tuning.h5'), 'r')

TF=retrospective_temporal_file['tuning_curves'][TC_cell_ID,:]
peak_prob=np.nanmax(TF)
TF[np.isnan(TF)]=0
TF=smooth_1D(TF, params)
plt.figure(figsize=(1,.75))
plt.plot(TF,linewidth=2,color='C1')
plt.title(peak_prob.round(2))
plt.xticks([0,len(TF)],[0,params['max_temporal_length']])
plt.yticks([])
plt.xlabel('Elapsed time (s)')

plt.savefig(params['path_to_results']+"/figures/example_timecell.pdf")
retrospective_temporal_file.close()

#%% Example distance cell in open field
#df_OF = df.query("region=='CA1' and task=='OF'")
dc_row=df_OF['retrospective_distance_info'].argmax()
session_info=df_OF.iloc[dc_row]
DC_cell_ID=session_info['cell_ID']
session=session_info['region']+'_'+session_info['subject']+'_'+session_info['task']+'_'+str(session_info['day'])
retrospective_distance_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'retrospective_distance_tuning.h5'), 'r')

DF=retrospective_distance_file['tuning_curves'][DC_cell_ID,:]
peak_prob=np.nanmax(DF)
DF[np.isnan(DF)]=0
DF=smooth_1D(DF, params)
plt.figure(figsize=(1,.75))
plt.plot(DF,linewidth=2,color='C5')
plt.title(peak_prob.round(2))
plt.xticks([0,len(DF)],[0,params['max_distance_length']])
plt.yticks([])
plt.xlabel('Distance travelled (cm)')

plt.savefig(params['path_to_results']+"/figures/example_distancecell.pdf")
retrospective_distance_file.close()

#%% Example speed cell in open field
#df_OF = df.query("region=='CA1' and task=='OF'")
vc_row=df_OF['velocity_info'].argmax()
session_info=df_OF.iloc[vc_row]
VC_cell_ID=session_info['cell_ID']
session=session_info['region']+'_'+session_info['subject']+'_'+session_info['task']+'_'+str(session_info['day'])
velocity_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'velocity_tuning.h5'), 'r')

VF=velocity_file['tuning_curves'][VC_cell_ID,:]
peak_prob=np.nanmax(VF)
VF[np.isnan(VF)]=0
VF=smooth_1D(VF, params)
plt.figure(figsize=(1,.75))
plt.plot(VF,linewidth=2,color='C6')
plt.title(peak_prob.round(2))
plt.xticks([0,len(VF)],[0,params['max_velocity_length']])
plt.yticks([])
plt.xlabel('Speed (cm.s$^{-1}$)')

plt.savefig(params['path_to_results']+"/figures/example_speedcell.pdf")
velocity_file.close()

#%% Example heading cell in open field
#df_OF = df.query("region=='CA1' and task=='OF'")
hc_row=df_OF['heading_info'].argmax()
session_info=df_OF.iloc[hc_row]
HC_cell_ID=session_info['cell_ID']
session=session_info['region']+'_'+session_info['subject']+'_'+session_info['task']+'_'+str(session_info['day'])
direction_file = h5py.File(os.path.join(params['path_to_results'],'tuning_data',session,'direction_tuning.h5'), 'r')

HF=direction_file['tuning_curves'][HC_cell_ID,:]
peak_prob=np.nanmax(HF)
HF[np.isnan(HF)]=0
HF=smooth_1D(HF, params)
plt.figure(figsize=(1,.75))
plt.polar(np.radians(np.arange(0,360,params['directionBinSize'])),
          HF,
          linewidth=2,color='C3')
plt.title(peak_prob.round(2))
#plt.xticks([0,len(HF)],[0,np.pi])
plt.yticks([])
plt.xlabel('Heading (º)')

plt.savefig(params['path_to_results']+"/figures/example_headingcell.pdf")
direction_file.close()

#%%