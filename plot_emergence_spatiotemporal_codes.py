# %% [markdown]
# # Origins of path spatiotemporal codes in hippocampal circuit
# Compare portions of time/distance cells vs place cells in light/dark environments, or tone vs no tone conditions
# 
# ### TODO:
# - Reward vs no reward LT
# - LT vs norewards vs toneLT
# - Plot timecourse (LT1, LT2, LT3, toneLT1, toneLT2, toneLT3, seqLT1, seqLT2, etc), average for all animals
# - Normalize per animal!
# - Information content (AMI distribution) vs specific cell tuning (portion time cells, portion place cells, etc over time)

# %%
# Imports
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('plot_style.mplstyle')

import yaml
import numpy as np
np.random.seed(42)
import pingouin as pg
import pandas as pd
import h5py
import os
from functions.signal_processing import smooth_1D
from tqdm import tqdm

# Params
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

# %%
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

# %%
# Specify the type of maze: linear or open
df['maze']=''
df.loc[df['task']=='LT','maze']='linear'
df.loc[df['task']=='legoLT','maze']='linear'
df.loc[df['task']=='legoToneLT','maze']='linear'
df.loc[df['task']=='legoSeqLT','maze']='linear'
df.loc[df['task']=='OF','maze']='open'
df.loc[df['task']=='legoOF','maze']='open'
df.loc[df['task']=='plexiOF','maze']='open'

# %% [markdown]
# # Spatiotemporal information in linear vs open environments

# %%
plt.figure(figsize=(.33,.5))
sns.barplot(
    data=df.query("region=='CA1' and rewards==True and prospective_temporal_pvalue<.05"),
    #x='region',
    x='maze',
    y='prospective_temporal_info',
    errorbar='se',
    palette=['C0','C6'],
    errcolor='k',
    capsize=.4
)
n_subjects=len(df.query("region=='CA1' and rewards==True and prospective_temporal_pvalue<.05").subject.unique())
n_neurons=len(df.query("region=='CA1' and rewards==True and prospective_temporal_pvalue<.05").index)
plt.title(f'Temporal')
plt.ylabel('AMI')
plt.xlabel('')
plt.xticks([])
#plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
#plt.savefig(os.path.join(params['path_to_results'],'figures', 'bar_temporal_open_vs_linear.pdf'))

# %%
# Stats

# %%
plt.figure(figsize=(.33,.5))
sns.barplot(
    data=df.query("region=='CA1' and rewards==True and spatial_pvalue<.05"),
    #x='region',
    x='maze',
    y='spatial_info',
    errorbar='se',
    palette=['C0','C6'],
    errcolor='k',
    capsize=.4
)
plt.title(f'Spatial')
plt.ylabel('AMI')
plt.xlabel('')
plt.xticks([])
plt.savefig(os.path.join(params['path_to_results'],'figures', 'bar_spatial_open_vs_linear.pdf'))

# %%
plt.figure(figsize=(.33,.5))
sns.barplot(
    data=df.query("region=='CA1' and rewards==True and distance_pvalue<.05"),
    #x='region',
    x='maze',
    y='distance_info',
    errorbar='se',
    palette=['C0','C6'],
    errcolor='k',
    capsize=.4
)
plt.title(f'Distance')
plt.ylabel('AMI')
plt.xlabel('')
plt.xticks([])
#plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures', 'bar_distance_open_vs_linear.pdf'))

# %%
plt.figure(figsize=(.33,.5))
sns.barplot(
    data=df.query("region=='CA1' and rewards==True and velocity_pvalue<.05"),
    #x='region',
    x='maze',
    y='velocity_info',
    errorbar='se',
    palette=['C0','C6'],
    errcolor='k',
    capsize=.4
)
plt.title(f'Velocity')
plt.ylabel('AMI')
plt.xlabel('')
plt.xticks([])
#plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures', 'bar_velocity_open_vs_linear.pdf'))

# %%
plt.figure(figsize=(.33,.5))
sns.barplot(
    data=df.query("region=='CA1' and rewards==True and heading_pvalue<.05"),
    #x='region',
    x='maze',
    y='heading_info',
    errorbar='se',
    palette=['C0','C6'],
    errcolor='k',
    capsize=.4
)
plt.title(f'Heading')
plt.ylabel('AMI')
plt.xlabel('')
plt.xticks([])
#plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures', 'bar_heading_open_vs_linear.pdf'))

# %%
# Stats:
print(f'N = {n_subjects} subjects, n = {n_neurons} neurons')

# %% [markdown]
# ## Normalize information metrics per animal

# %%
df.groupby(['subject','maze'])['spatial_info'].mean()

# %%
sns.barplot(data=)

# %%
# Compare tone vs no tone linear track
#tone_df = CA1_df.query("task=='legoLT' or task == 'legoSeqLT' and optoStim==False")
tone_df = CA1_df.query("task=='legoLT' or task == 'legoSeqLT'")

# Compare dark vs light conditions
dark_df = CA1_df.query("task=='legoLT'")

# %%
# Drop animals that do not have legoSeqLT exposure
tone_df = tone_df.drop(tone_df.query("subject=='M246'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M288'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M1087'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M314'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M1090'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M1092'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M1088'").index)
tone_df = tone_df.drop(tone_df.query("subject=='M1046'").index)

# %%
# For each subject, reset day count from 0
# This is to convert from date (e.g. 20180602) to the actual number of previously experienced days (e.g 3)
for subject in tone_df.subject.unique():
    unique_days = tone_df.loc[(tone_df.subject==subject) & (tone_df.task=='legoLT'),'day'].unique()
    sorted_days=np.argsort(unique_days)
    for j in range(len(unique_days)):
        tone_df.day[(tone_df['subject']==subject) & (tone_df['day']==unique_days[j]) & (tone_df['task']=='legoLT')] = sorted_days[j]

    unique_days = tone_df.loc[(tone_df.subject==subject) & (tone_df.task=='legoSeqLT'),'day'].unique()
    sorted_days=np.argsort(unique_days)
    for j in range(len(unique_days)):
        tone_df.day[(tone_df['subject']==subject) & (tone_df['day']==unique_days[j]) & (tone_df['task']=='legoSeqLT')] = sorted_days[j]


# %%
# Only keep first 3 days of exposure
tone_df = tone_df.drop(tone_df.query("day>3").index)

# %%
# Spatial coding with or without tones
plt.figure(figsize=(.75,.75))
sns.histplot(
             data=tone_df.query("task=='legoLT' and spatial_pvalue<.05"),
             x='spatial_info',
             bins=50,
             stat='probability',
             element='poly',
             alpha=.1,
             color='k',
             label='no tones')
sns.histplot(
             data=tone_df.query("task=='legoSeqLT' and spatial_pvalue<.05"),
             x='spatial_info',
             bins=50,
             stat='probability',
             element='poly',
             alpha=.1,
             color='C0',
             label='tones')
#plt.xscale('log')
plt.title('Spatial tuning')
plt.xlabel('AMI')
plt.xlim([0,.025])
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures', 'probhist_elapsed_spatial_noTone_vs_tone.pdf'))

# %%
# Temporal coding with or without tones
plt.figure(figsize=(.5,.75))
sns.barplot(
            data=tone_df.query("spatial_pvalue<.05"),
            x='day',
            hue='task',
            y='spatial_info',
            #label=['no tones','tones'],
            errorbar='se',
            capsize=.5,
            palette=['k', 'C0']
            )
plt.title('Spatial tuning')
plt.ylabel('AMI')
#plt.xticks([])
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
#plt.savefig(os.path.join(params['path_to_results'],'figures', 'bar_spatial_noTone_vs_tone.pdf'))

# %%
pg.ttest(x=tone_df.query("task=='legoLT' and spatial_pvalue<.05")['spatial_info'],
         y=tone_df.query("task=='legoSeqLT' and spatial_pvalue<.05")['spatial_info'])

# %%
# Temporal coding with or without tones
plt.figure(figsize=(2,.75))
sns.lineplot(
            data=tone_df.query("distance_pvalue<.05"),
            x='day',
            hue='task',
            y='distance_info',
            #label=['no tones','tones'],
            errorbar='se',
            hue_norm=(0,1),
            #capsize=.5,
            #palette=['k', 'C0']
            )
plt.title('distance tuning')
plt.ylabel('AMI')
#plt.xticks([])
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
#plt.savefig(os.path.join(params['path_to_results'],'figures', 'lines_overDays_temporal_noTone_vs_tone.pdf'))

# %%



