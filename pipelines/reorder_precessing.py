#%%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
import math
from sklearn.metrics import normalized_mutual_info_score
import scipy.io
import numpy as np

import mat73

# #%% dataloader
# mat_data  = mat73.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')
#%%
# age = mat_data['ageMOS']
# mask = mat_data['masks'][:,:,:]
# be_f = mat_data['expname'][:]
# ex_index = mat_data['expname'][:]
# #spike_sum = mat_data['nov_spikes']
# spike_sum = mat_data['fam1_spikes']
# include = mat_data['include']
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_neuron_list_age2.pkl', 'wb') as file:
#     pickle.dump(neuron_spike, file)
#%%
mat_trigger = np.load('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1.npy')
import pickle
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_neuron_list_age2.pkl', 'rb') as file:
    neuron_spike = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_ex_index_age2.pkl', 'rb') as file:
    ex_index = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_include_list_age2.pkl', 'rb') as file:
    include_list = pickle.load(file)

#%%
be_data = scipy.io.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_trackdata.mat')
import h5py
type_array = h5py.File('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')
gene = type_array ['genotype'][:,:].T
mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum = be_data['fam1_phi']
be_x = be_data['fam1_x']
be_y = be_data['fam1_y']
be_time = be_data['fam1_time']
#%%
# include_list = []
be_x_list = []
be_y_list = []
be_time_list = []
for i in range(10,46,2):#0, len(mat_trigger), 2

    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_x_list.append(be_x[int(i/2),0])
        be_y_list.append(be_y[int(i / 2), 0])
        be_time_list.append(be_time[int(i / 2), 0])
        # include_i = include[int(mat_trigger[i,0]):int(mat_trigger[i + 1,0])]
        # include_list.append(include_i)
        # 索引原始的be_list
#
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_include_list_age2.pkl', 'wb') as file:
#     pickle.dump(include_list, file)

