import pickle
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
from sklearn.manifold import MDS
import math
import scipy.io
import numpy as np
from scipy import stats
import pandas as pd

#%%
Type = 'Young'
#Type = 'Old'
if Type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_spike.pkl', 'rb') as file:
        All_spike = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_laps.pkl', 'rb') as file:
        All_laps = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
        All_mask = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        All_df_f = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_index.pkl', 'rb') as file:
        All_spike_index = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve = pickle.load(file)
elif Type == 'Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_spike.pkl', 'rb') as file:
        All_spike =  pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_laps.pkl', 'rb') as file:
        All_laps = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
        All_mask = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        All_df_f = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_index.pkl', 'rb') as file:
        All_spike_index = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve = pickle.load(file)

be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')
import h5py

type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')

gene = type_array['genotype'][:, :].T
# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum = be_data['fam1_phi']
be_x = be_data['fam1_x']
be_y = be_data['fam1_y']
be_time = be_data['fam1_time']
be_speed = be_data['fam1_speed']

# #%%
# # Create a vector with 100 elements representing weights
# weights = All_tuning_curve[0][126,:]  # Replace with your actual weight values
# num_parts = 10
# part_size = len(weights) // num_parts
#
# # Calculate the sum of weights for each part
# part_sums = []
# for i in range(num_parts):
#     start_index = i * part_size
#     end_index = start_index + part_size
#     part_sum = np.sum(weights[start_index:end_index])
#     part_sums.append(part_sum)
#
# # Normalize the part sums to calculate percentages
# percentages = part_sums / np.sum(part_sums) * 100
#
# # Create a pie chart
# fig, ax = plt.subplots()
# ax.pie(percentages, labels=[f"{i*36}° ~{(i+1)*36}°" for i in range(num_parts)], autopct='%1.1f%%')
# ax.set_title("Normalized Percentage of Total Weight Sum for Each Part")
#
# # Add a legend
# ax.legend(title="Parts", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
# # Adjust the layout to make space for the legend
# plt.tight_layout()
# # Display the pie chart
# plt.show()

include_list = []
be_x_list_young = []
be_y_list_young = []
be_time_list_young = []
be_speed_list_young = []
be_phi_list_young = []
gene_list_young = []
for i in range(10,46,2):#0, len(mat_trigger), 2

    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_x_list_young.append(be_x[int(i/2),0])
        be_y_list_young.append(be_y[int(i / 2), 0])
        be_time_list_young.append(be_time[int(i / 2), 0])
        be_speed_list_young.append(be_speed[int(i / 2), 0])
        be_phi_list_young.append(be_phi_sum[int(i / 2), 0])
        gene_list_young.append(mat_trigger[i, 1])
be_x_list_old = []
be_y_list_old  = []
be_time_list_old  = []
be_speed_list_old  = []
be_phi_list_old  = []
gene_list_old = []
for i in range(0,10,2):#0, len(mat_trigger), 2

    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_x_list_old.append(be_x[int(i/2),0])
        be_y_list_old.append(be_y[int(i / 2), 0])
        be_time_list_old.append(be_time[int(i / 2), 0])
        be_speed_list_old.append(be_speed[int(i / 2), 0])
        be_phi_list_old.append(be_phi_sum[int(i / 2), 0])
        gene_list_old.append(mat_trigger[i, 1])
del be_x, be_data,  be_y, be_time, be_speed, be_phi_sum
#%%
index = 9
neuron = 23
x_coords = be_x_list_young[index].flatten()  # Replace with your actual x-coordinates
y_coords = be_y_list_young[index].flatten()

# Replace with your actual y-coordinates
Spike_train = All_spike[index]
non_zero_indices_per_row = []
non_zero_strength_per_row = []
for row in Spike_train:
    # 找到每行中不为0的元素的列索引
    non_zero_indices = np.where(row != 0)[0]
    non_zero_strength = row[non_zero_indices]
    # 添加到列表中
    non_zero_indices_per_row.append(list(non_zero_indices))
    non_zero_strength_per_row.append(list(non_zero_strength))
# Spike data vector
spike_indices = non_zero_indices_per_row[neuron]  # Replace with your actual spike indices
spike_strengths = non_zero_strength_per_row[neuron]  # Replace with your actual spike strengths

# Define the number of bins in each direction
num_bins_x = 30
num_bins_y = 30

# Create bins based on the range of x and y coordinates
x_bins = np.linspace(min(x_coords), max(x_coords), num_bins_x + 1)
y_bins = np.linspace(min(y_coords), max(y_coords), num_bins_y + 1)

# Initialize the heatmap matrix
heatmap_matrix = np.zeros((num_bins_y, num_bins_x))
count_matrix = np.zeros((num_bins_y, num_bins_x))
# Iterate through the time series data
for i in range(len(x_coords)):
    # Determine the bin the mouse is in
    x_bin = np.digitize(x_coords[i], x_bins,right=True) - 1
    y_bin = np.digitize(y_coords[i], y_bins,right=True) - 1

    count_matrix[y_bin, x_bin] +=1
    # If the neuron spiked at this time step, add the spike strength to the corresponding bin
    if i in spike_indices:
        spike_index = spike_indices.index(i)
        heatmap_matrix[y_bin, x_bin] += spike_strengths[spike_index]


    # devide the time they suffer from this bin除以次数，停留的时间在bin内

heatmap_matrix = heatmap_matrix*30/count_matrix
#heatmap_matrix[np.isnan(heatmap_matrix)] = 0

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap plot
heatmap = ax.imshow(heatmap_matrix, cmap='hot', interpolation='nearest', origin='lower',
                    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])

# Set labels and title
ax.set_title("Firing rate map  (Fam)", fontsize=24)
ax.set_xticks([])
ax.set_yticks([])
# Add colorbar
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Firing rate (Hz)', fontsize=24)
cbar.ax.tick_params(labelsize=24)

# Adjust the plot layout
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/firing rate map/' + str(index) + '_young_fam1_'+str(neuron)+'.png',dpi=800)
plt.show()
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # Create the heatmap plot
# countmap = ax.imshow(count_matrix, cmap='hot', interpolation='nearest', origin='lower',
#                     extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
# # Set labels and title
# ax.set_xlabel('X-coordinate', fontsize=14)
# ax.set_ylabel('Y-coordinate', fontsize=14)
# ax.set_title("Count Field (Fam)", fontsize=14)
#
# # Add colorbar
# cbar = fig.colorbar(countmap, ax=ax)
# cbar.set_label('Spike Strength', fontsize=14)
# # Adjust the plot layout
# plt.tight_layout()
# # Display the plot
# plt.show()

#%%
print('Tuning Curve wt')
if Type == 'Young':
    gene_list = gene_list_young
elif Type == 'Old':
    gene_list = gene_list_old
for index in range(len(All_spike)):
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [6, 3]})
    #fig, axes = plt.subplots(2, 1)
    tuning_curve_new = All_tuning_curve[index]
    # sns.clustermap(tuning_curve_new, cmap='viridis', row_cluster=False,vmin=0)
    # axes[0]= clustergrid.ax_heatmap
    sns.heatmap(tuning_curve_new, cmap='viridis', vmin=0,ax=axes[0])
    if gene_list[index] == 119:
        type = 'wild type'
    else:
        type = 'AD'
    axes[0].set_xlabel('Position (cm)', fontsize=13)
    axes[0].set_ylabel('Cell ID', fontsize=13)
    axes[0].set_title("Spike train cross different location No." +f'{index} {type}')

    axes[1].plot(np.sum(tuning_curve_new, axis=0), color='blue')
    axes[1].set_title("tuning curve No." +f'{index} {type}')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].set_xlabel('Position (cm)', fontsize=13)
    axes[1].set_ylabel('Spike count', fontsize=13)

    if Type == 'Young':
        plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/tuning_curve_new/'+f'{type}_{index}'+'.jpg')
    elif Type =='Old':
        plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/tuning_curve_new/' + f'{type}_{index}' + '.jpg')
    plt.close()

# #%%
# print('Noise corr')
# All_angle_noise = []
# All_noise_corr = []
# it = 0
# for index in range(len(All_spike)):
#     lap_list_new = All_laps[index]
#     res_angle_list = []
#     size = lap_list_new.shape[0]
#     res_noise = np.zeros((size,size))
#     for i in range(size):
#         for j in range(i + 1,size):
#             res_angle = []
#             for k in range(lap_list_new.shape[2]):
#                 x, y = lap_list_new[i,:, k], lap_list_new[j,:, k]
#                 res = stats.pearsonr(x, y)
#                 res = res[0]
#                 res_angle.append(res)
#             res_angle = np.nan_to_num(np.array(res_angle), nan=0)
#             res_noise[i,j] = np.mean(res_angle)
#             res_noise[j, i] = np.mean(res_angle)
#             res_angle_list.append(res_angle)
#     print('Finished ' + f'{index}')
#     All_angle_noise.append(res_angle_list)
#     All_noise_corr.append(res_noise)
# if Type == 'Young':
#     with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_angle_noise.pkl', 'wb') as file:
#         pickle.dump(All_angle_noise, file)
#     with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_noise_corr.pkl', 'wb') as file:
#         pickle.dump(All_noise_corr, file)
# elif Type =='Old':
#     with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_angle_noise.pkl', 'wb') as file:
#         pickle.dump(All_angle_noise, file)
#     with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_noise_corr.pkl', 'wb') as file:
#         pickle.dump(All_noise_corr, file)
#%%
if Type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_noise_corr.pkl', 'rb') as file:
        All_noise_corr = pickle.load(file)
elif Type =='Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_noise_corr.pkl', 'rb') as file:
        All_noise_corr = pickle.load(file)
#%%
print('Signal corr wt')
it = 0
All_signal_corr_WT = []
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(8, 66))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
for index in range(len(All_spike)):
    if gene_list[index] == 119:
        type = 'wild type'
        tuning_curve_new = All_tuning_curve[index]
        size = tuning_curve_new.shape[0]
        res_signal = np.zeros((size,size))
        for i in range(size):
            for j in range(i + 1,size):
                res_signal_corr = stats.pearsonr(tuning_curve_new[i,:], tuning_curve_new[j,:])[0]
                res_signal[i,j] = res_signal_corr
                res_signal[j, i] = res_signal_corr
        print('Finished ' + f'{index}')
        sns.heatmap(res_signal, cmap='viridis', vmin=0, vmax = 1, ax=axes[it])
        axes[it].set_xlabel('Cell ID', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Signal correlation No." + f'{index} {type}')
        it += 1
        All_signal_corr_WT.append(res_signal)
    else:
        type = 'AD'
        pass
if Type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_signal_corr_WT', 'wb') as file:
        pickle.dump(All_signal_corr_WT, file)
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/' + 'Whole signal corr' + '.jpg')
    plt.close()
elif Type == 'Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_signal_corr_WT', 'wb') as file:
        pickle.dump(All_signal_corr_WT, file)
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/' + 'Whole signal corr' + '.jpg')
    plt.close()


#%%
print('Tuning Curve wt')
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(8, 66))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
it = 0
for index in range(len(All_tuning_curve)):
    if gene_list[index] == 119:
        type = 'wild type'
        print('Finished ' + f'{index}')
        sns.heatmap(All_tuning_curve[index], cmap='viridis',vmax=1,ax=axes[it])
        axes[it].set_xlabel('Location (cm)', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Tuning curve No." + f'{index} {type}')
        it += 1
    else:
        type = 'AD'
        pass
if Type == 'Young':
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole tuning curve'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole tuning curve'+'.jpg')
    plt.close()
#%%
print('Noise corr wt')
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(8, 66))
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_noise_corr.pkl', 'rb') as file:
        All_noise_corr = pickle.load(file)
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_noise_corr.pkl', 'rb') as file:
        All_noise_corr = pickle.load(file)

it = 0
for index in range(len(All_noise_corr)):
    if gene_list[index] == 119:
        type = 'wild type'
        print('Finished ' + f'{index}')
        sns.heatmap(All_noise_corr[index], cmap='viridis', vmax = 0.3,ax=axes[it])
        axes[it].set_xlabel('Cell ID', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Noise correlation  No." + f'{index} {type}')
        it += 1
    else:
        type = 'AD'
        pass
if Type == 'Young':
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/noise_corr/'+'Whole noise corr'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/noise_corr/'+'Whole noise corr'+'.jpg')
    plt.close()

# #%%
# import math
# import random
# def Spike_Shuffeler(Spike_Train):
#     if len(Spike_Train) > 0:
#         first = Spike_Train[0]
#         last = Spike_Train[-1]
#
#         Dif_list = []
#         for i in range(len(Spike_Train) - 1):
#             Dif_list.append(Spike_Train[i + 1] - Spike_Train[i])
#
#         random.shuffle(Dif_list)
#         Spike_Train_shuf = []
#         Accomulation = first
#         for i in range(len(Spike_Train) - 1):
#             Spike_Train_shuf.append(Accomulation)
#             Accomulation = Accomulation + Dif_list[i]
#         Spike_Train_shuf.append(Accomulation)
#     else:
#         Spike_Train_shuf = []
#     return Spike_Train_shuf
# def Strength_computer(Spike_train, i, j, tau):
#     Spike_train[int(j)].sort()
#     Spike_Train_B = [*set(Spike_train[int(i)])]
#     Spike_Train_B.sort()
#     #B = Spike_Shuffeler(Spike_Train_B)
#     B = Spike_Train_B
#
#     Spike_train[int(i)].sort()
#     Spike_Train_A = [*set(Spike_train[int(j)])]
#     Spike_Train_A.sort()
#     #A_i = Spike_Shuffeler(Spike_Train_A)
#     A_i = Spike_Train_A
#     A = np.append(-1000, A_i)
#
#     f = [];
#     f_null = [];
#
#     N_B = len(B)
#     N_A = len(A_i)
#
#     if N_A * N_B == 0:
#         S_AB = 0
#     else:
#         N_max_AB = max(N_A, N_B)
#         t = 0
#         A_last = 0
#         for s in range(int(B[-1])+1):
#             while (A[t] <= s and t <= N_A):
#                 if t == N_A:
#                     if A[t] <= s:
#                         t += 1
#                         break
#                 else:
#                     t += 1
#             t -= 1
#             A_last = A[t]
#             f_null.append(math.exp(-(s - A_last) / tau))
#         t = 0
#         A_last = 0
#         for spike in B:
#             while (A[t] <= spike and t <= N_A):
#                 if t == N_A:
#                     if A[t] <= spike:
#                         t += 1
#                         break
#                 else:
#                     t += 1
#             t -= 1
#             A_last = A[t]
#             f.append(math.exp(-(spike - A_last) / tau))
#
#         if np.mean(f_null) == 1:
#             S_AB = 1
#         else:
#             S_AB = max(np.sum((f - np.mean(f_null)) / (1 - np.mean(f_null))) / N_max_AB, 0)
#     return S_AB
#
#
#
# tau = 1
# All_dy_list = []
#
#
# for i in range(len(All_spike)):#len(neuron_index)
#     num_features = All_spike[i].shape[0]
#     dy_sum = []
#     for k in range(len(All_spike_index[i])):
#         spike_train = All_spike_index[i][k]
#         dy_d = np.zeros((num_features, num_features))
#         for m in range(num_features):
#             for n in range(num_features):
# 
#                 dy_d[m,n] = Strength_computer(spike_train, m, n, tau)
#         print('Finished No.' + f'{k}_dynamic connection')
#         # dy_d[np.isnan(dy_d)] = 0
#         dy_sum.append(dy_d)
#     dy_mean = np.sum(np.array(dy_sum),axis=0)
#     #dy_d[np.isnan(dy_d)] = 0
#     All_dy_list.append(dy_mean)
#     print('----------------------------------------------------------------')
#     print('Finished No.' + f'{i} mouse')
#
# if Type == 'Young':
#     with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_EPSP_young.pkl', 'wb') as file:
#         pickle.dump(All_dy_list, file)
#
# elif Type == 'Old':
#     with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_EPSP.pkl', 'wb') as file:
#         pickle.dump(All_dy_list, file)

#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(8, 66))
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_EPSP_young.pkl', 'rb') as file:
        All_dy_list = pickle.load(file)
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_EPSP.pkl', 'rb') as file:
        All_dy_list = pickle.load(file)
it = 0
for index in range(len(All_dy_list)):
    if gene_list[index] == 119:
        type = 'wild type'
        print('Finished ' + f'{index}')
        sns.heatmap(All_dy_list[index], cmap='viridis', vmax=1, ax=axes[it])
        axes[it].set_xlabel('Cell ID', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Strength of EPSP No." + f'{index} {type}')
        it += 1
    else:
        type = 'AD'
        pass
if Type == 'Young':
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole EPSP'+'.jpg')
    plt.close()
if Type == 'Old':
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole EPSP'+'.jpg')
    plt.close()
# plt.close()
# plt.figure(figsize=(8, 6))
# sns.heatmap(All_dy_list[16], cmap='viridis', vmax=0.1)
# plt.show()

#%%
from PIL import Image

# 图像文件路径
if Type == 'Young':
    image_files = ['/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole tuning curve.jpg',
                   '/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole signal corr.jpg',
                   '/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole EPSP.jpg',
                   '/Users/sonmjack/Downloads/age2 result_fam1/noise_corr/'+'Whole noise corr.jpg']
if Type == 'Old':
    image_files = ['/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole tuning curve.jpg',
                   '/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole signal corr.jpg',
                   '/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole EPSP.jpg',
                   '/Users/sonmjack/Downloads/age10 result_fam1/noise_corr/'+'Whole noise corr.jpg']
# 打开图像并放入列表
images = [Image.open(image) for image in image_files]

# 计算新图像的总宽度和最大高度
total_width = sum(image.width for image in images)
max_height = max(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (total_width, max_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (x_offset, 0))
    x_offset += image.width

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/combined_image_WT.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/combined_image_WT.jpg')


#%% Distribution



#%%
print('Signal corr AD')
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(8, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(8, 24))
it = 0
All_signal_corr_AD = []
for index in range(len(All_spike)):
    if gene_list[index] == 116:
        type = 'AD'
        tuning_curve_new = All_tuning_curve[index]
        size = tuning_curve_new.shape[0]
        res_signal = np.zeros((size,size))
        for i in range(size):
            for j in range(i + 1,size):
                res_signal_corr = stats.pearsonr(tuning_curve_new[i,:], tuning_curve_new[j,:])[0]
                res_signal[i,j] = res_signal_corr
                res_signal[j, i] = res_signal_corr
        print('Finished ' + f'{index}')
        sns.heatmap(res_signal, cmap='viridis', vmin=0, vmax = 1, ax=axes[it])
        axes[it].set_xlabel('Cell ID', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Signal correlation No." + f'{index} {type}')
        it += 1
        All_signal_corr_AD.append(res_signal)
    else:
        type = 'wild type'
        pass
if Type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_signal_corr_AD', 'wb') as file:
        pickle.dump(All_signal_corr_AD, file)
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole signal corr'+'.jpg')
elif Type == 'Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_signal_corr_AD', 'wb') as file:
        pickle.dump(All_signal_corr_AD, file)
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole signal corr'+'.jpg')
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(8, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(8, 24))
it = 0
for index in range(len(All_tuning_curve)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        sns.heatmap(All_tuning_curve[index], cmap='viridis', vmax=1, ax=axes[it])
        axes[it].set_xlabel('Location (cm)', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Tuning curve No." + f'{index} {type}')
        it += 1
    else:
        type = 'wild type'
        pass

if Type == 'Young':
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole tuning curve'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole tuning curve'+'.jpg')
    plt.close()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(8, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(8, 24))

it = 0
for index in range(len(All_noise_corr)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        sns.heatmap(All_noise_corr[index], cmap='viridis', vmax = 0.3,ax=axes[it])
        axes[it].set_xlabel('Cell ID', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Noise correlation No." + f'{index} {type}')
        it += 1
    else:
        type = 'wild type'
        pass
if Type == 'Young':
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/noise_corr/'+'Whole noise corr'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/noise_corr/'+'Whole noise corr'+'.jpg')
    plt.close()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(8, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(8, 24))
it = 0
for index in range(len(All_dy_list)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        sns.heatmap(All_dy_list[index], cmap='viridis', vmax=1, ax=axes[it])
        axes[it].set_xlabel('Cell ID', fontsize=13)
        axes[it].set_ylabel('Cell ID', fontsize=13)
        axes[it].set_title("Strength of EPSP No." + f'{index} {type}')
        it += 1
    else:
        type = 'wild type'
        pass

if Type == 'Young':
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole EPSP'+'.jpg')
    plt.close()
if Type == 'Old':
    plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole EPSP'+'.jpg')
    plt.close()
# plt.figure(figsize=(8, 6))
# sns.heatmap(All_dy_list[16], cmap='viridis', vmax=0.1)
# plt.show()

#%%
from PIL import Image

# 图像文件路径
if Type == 'Young':
    image_files = ['/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole tuning curve.jpg',
                   '/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole signal corr.jpg',
                   '/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/'+'Whole EPSP.jpg',
                   '/Users/sonmjack/Downloads/age2 result_fam1/noise_corr/'+'Whole noise corr.jpg']
if Type == 'Old':
    image_files = ['/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole tuning curve.jpg',
                   '/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole signal corr.jpg',
                   '/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/'+'Whole EPSP.jpg',
                   '/Users/sonmjack/Downloads/age10 result_fam1/noise_corr/'+'Whole noise corr.jpg']
# 打开图像并放入列表
images = [Image.open(image) for image in image_files]

# 计算新图像的总宽度和最大高度
total_width = sum(image.width for image in images)
max_height = max(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (total_width, max_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (x_offset, 0))
    x_offset += image.width

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1/signal_corr/combined_image_AD.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1/signal_corr/combined_image_AD.jpg')