import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
from sklearn.manifold import MDS
import math
import scipy.io
import numpy as np
from scipy import stats
import mat73

#%%
import random
import matplotlib.colors as mcolors
import networkx as nx
Color_Code2 = ["red", "orange", "pink", "olive", "cyan", "black", "yellow", "green", "brown", "gray", "tomato",
               "violet", "yellowgreen", "y", "crimson", "darkgoldenrod", "darkmagenta", "indigo", "darkred",
               "darkkhaki", "orangered", 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
               'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
               'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
               'darkkhaki', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darkseagreen', 'darkslateblue',
               'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue',
               'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'goldenrod',
               'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender',
               'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
               'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen',
               'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow']

list_color = []
for color in mcolors.CSS4_COLORS:
    list_color.append(color)
random.shuffle(list_color)
Color_Code1 = [*Color_Code2, *list_color]
Color_Code = [*Color_Code1, *list_color]

#%%
## Load age 2
mat_trigger = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')
import pickle

with open('/Users/sonmjack/Downloads/simon_paper/fam1_neuron_list_age2.pkl', 'rb') as file:
    neuron_spike_young = pickle.load(file)
with open('/Users/sonmjack/Downloads/simon_paper/fam1_include_list_age2.pkl', 'rb') as file:
    include_list_young = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/fam1_old_include_list_age2.pkl', 'rb') as file:
    include_list_old = pickle.load(file)

## Load age 10
with open('/Users/sonmjack/Downloads/simon_paper/fam1_neuron_list_age10.pkl', 'rb') as file:
    neuron_spike_old = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/fam1_df_f_list_age2.pkl', 'rb') as file:
    neuron_time_list_young = pickle.load(file)
with open('/Users/sonmjack/Downloads/simon_paper/fam1_df_f_list_age10.pkl', 'rb') as file:
    neuron_time_list_old = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/mask_fam1.pkl', 'rb') as file:
    mask = pickle.load(file)

be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
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
#%%
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
All_spike = []
All_mask = []
All_laps = []
All_spike_index = []
All_df_f = []
All_tuning_curve = []
type = 'Young'

if type == 'Young':
    be_phi_list = be_phi_list_young
    neuron_spike_all= neuron_spike_young
    include_list_all = include_list_young
    neuron_time_list = neuron_time_list_young
elif type == 'Old':
    be_phi_list = be_phi_list_old
    neuron_spike_all = neuron_spike_old
    include_list_all = include_list_old
    neuron_time_list = neuron_time_list_old

for index in range(len(neuron_spike_all)):
    #time_data = be_time_list_young[index][:, 0]
    #mouse_position = be_phi_list_young[index][:, 0].reshape(1, -1)
    mouse_position = be_phi_list[index][:, 0].reshape(1, -1)
    df_f = np.array(neuron_time_list[index])
    spike_data = neuron_spike_all[index]
    spike_data[np.where(spike_data != 0)] = 1
    include_list = include_list_all[index]
    if type == 'Young':
        mask_index = mask[:, :, int(mat_trigger[10 + index * 2, 0]):int(mat_trigger[11 + index * 2, 0])]
        if index > 3:
            mask_index = mask[:, :, int(mat_trigger[10 + (index + 1) * 2, 0]):int(mat_trigger[11 + (index + 1) * 2, 0])]
    elif type == 'Old':
        mask_index = mask[:, :, int(mat_trigger[0 + index * 2, 0]):int(mat_trigger[1 + index * 2, 0])]
    delete_index_list = []

    for i in range(spike_data.shape[0]):  # sorting之后 再按排序后的数据进行分析，sorting用的是最大部分
        if include_list[i] == False:
            delete_index_list.append(i)
        elif sum(spike_data[i, :]) == 0:
            delete_index_list.append(i)
            print('silence  case')

    spike_data = np.delete(spike_data, delete_index_list, axis=0)
    mask_index = np.delete(mask_index, delete_index_list, axis=2)
    df_f = np.delete(df_f, delete_index_list, axis=0)

    max_index_list = []
    tuning_curve_list = []
    lap_neuron_list = []

    crossing_indices = np.where(np.diff(mouse_position) < -300)[1] + 1

    # 根据位置索引将位置数据分割成 lap
    lap_indices = np.split(np.arange(mouse_position.shape[1]), crossing_indices)
    angle_bins = np.linspace(0, 360, num=101)

    for i in range(spike_data.shape[0]):
        neuron_activity = spike_data[i, :].reshape(1, -1)
        lap_neuron_activity = []
        for lap_index in lap_indices:
            lap_activity = []
            for bin_index in range(len(angle_bins) - 1):
                bin_indices = np.where((mouse_position[:, lap_index[0]:lap_index[-1] + 1] >= angle_bins[bin_index]) & (
                            mouse_position[:, lap_index[0]:lap_index[-1] + 1] < angle_bins[bin_index + 1]))[1]
                neuron_slice = neuron_activity[:, lap_index[0]:lap_index[-1] + 1]
                lap_activity.append(np.sum(neuron_slice[:, bin_indices]))
            lap_neuron_activity.append(lap_activity)

        lap_neuron_activity = np.array(lap_neuron_activity)
        lap_neuron_list.append(lap_neuron_activity)

        # In trail level, equal with norm
        #tuning_curve = np.mean(lap_neuron_activity, axis=0)
        tuning_curve = np.sum(lap_neuron_activity, axis=0)
        tuning_curve_list.append(tuning_curve)


        max_index_list.append(np.argmax(tuning_curve))

    ColorCode = []
    tuning_curve_new = []
    spike_data_new = []
    spike_indices_new = []

    df_f_new = []
    lap_list_new = []


    mask_new_list = np.zeros_like(mask_index)



    it = 0
    Nodes = range(len(tuning_curve_list))
    for i in range(max(max_index_list) + 1):
        for m, j in zip(Nodes, range(len(max_index_list))):
            if max_index_list[j] == i:
                tuning_curve_new.append(tuning_curve_list[int(m)])

                spike_data_new.append(spike_data[m, :])

                # non_zero_indices_per_row[int(m)].sort()
                # spike_indices_new.append(non_zero_indices_per_row[int(m)])

                mask_new_list[:, :, it] = mask_index[:, :, m]

                df_f_new.append(df_f[m])

                lap_list_new.append(lap_neuron_list[m])
                # A_ordered_row[it,:]=Q[j,:]
                it += 1

    del tuning_curve_list, spike_data,  df_f, mask_index,lap_neuron_list

    # for lap_index in lap_indices:
    #     non_zero_indices_per_row = []
    #     for row in spike_data_new[:,lap_index[0]:lap_index[-1] + 1]:
    #         # 找到每行中不为0的元素的列索引
    #         non_zero_indices = np.where(row != 0)[0]
    #         # 添加到列表中
    #         non_zero_indices_per_row.append(list(non_zero_indices))
    spike_data_new = np.array(spike_data_new)
    All_spike.append(spike_data_new)

    spike_indices_new = []
    for lap_index in lap_indices:
        non_zero_indices_per_row = []
        for row in spike_data_new[:,lap_index[0]:lap_index[-1] + 1]:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))
        spike_indices_new.append(non_zero_indices_per_row)



    mask_new_list = np.array(mask_new_list)
    All_mask.append(mask_new_list)

    df_f_new = np.array(df_f_new)
    All_df_f.append(df_f_new)

    lap_list_new = np.array(lap_list_new)
    All_laps.append(lap_list_new)

    tuning_curve_new = np.array(tuning_curve_new)
    All_tuning_curve.append(tuning_curve_new)

    All_spike_index.append(spike_indices_new)
    print('Finish '+f'{index}')
#%%
if type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_spike.pkl', 'wb') as file:
        pickle.dump(All_spike, file)

    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_laps.pkl', 'wb') as file:
        pickle.dump(All_laps, file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_mask.pkl', 'wb') as file:
        pickle.dump(All_mask, file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f, file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_spike_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index, file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve, file)
elif type == 'Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_spike.pkl', 'wb') as file:
        pickle.dump(All_spike, file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_laps.pkl', 'wb') as file:
        pickle.dump(All_laps, file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_mask.pkl', 'wb') as file:
        pickle.dump(All_mask, file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f, file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_spike_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index, file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve, file)
#%%
# sns.heatmap(tuning_curve_list, cmap='viridis', vmin=0)
# #%%
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
#
#
# def Strength_computer(Spike_train, i, j, tau):
#     Spike_train[int(j)].sort()
#     Spike_Train_B = [*set(Spike_train[int(i)])]
#     Spike_Train_B.sort()
#     B = Spike_Shuffeler(Spike_Train_B)
#
#     Spike_train[int(i)].sort()
#     Spike_Train_A = [*set(Spike_train[int(j)])]
#     Spike_Train_A.sort()
#     A_i = Spike_Shuffeler(Spike_Train_A)
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
#         for s in range(int(B[-1])):
#             while (A[t] <= s and t < N_A):
#                 t += 1;
#             t -= 1
#             A_last = A[t];
#             f_null.append(math.exp(-(s - A_last) / tau));
#         t = 0
#         A_last = 0
#         for s in range(N_B):
#             while (A[t] <= B[s] and t < N_A):
#                 t += 1;
#             t -= 1
#             A_last = A[t];
#             f.append(math.exp(-(B[s] - A_last) / tau));
#         S_AB = max(np.sum((f - np.mean(f_null)) / (1 - np.mean(f_null))) / N_max_AB, 0)
#     return S_AB
#
# tau = 1
# k = 0
# Mouse_spike_index = []
# dy_list = []
# for lap_index in lap_indices:
#     non_zero_indices_per_row = []
#     for row in spike_data_new[:,lap_index[0]:lap_index[-1] + 1]:
#         # 找到每行中不为0的元素的列索引
#         non_zero_indices = np.where(row != 0)[0]
#         # 添加到列表中
#         non_zero_indices_per_row.append(list(non_zero_indices))
#
#     num_features = spike_data_new.shape[0]
#     spike_train = non_zero_indices_per_row
#     dy_d = np.zeros((num_features, num_features))
#
#     for m in range(num_features):
#         for n in range(num_features):
#             dy_d[m, n] = Strength_computer(spike_train, m, n, tau)
#     # dy_d[np.isnan(dy_d)] = 0
#     print('Finished No.' + f'{k}_dynamic connection')
#     dy_list.append(dy_d)
#     Mouse_spike_index.append(non_zero_indices_per_row)
#     k +=1

#%%
