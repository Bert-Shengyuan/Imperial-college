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
import pickle
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
mat_trigger = np.load('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1.npy')
33

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_neuron_list_age2.pkl', 'rb') as file:
    neuron_spike_young_fam1 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/nov_list_age2.pkl', 'rb') as file:
    neuron_spike_young_nov = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_neuron_list_age2.pkl', 'rb') as file:
    neuron_spike_young_fam1r2 = pickle.load(file)


with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_include_list_age2.pkl', 'rb') as file:
    include_list_young = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_old_include_list_age2.pkl', 'rb') as file:
    include_list_old = pickle.load(file)

## Load age 10
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_neuron_list_age10.pkl', 'rb') as file:
    neuron_spike_old_fam1 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/nov_neuron_list_age10.pkl', 'rb') as file:
    neuron_spike_old_nov = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_neuron_list_age10.pkl', 'rb') as file:
    neuron_spike_old_fam1r2 = pickle.load(file)



with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_df_f_list_age2.pkl', 'rb') as file:
    neuron_time_list_young_fam1 = pickle.load(file)
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1_df_f_list_age10.pkl', 'rb') as file:
    neuron_time_list_old_fam1 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_df_f_list_age2.pkl', 'rb') as file:
    neuron_time_list_young_fam1r2 = pickle.load(file)
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_df_f_list_age10.pkl', 'rb') as file:
    neuron_time_list_old_fam1r2 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/nov_df_f_list_age2.pkl', 'rb') as file:
    neuron_time_list_young_nov = pickle.load(file)
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/nov_df_f_list_age10.pkl', 'rb') as file:
    neuron_time_list_old_nov = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/mask_fam1.pkl', 'rb') as file:
    mask = pickle.load(file)
#%%
be_data = scipy.io.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_trackdata.mat')
import h5py

type_array = h5py.File('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')

gene = type_array['genotype'][:, :].T
# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum_fam1 = be_data['fam1_phi']
be_phi_sum_nov = be_data['nov_phi']
be_phi_sum_fam1r2 = be_data['fam1r2_phi']

be_speed_sum_fam1 = be_data['fam1_speed']
be_speed_sum_nov = be_data['nov_speed']
be_speed_sum_fam1r2 = be_data['fam1r2_speed']


#%%
be_phi_list_young_fam1 = []
be_phi_list_young_fam1r2 = []
be_phi_list_young_nov = []

be_speed_list_young_fam1 = []
be_speed_list_young_nov = []
be_speed_list_young_fam1r2 = []


gene_list_young = []
for i in range(10,46,2):#0, len(mat_trigger), 2

    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_phi_list_young_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
        be_phi_list_young_nov.append(be_phi_sum_nov[int(i / 2), 0])
        be_phi_list_young_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

        be_speed_list_young_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
        be_speed_list_young_nov.append(be_speed_sum_nov[int(i / 2), 0])
        be_speed_list_young_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

        gene_list_young.append(mat_trigger[i, 1])

be_phi_list_old_fam1 = []
be_phi_list_old_nov = []
be_phi_list_old_fam1r2 = []

be_speed_list_old_fam1 = []
be_speed_list_old_nov = []
be_speed_list_old_fam1r2 = []


gene_list_old = []
for i in range(0,10,2):#0, len(mat_trigger), 2
        be_phi_list_old_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
        be_phi_list_old_nov.append(be_phi_sum_nov[int(i / 2), 0])
        be_phi_list_old_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

        be_speed_list_old_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
        be_speed_list_old_nov.append(be_speed_sum_nov[int(i / 2), 0])
        be_speed_list_old_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])


        gene_list_old.append(mat_trigger[i, 1])

del be_data, be_phi_sum_fam1,be_phi_sum_nov,be_phi_sum_fam1r2
#%%

type = 'Young'

if type == 'Young':
    be_phi_list_fam1 = be_phi_list_young_fam1
    be_phi_list_nov = be_phi_list_young_nov
    be_phi_list_fam1r2 = be_phi_list_young_fam1r2

    be_speed_list_fam1 = be_speed_list_young_fam1
    be_speed_list_nov = be_speed_list_young_nov
    be_speed_list_fam1r2 = be_speed_list_young_fam1r2

    neuron_spike_fam1 = neuron_spike_young_fam1
    neuron_spike_nov = neuron_spike_young_nov
    neuron_spike_fam1r2 = neuron_spike_young_fam1r2

    include_list_all = include_list_young

    neuron_time_fam1 = neuron_time_list_young_fam1
    neuron_time_nov = neuron_time_list_young_nov
    neuron_time_fam1r2 = neuron_time_list_young_fam1r2

elif type == 'Old':
    be_phi_list_fam1 = be_phi_list_old_fam1
    be_phi_list_nov = be_phi_list_old_nov
    be_phi_list_fam1r2 = be_phi_list_old_fam1r2

    be_speed_list_fam1 = be_speed_list_old_fam1
    be_speed_list_nov = be_speed_list_old_nov
    be_speed_list_fam1r2 = be_speed_list_old_fam1r2

    neuron_spike_fam1 = neuron_spike_old_fam1
    neuron_spike_nov = neuron_spike_old_nov
    neuron_spike_fam1r2 = neuron_spike_old_fam1r2

    include_list_all = include_list_old

    neuron_time_fam1 = neuron_time_list_old_fam1
    neuron_time_nov = neuron_time_list_old_nov
    neuron_time_fam1r2 = neuron_time_list_old_fam1r2

import pandas as pd
def seperate_data(spike_data,mouse_position):
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
        # tuning_curve = np.mean(lap_neuron_activity, axis=0)
        # if type =='fam1':
        #     angle = be_phi_list_young_fam1[i].reshape(-1)
        # elif type == 'nov':
        #     angle = be_phi_list_young_nov[i].reshape(-1)
        # elif type=='fam1r2':
        #     angle = be_phi_list_young_fam1r2[i].reshape(-1)

        angle = mouse_position.reshape(-1)
        bins = np.linspace(0, 360, num=101)

        angle_categories = pd.cut(angle, bins, right=False)

        # 计算每个区间的占比
        angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()
        time = angle.shape[0]/30
        tuning_curve = np.sum(lap_neuron_activity, axis=0) / (time*angle_distribution.values)
        tuning_test = np.sum(lap_neuron_activity, axis=0)
        tuning_curve_list.append(tuning_curve)

        max_index_list.append(np.argmax(tuning_curve))

    return max_index_list,tuning_curve_list, lap_neuron_list,lap_indices

#%%
All_spike_Sfam1 = []
All_spike_Snov = []
All_spike_Sfam1r2 = []

#All_spike_index = []
All_df_f_Sfam1 = []
All_df_f_Snov = []
All_df_f_Sfam1r2 = []

All_tuning_curve_Sfam1 = []
All_tuning_curve_Snov = []
All_tuning_curve_Sfam1r2 = []

All_mask = []


All_spike_index_Sfam1 = []
All_spike_index_Snov = []
All_spike_index_Sfam1r2 = []

for index in range(len(neuron_spike_fam1)):
    #time_data = be_time_list_young[index][:, 0]
    #mouse_position = be_phi_list_young[index][:, 0].reshape(1, -1)

    mouse_position_fam1 = be_phi_list_fam1[index][:, 0].reshape(1, -1)
    mouse_position_nov = be_phi_list_nov[index][:, 0].reshape(1, -1)
    mouse_position_fam1r2 = be_phi_list_fam1r2[index][:, 0].reshape(1, -1)

    df_f_fam1 = np.array(neuron_time_fam1[index])
    df_f_nov = np.array(neuron_time_nov[index])
    df_f_fam1r2 = np.array(neuron_time_fam1r2[index])

    spike_data_fam1 = neuron_spike_fam1[index]
    spike_data_nov = neuron_spike_nov[index]
    spike_data_fam1r2 = neuron_spike_fam1r2[index]

    include_list = include_list_all[index]
    if type == 'Young':
        mask_index = mask[:, :, int(mat_trigger[10 + index * 2, 0]):int(mat_trigger[11 + index * 2, 0])]
        if index > 3:
            mask_index = mask[:, :, int(mat_trigger[10 + (index + 1) * 2, 0]):int(mat_trigger[11 + (index + 1) * 2, 0])]
    elif type == 'Old':
        mask_index = mask[:, :, int(mat_trigger[0 + index * 2, 0]):int(mat_trigger[1 + index * 2, 0])]
    delete_index_list = []

    for i in range(spike_data_fam1.shape[0]):  # sorting之后 再按排序后的数据进行分析，sorting用的是最大部分
        if include_list[i] == False:
            delete_index_list.append(i)
            print('silence  case')

    #和1一致
    spike_data_fam1 = np.delete(spike_data_fam1, delete_index_list, axis=0)
    spike_data_nov = np.delete(spike_data_nov, delete_index_list, axis=0)
    spike_data_fam1r2= np.delete(spike_data_fam1r2, delete_index_list, axis=0)

    spike_strength_fam1 = spike_data_fam1.copy()
    spike_strength_nov = spike_data_nov.copy()
    spike_strength_fam1r2 = spike_data_fam1r2.copy()

    mask_index = np.delete(mask_index, delete_index_list, axis=2)
    df_f_fam1 = np.delete(df_f_fam1, delete_index_list, axis=0)
    df_f_nov = np.delete(df_f_nov, delete_index_list, axis=0)
    df_f_fam1r2 = np.delete(df_f_fam1r2, delete_index_list, axis=0)

    spike_strength_fam1[np.where(spike_strength_fam1 != 0)] = 1
    spike_strength_nov[np.where(spike_strength_nov != 0)] = 1
    spike_strength_fam1r2[np.where(spike_strength_fam1r2 != 0)] = 1

    # After processing
    max_fam1, tuning_curve_fam1,lap_neuron_fam1,lap_indices_fam1 = seperate_data(spike_strength_fam1,mouse_position_fam1)
    #tuning_curve_fam1 = np.array(tuning_curve_fam1)
    max_nov, tuning_curve_nov, lap_neuron_nov,lap_indices_nov = seperate_data(spike_strength_nov, mouse_position_nov)
    #tuning_curve_nov = np.array(tuning_curve_nov)
    max_fam1r2, tuning_curve_fam1r2, lap_neuron_fam1r2, lap_indices_fam1r2 = seperate_data(spike_strength_fam1r2, mouse_position_fam1r2)
    # seperate data


    ColorCode = []
    tuning_curve_fam1_new = []
    spike_data_fam1_new = []
    df_f_fam1_new = []
    lap_list_fam1_new = []

    tuning_curve_fam1r2_new = []
    spike_data_fam1r2_new = []
    df_f_fam1r2_new = []
    lap_list_fam1r2_new = []

    tuning_curve_nov_new = []
    spike_data_nov_new = []
    df_f_nov_new = []
    lap_list_nov_new = []

    mask_new_list = np.zeros_like(mask_index)

    it = 0
    Nodes = range(len(tuning_curve_fam1))
    for i in range(max(max_fam1) + 1):
        for m, j in zip(Nodes, range(len(max_fam1))):
            if max_fam1[j] == i:
                tuning_curve_fam1_new.append(tuning_curve_fam1[int(m)])
                tuning_curve_fam1r2_new.append(tuning_curve_fam1r2[int(m)])
                tuning_curve_nov_new.append(tuning_curve_nov[int(m)])

                spike_data_fam1_new.append(spike_data_fam1[m, :])
                spike_data_nov_new.append(spike_data_nov[m, :])
                spike_data_fam1r2_new.append(spike_data_fam1r2[m, :])

                # non_zero_indices_per_row[int(m)].sort()
                # spike_indices_new.append(non_zero_indices_per_row[int(m)])

                mask_new_list[:, :, it] = mask_index[:, :, m]

                df_f_fam1_new.append(df_f_fam1[m])
                df_f_nov_new.append(df_f_nov[m])
                df_f_fam1r2_new.append(df_f_fam1r2[m])

                lap_list_fam1_new.append(lap_neuron_fam1[m])
                lap_list_nov_new.append(lap_neuron_nov[m])
                lap_list_fam1r2_new.append(lap_neuron_fam1r2[m])
                # A_ordered_row[it,:]=Q[j,:]
                it += 1


    spike_data_fam1_new = np.array(spike_data_fam1_new)
    spike_data_nov_new = np.array(spike_data_nov_new)
    spike_data_fam1r2_new = np.array(spike_data_fam1r2_new)

    All_spike_Sfam1.append(spike_data_fam1_new)
    All_spike_Snov.append(spike_data_nov_new)
    All_spike_Sfam1r2.append(spike_data_fam1r2_new)

    spike_indices_new_Sfam1 = []
    for lap_index in lap_indices_fam1:
        non_zero_indices_per_row = []
        for row in spike_data_fam1_new[:, lap_index[0]:lap_index[-1] + 1]:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))
        spike_indices_new_Sfam1.append(non_zero_indices_per_row)

    spike_indices_new_Snov = []
    for lap_index in lap_indices_nov:
        non_zero_indices_per_row = []
        for row in spike_data_nov_new[:, lap_index[0]:lap_index[-1] + 1]:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))
        spike_indices_new_Snov.append(non_zero_indices_per_row)

    spike_indices_new_Sfam1r2 = []
    for lap_index in lap_indices_fam1r2:
        non_zero_indices_per_row = []
        for row in spike_data_fam1r2_new[:, lap_index[0]:lap_index[-1] + 1]:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))
        spike_indices_new_Sfam1r2.append(non_zero_indices_per_row)




    df_f_fam1_new = np.array(df_f_fam1_new)
    df_f_nov_new = np.array(df_f_nov_new)
    df_f_fam1r2_new = np.array(df_f_fam1r2_new)

    All_df_f_Sfam1.append(df_f_fam1_new)
    All_df_f_Snov.append(df_f_nov_new)
    All_df_f_Sfam1r2.append(df_f_fam1r2_new)

    tuning_curve_fam1_new = np.array(tuning_curve_fam1_new)
    tuning_curve_nov_new = np.array(tuning_curve_nov_new)
    tuning_curve_fam1r2_new = np.array(tuning_curve_fam1r2_new)

    All_tuning_curve_Sfam1.append(tuning_curve_fam1_new)
    All_tuning_curve_Snov.append(tuning_curve_nov_new)
    All_tuning_curve_Sfam1r2.append(tuning_curve_fam1r2_new)

    All_spike_index_Sfam1.append(spike_indices_new_Sfam1)
    All_spike_index_Snov.append(spike_indices_new_Snov)
    All_spike_index_Sfam1r2.append(spike_indices_new_Sfam1r2)



    All_mask.append(mask_new_list)

    #还缺一个算corr的lap

    print('Finish '+f'{index}')
#%%
if type == 'Young':
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_S_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f_Sfam1, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_S_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve_Sfam1, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_S_spike.pkl', 'wb') as file:
        pickle.dump(All_spike_Sfam1, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_S_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index_Sfam1, file)

    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_S_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f_Snov, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_S_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve_Snov, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_S_spike.pkl', 'wb') as file:
        pickle.dump(All_spike_Snov, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_S_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index_Snov, file)


    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_S_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f_Sfam1r2, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_S_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve_Sfam1r2, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_S_spike.pkl', 'wb') as file:
        pickle.dump(All_spike_Sfam1r2, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_S_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index_Sfam1r2, file)



    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_all_Smask.pkl', 'wb') as file:
        pickle.dump(All_mask, file)



elif type == 'Old':
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_S_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f_Sfam1, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_S_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve_Sfam1, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_S_spike.pkl', 'wb') as file:
        pickle.dump(All_spike_Sfam1, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_S_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index_Sfam1, file)

    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_S_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f_Snov, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_S_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve_Snov, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_S_spike.pkl', 'wb') as file:
        pickle.dump(All_spike_Snov, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_S_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index_Snov, file)


    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_S_df_f.pkl', 'wb') as file:
        pickle.dump(All_df_f_Sfam1r2, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_S_tuning curve.pkl', 'wb') as file:
        pickle.dump(All_tuning_curve_Sfam1r2, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_S_spike.pkl', 'wb') as file:
        pickle.dump(All_spike_Sfam1r2, file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_S_index.pkl', 'wb') as file:
        pickle.dump(All_spike_index_Sfam1r2, file)


    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_all_Smask.pkl', 'wb') as file:
        pickle.dump(All_mask, file)