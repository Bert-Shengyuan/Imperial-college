import numpy as np
import matplotlib.colors as mcolors
import random
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import numpy as np
import mat73
import scipy.io
import h5py
import networkx as nx
import networkx as nx
import pygenstability as pgs
import scipy.sparse as sp
import pandas as pd
import seaborn as sns
#%%

be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')
type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')

gene = type_array['genotype'][:, :].T
# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum_fam1 = be_data['fam1_phi']
be_phi_sum_nov = be_data['nov_phi']
be_phi_sum_fam1r2 = be_data['fam1r2_phi']
be_x = be_data['fam1_x']
be_y = be_data['fam1_y']
be_time = be_data['fam1_time']
be_speed = be_data['fam1_speed']
include_list = []
be_x_list_young = []
be_y_list_young = []
be_time_list_young = []
be_speed_list_young = []
be_phi_list_young_fam1 = []
be_phi_list_young_nov = []
be_phi_list_young_fam1r2 = []
gene_list = []
for i in range(10, 46, 2):  # 0, len(mat_trigger), 2
    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_x_list_young.append(be_x[int(i / 2), 0])
        be_y_list_young.append(be_y[int(i / 2), 0])
        be_time_list_young.append(be_time[int(i / 2), 0])
        be_speed_list_young.append(be_speed[int(i / 2), 0])
        be_phi_list_young_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
        be_phi_list_young_nov.append(be_phi_sum_nov[int(i / 2), 0])
        be_phi_list_young_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])
        gene_list.append(mat_trigger[i, 1])

del be_x, be_y, be_time, be_speed, be_phi_sum_fam1, be_phi_sum_nov

with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
    mask = pickle.load(file)

#env = 'nov'
#env = 'fam1'
env = 'fam1r2'
print('Environment:'+env)
if env == 'fam1':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_EPSP_young.pkl', 'rb') as file:
        dy_list = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_spike.pkl', 'rb') as file:
        neuron_spike = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_signal_corr_WT', 'rb') as file:
        signal_list = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_signal_corr_AD', 'rb') as file:
        signal_list_AD = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve = pickle.load(file)
elif env == "nov":
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_all_EPSP_young.pkl', 'rb') as file:
        dy_list = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_spike.pkl', 'rb') as file:
        neuron_spike = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_signal_corr_WT', 'rb') as file:
        signal_list = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_signal_corr_AD', 'rb') as file:
        signal_list_AD = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve = pickle.load(file)
elif env == 'fam1r2':
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_all_EPSP_young.pkl', 'rb') as file:
        dy_list = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_spike.pkl', 'rb') as file:
        neuron_spike = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_signal_corr_WT', 'rb') as file:
        signal_list = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_signal_corr_AD', 'rb') as file:
        signal_list_AD = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve = pickle.load(file)
#%%
#%%

from matplotlib.colors import BoundaryNorm
cluster_number = []

Color_Code2 = ['white', 'darkblue', 'red', "orange", "pink", "olive", "cyan", "yellow", "green", "brown", "gray",
               "tomato", 'limegreen']

list_color = []
for color in mcolors.CSS4_COLORS:
    list_color.append(color)
Color_Code1 = [*Color_Code2, *list_color]
Color_Code = [*Color_Code1, *list_color]

#%%
index  = 9;
z = 5
tuning_curve = All_tuning_curve[index]
mask_index = mask[index]
tuning_curve_df = pd.DataFrame(tuning_curve)
if env == 'fam1':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
        all_results = pickle.load(file)

elif env =="nov":
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
        all_results = pickle.load(file)

elif env =="fam1r2":
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
        all_results = pickle.load(file)

community_ids = all_results["community_id"][-100::5]
scale = 0
#%%
for k in range(len(community_ids)):
    Community = community_ids[k]
# 按community分
    labels_df = pd.DataFrame({'community': Community})

    # 按社区标签对神经元进行排序
    sorted_indices = labels_df.sort_values(by='community').index
    sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]
    grouped = sorted_tuning_curve_df.groupby(labels_df['community'])

    community_max_indices = {}
    for community, group in grouped:
        mean_tuning_curve = group.mean(axis=0)
        max_index = np.argmax(mean_tuning_curve)
        community_max_indices[community] = max_index

    # 根据平均tuning curve最大值的index对community进行排序
    sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

    community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

    # 更新labels_df中的community标签
    labels_df['community'] = labels_df['community'].map(community_mapping)
    Sort_community = labels_df['community'].values
    #%%
    pic = np.zeros((512, 512))
    for i in range(mask_index.shape[2]):
        index_row = np.where(mask_index[:, :, i] == True)[0]
        index_con = np.where(mask_index[:, :, i] == True)[1]
        for j in range(len(index_row)):
            pic[index_row[j], index_con[j]] = Sort_community[i]
    #cmap = sns.diverging_palette(220, 20, as_cmap=True)
    #cmap = plt.get_cmap('tab10')

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
    bounds = np.linspace(-0.5, 11.5, 13)
    norm = BoundaryNorm(bounds, cmap.N)
    sns.heatmap(pic, cmap=cmap, norm=norm)

    plt.title('Scale='+str(k*1.5/20)+', N='+str(len(sorted_communities)))
    # plt.show()
    if env == 'fam1':
        # plt.savefig('/Users/sonmjack/Downloads/figure_compare/animation/fam1/' + str(scale) +'_'+ str(
        #     index) + '_young_fam1_markov_neuron_mask.svg')
        plt.savefig('/Users/sonmjack/Downloads/figure_compare/animation/fam1/' + str(scale)+'_'+ str(
            index) + '_young_fam1_markov_neuron_mask.png')
        plt.close()

    elif env == "nov":
        # plt.savefig('/Users/sonmjack/Downloads/figure_compare/animation/nov/' + str(scale) +'_'+ str(
        #     index)+ '_young_nov_markov_neuron_mask.svg')
        plt.savefig('/Users/sonmjack/Downloads/figure_compare/animation/nov/' + str(scale) +'_'+ str(
            index)+ '_young_nov_markov_neuron_mask.png')
        plt.close()
    elif env == "fam1r2":
        # plt.savefig('/Users/sonmjack/Downloads/figure_compare/animation/fam1r2/' + str(scale) +'_'+ str(
        #     index) + '_young_fam1r2_markov_neuron.svg')
        plt.savefig('/Users/sonmjack/Downloads/figure_compare/animation/fam1r2/' + str(scale) +'_'+ str(
            index)+ '_young_fam1r2_markov_neuron.png')
        plt.close()

    scale = scale+1