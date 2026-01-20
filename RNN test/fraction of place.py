#%%
import numpy as np
import matplotlib.colors as mcolors
import random
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from pygenstability.optimal_scales import identify_optimal_scales
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
with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
    All_tuning_curve_fam1 = pickle.load(file)

with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
    All_tuning_curve_nov = pickle.load(file)

with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
    All_tuning_curve_fam1r2 = pickle.load(file)

def prepare_data_B1(data1, group_name):
    Curve_wide = []
    for row in data1:
        for carve in row:
            # Calculate the total number of elements
            total_elements = carve.size
            # percentage = carve/np.sum(carve)
            count_above_0_5 = np.sum(carve > 1)
            # Calculate the percentage of elements above 0.5
            percentage_above_0_5 = (count_above_0_5 / total_elements)
            if percentage_above_0_5 > 0:
                Curve_wide.append(percentage_above_0_5)
        test = Curve_wide
    Curve_wide = np.array(Curve_wide)
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ 'Curve_Wide': Curve_wide.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    return df


df1 = prepare_data_B1(All_tuning_curve_fam1, 'Fam')
df2 = prepare_data_B1(All_tuning_curve_nov,'Nov')
df3 = prepare_data_B1(All_tuning_curve_fam1r2,'Fam*')

df = pd.concat([df1, df2, df3])
#%%
plt.figure(figsize=(4, 10))
positions = {'Fam': 0, 'Nov': 1, 'Fam*':2}
sns.catplot(data=df, y="Group", x="Curve_Wide", hue="Group", kind="boxen",width=0.3,k_depth = 'full',gap = 3.5)
#sns.stripplot(data=df, y="Group", x="Curve_Wide", hue="Group", color="k", size=2, ax=ax.ax)
# plt.xticks([positions['Fam'], positions['Nov'],positions['Fam*']], ['Fam', 'Nov','Fam*'],fontsize=25,)
plt.tick_params(axis='y', labelsize=18)
plt.tick_params(axis='x', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.xlabel("Percentage",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole curve WT_J' + '.pdf')
plt.show()
import scipy.stats as stats
#%%
t, p1 = stats.ttest_ind(df1['Curve_Wide'], df2['Curve_Wide'])
t, p2 = stats.ttest_ind(df1['Curve_Wide'], df3['Curve_Wide'])
t, p3 = stats.ttest_ind(df2['Curve_Wide'], df3['Curve_Wide'])

#%%
# ax = sns.jointplot(data =df , x = 'Asymmetry_W', y = 'Asymmetry_B', hue="Group")
# #ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
# ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.1)
# #plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
# plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
# #sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# #plt.title("5xFAD (age < 6)",fontsize=20)
# plt.xlabel('Asymmetry (Within subgraph)',fontsize=18)
# plt.ylabel("Asymmetry (Between subgraph)",fontsize=18)
# plt.tick_params(axis='y', labelsize=15)
# plt.tick_params(axis='x', labelsize=15)
# plt.tight_layout()
# plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole asy corr AD_J' + '.pdf')
# #plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole asy corr AD' + '.svg')
# plt.show()
#%%
import h5py
type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')
be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')

gene = type_array['genotype'][:, :].T

# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum_fam1 = be_data['fam1_phi']
be_phi_sum_nov = be_data['nov_phi']
be_phi_sum_fam1r2 = be_data['fam1r2_phi']

be_speed_sum_fam1 = be_data['fam1_speed']
be_speed_sum_nov = be_data['nov_speed']
be_speed_sum_fam1r2 = be_data['fam1r2_speed']

Type = 'Young'
env = 'fam1'
#env = 'nov'
#env = 'fam1r2'
#Type = 'Old'
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

z = 0
Fam1_cluster = []
Nov_cluster = []
Fam1r2_cluster = []


for index in range(len(gene_list_young)):#len(dy_list)
    if gene_list_young[index] == 119:
        type = 'wild type'
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
            all_results_fam1 = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
            all_results_nov = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
            all_results_fam1r2 = pickle.load(file)

        # closest_number1 = min(all_results_fam1['selected_partitions'], key=lambda x: abs(x - 133))
        # Community_fam1 = all_results_fam1['community_id'][closest_number1]
        #
        # closest_number2 = min(all_results_fam1r2['selected_partitions'], key=lambda x: abs(x - 133))
        # Community_fam1r2 = all_results_fam1r2['community_id'][closest_number2]
        #
        # # closest_number3 = min(all_results_nov['selected_partitions'], key=lambda x: abs(x - 133))
        # # Community_nov = all_results_nov['community_id'][closest_number3]
        # if index % 2 ==0:
        #     Community_nov = all_results_nov['community_id'][149]
        #     closest_number = 133
        # else:
        #     Community_nov = all_results_nov['community_id'][199]
        #     closest_number = 133
        all_results_fam1 = identify_optimal_scales(all_results_fam1, kernel_size=3, window_size=3)
        all_results_fam1r2 = identify_optimal_scales(all_results_fam1r2, kernel_size=3, window_size=3)
        all_results_nov = identify_optimal_scales(all_results_nov, kernel_size=3, window_size=3)

        for index1 in all_results_fam1['selected_partitions']:
            if index1 >= 130 and index1 <= 180:
                Fam1_cluster.append(max(all_results_fam1['community_id'][index1])+1)
        for index2 in all_results_fam1r2['selected_partitions']:
            if index2 >= 130 and index2 <= 180:
                Fam1r2_cluster.append(max(all_results_fam1r2['community_id'][index2])+1)
        for index3 in all_results_nov['selected_partitions']:
            if index3 >= 130 and index3 <= 180:
                Nov_cluster.append(max(all_results_nov['community_id'][index3]) + 1)
                # Fam1_cluster.append(max(Community_fam1) + 1)

        # Fam1r2_cluster.append(max(Community_fam1r2) + 1)
        # Nov_cluster.append(max(Community_nov) + 1)

        z = z + 1
        print('Fininsh WT' + str(index))

def prepare_data(data1,group_name):
    #dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))
    data1 = np.array(data1)
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({'Cluster': data1.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    return df

df1 = prepare_data(Fam1_cluster, 'Fam')
df2 = prepare_data(Nov_cluster, 'Nov')
df3 = prepare_data(Fam1r2_cluster, 'Fam*')

df = pd.concat([df1, df2, df3])
#%%
plt.figure(figsize=(4, 10))
positions = {'Fam': 0, 'Nov': 1, 'Fam*':2}
sns.catplot(data=df, y="Group", x="Cluster", hue="Group", kind="boxen",width=0.1,k_depth = 'full',gap = 3.5)
#sns.stripplot(data=df, y="Group", x="Curve_Wide", hue="Group", color="k", size=2, ax=ax.ax)
# plt.xticks([positions['Fam'], positions['Nov'],positions['Fam*']], ['Fam', 'Nov','Fam*'],fontsize=25,)
plt.tick_params(axis='y', labelsize=18)
plt.tick_params(axis='x', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.xlabel("Numbers of subgraph",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole cluster WT' + '.pdf')
plt.show()

#%%
import scipy.stats as stats
t, p1 = stats.ttest_ind(df1['Cluster'], df2['Cluster'])
t, p2 = stats.ttest_ind(df1['Cluster'], df3['Cluster'])
t, p3 = stats.ttest_ind(df2['Cluster'], df3['Cluster'])