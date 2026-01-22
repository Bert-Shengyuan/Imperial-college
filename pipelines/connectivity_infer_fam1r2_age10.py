#%%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
import math
from sklearn.metrics import normalized_mutual_info_score
import scipy.io
import numpy as np

import mat73

# #%% COMMENTED OUT: dataloader - pickle files already exist
# mat_data  = mat73.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')
# #%%
# spike_time_all = mat_data['fam1r2_df_f']
# #%%
# age = mat_data['ageMOS']
# be_f = mat_data['expname'][:]
# ex_index = mat_data['expname'][:]
# #spike_sum = mat_data['nov_spikes']
# spike_sum = mat_data['fam1r2_spikes']
# #%%
# be_data = scipy.io.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_trackdata.mat')
# import h5py
# type_array = h5py.File('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')
# gene = type_array ['genotype'][:,:].T
# mat_label = np.zeros((gene.shape[0],4))
# # be_phi_sum = be_data['nov_phi']
# be_phi_sum = be_data['fam1r2_phi']
#
# #%% exist conflict of index, so just use part of data
# for i in range(gene.shape[0]):#gene.shape[0] 969,3236
#     if i == gene.shape[0]-1:
#         mat_label[i, 0] = i
#         mat_label[i, 1] = gene[i, 0]
#         mat_label[i, 2] = age[i]
#         mat_label[i, 3] = len(spike_sum[i][0])
#         break
#     if i ==0:
#         mat_label[i,0] = 0.1
#         mat_label[i, 1] = gene[i, 0]
#         mat_label[i, 2] = age[i]
#         mat_label[i, 3] = len(spike_sum[i][0])
#     else:
#         if len(spike_sum[i][0]) != len(spike_sum[i+1][0])or len(spike_sum[i][0]) != len(spike_sum[i-1][0]):
#         #if ex_index[i] != ex_index[i + 1]:
#             mat_label[i, 0] = i
#             mat_label[i, 1] = gene[i, 0]
#             mat_label[i, 2] = age[i]
#             mat_label[i, 3] = len(spike_sum[i][0])
#         elif be_f[i] != be_f[i+1] or be_f[i] != be_f[i-1]:
#             mat_label[i, 0] = i
#             mat_label[i, 1] = gene[i, 0]
#             mat_label[i, 2] = age[i]
#             mat_label[i, 3] = len(spike_sum[i][0])
#         else:
#             mat_label[i,0] = 0
#
# mat_trigger = mat_label[mat_label[:,0]!=0]
# path = '/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1r2.npy'
# np.save(path,mat_trigger)
# #%%
# mat_trigger = np.load('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1r2.npy')
# #%%  age older  fam1
# neuron_spike = []
# be_list = []
# neuron_time_list = []
# gene_list = []
# age_list = []
# mutual_list = []
# for i in range(0,10,2):#0, len(mat_trigger), 2
#     neuron_times = []
#     if i == len(mat_trigger):
#         break
#     else:
#         cell_df = spike_time_all[int(mat_trigger[i, 0]):int(mat_trigger[i + 1, 0])]
#         num_rows = len(cell_df)
#         for j in range(num_rows):
#             window_size = 50
#             # 计算移动平均值
#             smooth_data = np.convolve(cell_df[j][0], np.ones(window_size) / window_size, mode='valid')
#             neuron_times.append(smooth_data)
#         neuron_time_list.append(neuron_times)
#         # cell_array = spike_sum[int(mat_trigger[i,0]):int(mat_trigger[i + 1,0])]
#         # num_rows = len(cell_array)
#         # num_column = len(cell_array[0][0])
#         # neurons = np.zeros((num_rows, num_column))
#         #
#         # # 长度不一致需要截断处理然后再看
#         # for j in range(num_rows):
#         #     neurons[j, :] = (cell_array[j][0] * 10).astype(int)
#         # neuron_spike.append(neurons)
#         # # 索引原始的be_list
#         # be_list.append(be_phi_sum[int(i/2),0])
#         # gene_list.append(mat_trigger[i,1])
#         # age_list.append(mat_trigger[i,2])
#
#         # num_features = num_rows
#         # mutinfo_d = np.zeros((num_features, num_features))
#         # #计算每两line数据之间的互信息
#         # for m in range(num_features):
#         #     for n in range(m + 1, num_features):
#         #         a = neurons[m, :][neurons[m, :] != 0]
#         #         b = neurons[n, :][neurons[n, :] != 0]
#         #         max_size = max(len(a), len(b))
#         #         a= np.pad(a, (0, max_size - len(a)), mode='constant')
#         #         b= np.pad(b, (0, max_size - len(b)), mode='constant')
#         #         mi = normalized_mutual_info_score(a, b)
#         #
#         #         mutinfo_d[m, n] = mi
#         #         mutinfo_d[n, m] = mi
#         # mutual_list.append(mutinfo_d)
# import pickle
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_df_f_list_age10.pkl', 'wb') as file:
#     pickle.dump(neuron_time_list, file)
#
# # with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/nov_neuron_list_age10.pkl', 'wb') as file:
# #     pickle.dump(neuron_spike, file)
# #%% save data
# import pickle
#
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_ex_index_age10.pkl', 'wb') as file:
#     pickle.dump(ex_index, file)
#
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_neuron_list_age10.pkl', 'wb') as file:
#     pickle.dump(neuron_spike, file)
#
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_mutual_list_age10.pkl', 'wb') as file:
#     pickle.dump(mutual_list, file)
#
# with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/gene_list_age10.pkl', 'wb') as file:
#     pickle.dump(gene_list, file)
#%% Load pre-existing pickle files
import pickle
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_neuron_list_age10.pkl', 'rb') as file:
    neuron_spike = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_ex_index_age10.pkl', 'rb') as file:
    ex_index = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_mutual_list_age10.pkl', 'rb') as file:
    mutual_list = pickle.load(file)
#%%
import math
from joblib import Parallel, delayed
import random
def Spike_Shuffeler(Spike_Train):
    if len(Spike_Train) > 0:
        first = Spike_Train[0]
        last = Spike_Train[-1]

        Dif_list = []
        for i in range(len(Spike_Train) - 1):
            Dif_list.append(Spike_Train[i + 1] - Spike_Train[i])

        random.shuffle(Dif_list)
        Spike_Train_shuf = []
        Accomulation = first
        for i in range(len(Spike_Train) - 1):
            Spike_Train_shuf.append(Accomulation)
            Accomulation = Accomulation + Dif_list[i]
        Spike_Train_shuf.append(Accomulation)
    else:
        Spike_Train_shuf = []
    return Spike_Train_shuf


def Strength_computer(Spike_train, i, j, tau):
    Spike_train[int(j)].sort()
    Spike_Train_B = [*set(Spike_train[int(i)])]
    Spike_Train_B.sort()
    B = Spike_Shuffeler(Spike_Train_B)

    Spike_train[int(i)].sort()
    Spike_Train_A = [*set(Spike_train[int(j)])]
    Spike_Train_A.sort()
    A_i = Spike_Shuffeler(Spike_Train_A)
    A = np.append(-1000, A_i)

    f = [];
    f_null = [];

    N_B = len(B)
    N_A = len(A_i)

    if N_A * N_B == 0:
        S_AB = 0
    else:
        N_max_AB = max(N_A, N_B)
        t = 0
        A_last = 0
        for s in range(int(B[-1])):
            while (A[t] <= s and t < N_A):
                t += 1;
            t -= 1
            A_last = A[t];
            f_null.append(math.exp(-(s - A_last) / tau));
        t = 0
        A_last = 0
        for s in range(N_B):
            while (A[t] <= B[s] and t < N_A):
                t += 1;
            t -= 1
            A_last = A[t];
            f.append(math.exp(-(B[s] - A_last) / tau));
        S_AB = max(np.sum((f - np.mean(f_null)) / (1 - np.mean(f_null))) / N_max_AB, 0)
    return S_AB
#%%
freq = 30.9
tau = 5
dy_list = []
#这里把数据给破坏了
for k in range(len(neuron_spike)):
    Spike_train = neuron_spike[k].copy()
    num_features = neuron_spike[k].shape[0]
    dy_d = np.zeros((num_features, num_features))
    for m in range(num_features):
        for n in range(num_features):
            dy_d[m,n] = Strength_computer(Spike_train, m, n, tau)
    dy_d[np.isnan(dy_d)] = 0
    dy_list.append(dy_d)
#%%

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_famr2_age10.pkl', 'wb') as file:
    pickle.dump(dy_list, file)

#%%

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_famr2_age10.pkl', 'rb') as file:
    dy_list = pickle.load(file)
#%%
def embeding_color(neuron,be,index):
    color_tensor = np.zeros([neuron.shape[0],index])
    top_n_indices = np.argpartition(neuron, -index, axis=1)[:, -index:]

    for i in range(top_n_indices.shape[0]):
        for j in range(top_n_indices.shape[1]):
            color_tensor[i,j] = be[top_n_indices[i,j]]

    # 找到每一行最大的6个值对应的列索引
    color_mean = np.mean(color_tensor, axis=1)
    return color_mean
#%%
import matplotlib.colors as mcolors
for k in range(len(neuron_spike)):
    color_means1 = embeding_color(neuron_spike[k], be_list[k], 10)
    mds = MDS(n_components=3, random_state=42)
    mds_result = mds.fit_transform(dy_list[k])
    if gene_list[k] == 119:
        type = 'wt'
    else:
        type = 'AD'
    plt.figure(figsize=(13, 10))
    sns.heatmap(dy_list[k],vmin=0, vmax=1)
    plt.title("Strength matrix")
    plt.xlabel("Neurons")
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/' + 'dynamic' + f'-{type}-' + f'{k}.jpg')
    plt.close()
    # 可视化降维结果
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2],c=color_means1)

    ax.set_title('Modified locally linear embedding of dynamic distence'+f'-{type}')
    fig.colorbar(p)
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/'+'dynamic distence'+f'-{type}-'+f'{k}.jpg')
    plt.close()

    for m in range(np.size(neuron_spike[k],0)):
        cmap = plt.get_cmap('viridis')  # 使用'viridis'色彩映射表，可以根据需要更改
        neuron = neuron_spike[k][m, :]
        theta = np.deg2rad(be_list[k])
        non_zero_indices = np.nonzero(neuron)
        neuron = neuron[neuron != 0]
        if  len(neuron) == 0:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/trace/' + 'phi' + f'-{type}-{k}-' + f'{m}.jpg')
            plt.close()
        else:
        # 使用这些位置来过滤vector2
            theta = theta[non_zero_indices]
            norm = mcolors.Normalize(vmin=0, vmax=max(neuron))
            # 创建极坐标图
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            # 将角度数据映射到环上，绘制211个point，每个point的颜色不同
            for i in range(np.size(neuron, 0)):  # 将角度转换为弧度
                radii = 1  # 小块的半径可以根据需要设置
                color = cmap(norm(neuron[i]))  # 根据发放值获取颜色
                ax.plot(theta[i], radii, marker='o', markersize=5, color=color)
            ax.set_rticks([np.pi / 2])
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/trace/' + 'phi' + f'-{type}-{k}-' + f'{m}.jpg')
            plt.close()

#%% shuffled data comparing

freq = 30.9
tau = 5
dy_list_shuffled = []
#这里把数据给破坏了
for k in range(len(neuron_spike)):
    Spike_train = neuron_spike[k].copy()
    dy_distance = []
    num_features = neuron_spike[k].shape[0]
    dy_d = np.zeros((num_features, num_features))

    np.random.shuffle(Spike_train)
    for m in range(num_features):
        for n in range(num_features):
            dy_d[m,n] = Strength_computer(Spike_train, m, n, tau)
    dy_d[np.isnan(dy_d)] = 0
    dy_list_shuffled.append(dy_d)

#%%
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_famr2_age10_shuffled.pkl', 'rb') as file:
    dy_list_shuffled = pickle.load(file)
#%%
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_famr2_age10_shuffled.pkl', 'wb') as file:
    pickle.dump(dy_list_shuffled, file)
#%%
# def pre_process(data):
#     dy_r = data
#     threshold = (np.max(dy_r)-np.min(dy_r))*0.2+np.min(dy_r)
#     t_p = np.ravel(dy_r.copy())
#     t_p[(t_p <= threshold)] = 0
#     t_p = t_p[t_p != 0]
#     return t_p
#
# import scipy.stats as stats
# for i in range(len(dy_list_shuffled)):
#     t_p = pre_process(dy_list[i])
#     t_s = pre_process(dy_list_shuffled[i])
#
#     fig1 = plt.figure(figsize=(10,5))
#     ax1 = plt.subplot(121)
#     sns.histplot(t_p, element='poly', fill=True, label="Original data", color='skyblue',ax=ax1)
#     ax2 = plt.subplot(122)
#     sns.histplot(t_s, element='poly', fill=True, label="Shuffled data", color='coral',ax=ax2)
#
#     if gene_list[i] == 119:
#         type = 'wt'
#     else:
#         type = 'AD'
#     fig1.suptitle("Weight distribution" + f'-{type}-' + f'{i}')
#     t, p = stats.ttest_ind(t_p, t_s)
#     plt.text(0.25, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
#     plt.xlabel("weight of dynamic connection")
#     plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/graph/' + 'weight' + f'-{type}-' + f'{i}.jpg')
#     plt.close()
#%%
import networkx as nx
import scipy.stats as stats
def build_graph(g,label):
    dy_r = g.copy()
    t_p_G = dy_r
    threshold = (np.max(dy_r)-np.min(dy_r))*0.2+np.min(dy_r)
    if label == 'weak':
        t_p_G[(t_p_G >= threshold)] = 0
    if label == 'strong':
        t_p_G[(t_p_G <= threshold)] = 0
    G = nx.Graph(t_p_G)
    degrees = dict(G.degree())
    labels = list(degrees.keys())
    degree_values_p = degrees.values()
    return degree_values_p

def draw_pic(ax,degree,legend,color):
    degree = np.array(list(degree))
    sns.histplot(degree, bins=15, alpha=0.5, color=color, label=legend,kde=True,ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Degree")
    ax.legend()
    return ax

for i in range(len(dy_list_shuffled)):
    degree_values_p1 = build_graph(dy_list[i],'weak')
    degree_values_p2 = build_graph(dy_list_shuffled[i],'weak')
    fig1 = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(121)
    ax1 = draw_pic(ax1,degree_values_p1,'Original data','skyblue')
    ax2 = plt.subplot(122)
    ax2 = draw_pic(ax2, degree_values_p2,'shuffled data','coral')
    if gene_list[i] == 119:
        type = 'wt'
    else:
        type = 'AD'
    fig1.suptitle("Degree Distribution of original data (weak)" + f'-{type}-' + f'{i}')
    #t, p = stats.ttest_ind(dy_list[i], dy_list_shuffled[i])
    t, p = stats.ttest_ind(list(degree_values_p1), list(degree_values_p2))
    plt.text(0.15, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/graph/' + 'weak connection' + f'-{type}-' + f'{i}.jpg')
    plt.close()

    degree_values_p1 = build_graph(dy_list[i], 'strong')
    degree_values_p2 = build_graph(dy_list_shuffled[i], 'strong')
    fig2 = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)
    ax1 = draw_pic(ax1, degree_values_p1, 'Original data', 'skyblue')
    ax2 = plt.subplot(122)
    ax2 = draw_pic(ax2, degree_values_p2, 'shuffled data', 'coral')
    if gene_list[i] == 119:
        type = 'wt'
    else:
        type = 'AD'
    fig2.suptitle("Degree Distribution of original data (strong)" + f'-{type}-' + f'{i}')
    #t, p = stats.ttest_ind(dy_list[i], dy_list_shuffled[i])
    t, p = stats.ttest_ind(list(degree_values_p1), list(degree_values_p2))
    plt.text(0.15, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/graph/' + 'strong connection' + f'-{type}-' + f'{i}.jpg')
    plt.close()
