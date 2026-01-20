#%%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
import math
from sklearn.metrics import normalized_mutual_info_score
import scipy.io
import numpy as np

import mat73

#%% dataloader
mat_data  = mat73.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')
#%%
# #%%
spike_time_all = mat_data['fam1_df_f']

# age = mat_data['ageMOS']
# mask = mat_data['masks'][:,:,:]
# be_f = mat_data['expname'][:]
# ex_index = mat_data['expname'][:]
# #spike_sum = mat_data['nov_spikes']
spike_sum = mat_data['fam1_spikes']

#%%
be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
import h5py
type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')
gene = type_array ['genotype'][:,:].T
mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum = be_data['fam1_phi']

# #%% exist conflict of index, so just use part of data
# for i in range(gene.shape[0]):#gene.shape[0] 969,3236
#     if i == gene.shape[0]-1:
#         mat_label[i, 0] = i
#         mat_label[i, 1] = gene[i, 0]
#         mat_label[i, 2] = age[i]
#         mat_label[i, 3] = len(spike_sum[i][0])
#         break
#     if i ==0:
# #         mat_label[i,0] = 0.1
# #         mat_label[i, 1] = gene[i, 0]
# #         mat_label[i, 2] = age[i]
# #         mat_label[i, 3] = len(spike_sum[i][0])
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
# path = '/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy'
# np.save(path,mat_trigger)
#%%
mat_trigger1 = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')

#%%  age older  fam1
neuron_spike = []
neuron_time_list = []
be_list = []
gene_list = []
age_list = []
mutual_list = []
for i in range(10,46,2):#0, len(mat_trigger), 2
    neuron_times = []
    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        cell_df = spike_time_all[int(mat_trigger[i, 0]):int(mat_trigger[i + 1, 0])]
        num_rows = len(cell_df)
        for j in range(num_rows):
            window_size = 50
            # 计算移动平均值
            smooth_data = np.convolve(cell_df[j][0], np.ones(window_size) / window_size, mode='valid')
            neuron_times.append(smooth_data)
        neuron_time_list.append(neuron_times)

        cell_array = spike_sum[int(mat_trigger[i,0]):int(mat_trigger[i + 1,0])]
        num_rows = len(cell_array)
        num_column = len(cell_array[0][0])
        neurons = np.zeros((num_rows, num_column))

        # 长度不一致需要截断处理然后再看
        for j in range(num_rows):
            neurons[j, :] = (cell_array[j][0] * 10).astype(int)
        neuron_spike.append(neurons)
        # 索引原始的be_list
        # be_list.append(be_phi_sum[int(i/2),0])
        # gene_list.append(mat_trigger[i,1])
        # age_list.append(mat_trigger[i,2])

        # num_features = num_rows
        # mutinfo_d = np.zeros((num_features, num_features))
        # #计算每两line数据之间的互信息
        # for m in range(num_features):
        #     for n in range(m + 1, num_features):
        #         a = neurons[m, :][neurons[m, :] != 0]
        #         b = neurons[n, :][neurons[n, :] != 0]
        #         max_size = max(len(a), len(b))
        #         a= np.pad(a, (0, max_size - len(a)), mode='constant')
        #         b= np.pad(b, (0, max_size - len(b)), mode='constant')
        #         mi = normalized_mutual_info_score(a, b)
        #
        #         mutinfo_d[m, n] = mi
        #         mutinfo_d[n, m] = mi
        # mutual_list.append(mutinfo_d)

#%%
plt.plot(neuron_spike[10][1,:])
plt.show()
#%% save data
import pickle
with open('/Users/sonmjack/Downloads/simon_paper/fam1_df_f_list_age2.pkl', 'wb') as file:
    pickle.dump(neuron_time_list, file)
# with open('/Users/sonmjack/Downloads/simon_paper/fam1_neuron_list_age2.pkl', 'wb') as file:
#     pickle.dump(neuron_spike, file)
#
# with open('/Users/sonmjack/Downloads/simon_paper/fam1_ex_index_age2.pkl', 'wb') as file:
#     pickle.dump(ex_index, file)
#
# with open('/Users/sonmjack/Downloads/simon_paper/fam1_mutual_list_age2.pkl', 'wb') as file:
#     pickle.dump(mutual_list, file)


#%%
import pickle
with open('/Users/sonmjack/Downloads/simon_paper/fam1_neuron_list_age2.pkl', 'rb') as file:
    neuron_spike = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/fam1_ex_index_age2.pkl', 'rb') as file:
    ex_index = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/fam1_mutual_list_age2.pkl', 'rb') as file:
    mutual_list = pickle.load(file)


neuron_index = []

for i in range(len(neuron_spike)):
    non_zero_indices_per_row = []
    for row in neuron_spike[i]:
         # 找到每行中不为0的元素的列索引
         non_zero_indices = np.where(row != 0)[0]
         # 添加到列表中
         non_zero_indices_per_row.append(list(non_zero_indices))
    neuron_index.append(non_zero_indices_per_row)

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
tau = 1
dy_list = []
#这里把数据给破坏了
for k in range(len(neuron_index)):#len(neuron_index)
    Spike_train = neuron_index[k]
    num_features = neuron_spike[k].shape[0]
    dy_d = np.zeros((num_features, num_features))
    for m in range(num_features):
        for n in range(num_features):
            dy_d[m,n] = Strength_computer(Spike_train, m, n, tau)
    #dy_d[np.isnan(dy_d)] = 0
    print('Finished'+f'{k}')
    dy_list.append(dy_d)
#%%
with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_fam_age2_version2.pkl', 'wb') as file:
    pickle.dump(dy_list, file)

#%%
with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_fam_age2_version2.pkl', 'rb') as file:
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
for k in range(0,1):
    color_means1 = embeding_color(neuron_spike[k], be_list[k], 10)
    mds = MDS(n_components=3, random_state=42)
    mds_result = mds.fit_transform(dy_list[k])
    if gene_list[k] == 119:
        type = 'wt'
    else:
        type = 'AD'
    plt.figure(figsize=(13, 10))
    sns.heatmap(dy_list[k],vmin=0, vmax=1)
    plt.title("Strength matrix "+f'-{type}-{k}')
    plt.xlabel("Neurons")
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/' + 'dynamic_version2' + f'-{type}-' + f'{k}.jpg')
    # 可视化降维结果
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2],c=color_means1)

    ax.set_title('Modified locally linear embedding of dynamic distence'+f'-{type}-{k}')
    fig.colorbar(p)
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/'+'dynamic distence_version2'+f'-{type}-'+f'{k}.jpg')
#%%
import networkx as nx
def Connector(Q):
    D = nx.to_networkx_graph(Q, create_using=nx.DiGraph())
    Isolate_list = list(nx.isolates(D))
    if len(Isolate_list) > 0:
        for i in Isolate_list:
            if i == 0:
                Q[i + 1, i] = 0.0001
            else:
                Q[i - 1, i] = 0.0001
    del D
    return Q
Q_list = []
for i in range(len(dy_list)):
    A = dy_list[i]
    N = A.shape[0]

    np.fill_diagonal(A, 0)
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    k = 10

    # W.sort(reverse=True)
    B1 = np.zeros((N, N))
    for i in range(N):
        W = sorted(A[i, :], reverse=True)
        #     print( W[k])
        B1[i, :] = np.where(A[i, :] > W[k], 1, 0)

    # B=np.multiply(B1,A)
    # print(W[k])
    # print(A[20,1:20])
    # print(B[20,1:20])


    C1 = np.zeros((N, N))
    for i in range(N):
        W = sorted(A[:, i], reverse=True)
        #     print( W[k])
        C1[:, i] = np.where(A[:, i] > W[k], 1, 0)
    # C=np.multiply(C1,A)
    Q1 = B1 + C1
    Q2 = np.where(Q1 > .9, 1, 0)

    Q = np.multiply(Q2, A)
    Q_list.append(Q)


#%%
for k in range(len(Q_list)):
    color_means1 = embeding_color(neuron_spike[k], be_list[k], 10)
    mds = MDS(n_components=3, random_state=42)
    mds_result = mds.fit_transform(Q_list[k])
    if gene_list[k] == 119:
        type = 'wt'
    else:
        type = 'AD'
    plt.figure(figsize=(13, 10))
    sns.heatmap(Q_list[k],vmin=0, vmax=0.05)
    plt.title("Coupled strength matrix "+f'-{type}-{k}')
    plt.xlabel("Neurons")
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/' + 'dynamic' + f'-{type}-' + f'{k}.jpg')
    # 可视化降维结果
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2],c=color_means1)

    ax.set_title('Modified locally linear embedding of dynamic distence'+f'-{type}-{k}')
    fig.colorbar(p)
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/'+'dynamic distence'+f'-{type}-'+f'{k}.jpg')
#%% shuffled data comparing

freq = 30.9
tau = 1
dy_list_shuffled = []
#这里把数据给破坏了
for k in range(len(neuron_spike)):
    Spike_train = neuron_spike[k].copy()
    num_features = neuron_spike[k].shape[0]
    dy_d = np.zeros((num_features, num_features))
    non_zero_indices_per_row = []

    for row in Spike_train:
        # 找到每行中不为0的元素的列索引
        np.random.shuffle(row)
        non_zero_indices = np.where(row != 0)[0]
        # 添加到列表中
        non_zero_indices_per_row.append(list(non_zero_indices))

    for m in range(num_features):
        for n in range(num_features):
            dy_d[m,n] = Strength_computer(non_zero_indices_per_row, m, n, tau)
    print('Finished' + f'{k}')
    dy_list_shuffled.append(dy_d)
#%%
with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_fam_age2_shuffled.pkl', 'wb') as file:
    pickle.dump(dy_list_shuffled, file)
#%%在更小尺度shuffled
with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_fam_age2_shuffled.pkl', 'rb') as file:
    dy_list_shuffled = pickle.load(file)


#%%
def pre_process(data):
    dy_r = data
    threshold = (np.max(dy_r)-np.min(dy_r))*0.2+np.min(dy_r)
    t_p = np.ravel(dy_r.copy())
    t_p[(t_p <= threshold)] = 0
    t_p = t_p[t_p != 0]
    return t_p

import scipy.stats as stats
for i in range(len(dy_list_shuffled)):
    t_p = pre_process(Q_list[i])
    t_s = pre_process(dy_list_shuffled[i])

    fig1 = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(121)
    sns.histplot(t_p, bins=50,fill=True, label="Original data", color='skyblue',ax=ax1)
    ax2 = plt.subplot(122)
    sns.histplot(t_s, bins=50, fill=True, label="Shuffled data", color='coral',ax=ax2)

    if gene_list[i] == 119:
        type = 'wt'
    else:
        type = 'AD'
    fig1.suptitle("Weight distribution" + f'-{type}-' + f'{i}')
    t, p = stats.ttest_ind(t_p, t_s)
    plt.text(0.25, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
    plt.xlabel("weight of dynamic connection")
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/graph/' + 'version2_shuffled compared' + f'-{type}-' + f'{i}.jpg')
    plt.close()
#%%
def build_graph(g,label):
    dy_r = g.copy()
    t_p_G = dy_r
    threshold = (np.max(dy_r)-np.min(dy_r))*0.1+np.min(dy_r)
    if label == 'weak':
        t_p_G[(t_p_G >= threshold)] = 0
    if label == 'strong':
        t_p_G[(t_p_G <= threshold)] = 0
    G = nx.DiGraph(t_p_G)
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
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/graph/' + 'weak connection' + f'-{type}-' + f'{i}.jpg')
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
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/graph/'+'version2_strong connection'+f'-{type}-'+f'{i}.jpg')
    plt.close()

#%%
with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_fam_age2_version2.pkl', 'rb') as file:
    dy_list = pickle.load(file)
def Connector(Q):
    D = nx.to_networkx_graph(Q,create_using=nx.DiGraph())
    Isolate_list=list(nx.isolates(D))
    if len(Isolate_list)>0:
        for i in Isolate_list:
            if i==0:
                Q[i+1,i]=0.0001
            else:
                Q[i-1,i]=0.0001
    del D
    return Q
def build_graph(g,label):
    G = nx.DiGraph(g)
    if nx.is_strongly_connected(G):
        avg_clustering = nx.average_clustering(G)
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        avg_clustering = 0
        avg_path_length = 0
    return avg_clustering, avg_path_length

def pre_process(data):
    dy_r = data
    t_p = np.ravel(dy_r.copy())
    t_p = t_p[t_p != 0]
    #t_p = t_p[t_p <= 0.10]
    return t_p

for index in range(len(dy_list)):
    A  = dy_list[index]
    # KNN for sparse
    import networkx as nx
    from scipy import sparse

    N = A.shape[0]
    np.fill_diagonal(A, 0)
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    k = 10

    # W.sort(reverse=True)
    B1 = np.zeros((N, N))
    for i in range(N):
        W = sorted(A[i, :], reverse=True)
        #     print( W[k])
        B1[i, :] = np.where(A[i, :] > W[k], 1, 0)

    # B=np.multiply(B1,A)
    # print(W[k])
    # print(A[20,1:20])
    # print(B[20,1:20])

    C1 = np.zeros((N, N))
    for i in range(N):
        W = sorted(A[:, i], reverse=True)
        #     print( W[k])
        C1[:, i] = np.where(A[:, i] > W[k], 1, 0)
    # C=np.multiply(C1,A)
    Q1 = B1 + C1
    Q2 = np.where(Q1 > .9, 1, 0)

    Q = A#np.multiply(Q2, A)
    # del A
    del B1
    del C1
    del Q1
    del Q2
    Connector(Q)
    for j in range(Q.shape[0]):
        # 检查该行是否全为零
        if np.all(Q[j] == 0):
            # 如果是全为零，随机选择一个元素，并将其赋值为 0.001
            random_index = np.random.randint(0, Q.shape[1])  # 随机选择一个列索引
            Q[j, random_index] = 0.001
    avg_clustering, avg_path_length = build_graph(Q, 'strong')
    real = pre_process(Q)
    bins = np.linspace(0, 0.1, num=25)
    plt.figure(figsize=(8, 6))
    sns.histplot(real, color='red', kde=True, label='Real data', alpha= 0.5, bins=bins,stat='density')
    plt.axvline(np.mean(real), color='red', linestyle='--', label='Mean of real data')
    plt.legend()
    plt.xlabel("weight of dynamic connection")
    plt.xlim(0, 0.1)
    plt.text(0.55, 0.85, f'average_cluster = {avg_clustering:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.55, 0.65, f'average_length = {avg_path_length:.4f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/' + 'weight_'  + f'{index}.jpg')
    plt.close()

#%%
import cv2
import os

# 设置图片文件夹路径和视频输出路径
image_folder = '/Users/sonmjack/Downloads/age10 result_fam1/mask'
video_name = 'output_video_10.mp4'

# 获取图片文件夹中的所有图片文件名
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

# 读取第一张图片，获取图片尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 使用 OpenCV VideoWriter 创建视频文件
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

# 遍历所有图片并将其写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 关闭视频文件
cv2.destroyAllWindows()
video.release()
