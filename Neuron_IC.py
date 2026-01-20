#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib as jl

from cebra import CEBRA
from matplotlib.collections import LineCollection

from sklearn.metrics import normalized_mutual_info_score


import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import math

#dataloader

import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import mat73

mat_data  = mat73.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1_timeseries.mat')
be_data  = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1_trackdata.mat')


#%%
import h5py
type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1_timeseries.mat')
gene = type_array ['genotype'][:,:].T

be_f = mat_data['expname'][:]
be_label  = np.zeros((9060,1))

for i in range(len(be_label)):
    if i == 9060:
        break
    if i ==0:
        be_label[i] = 1
    else:
        if be_f[i] != be_f[i+1]:
            be_label[i] = i+1
        else:
            be_label[i] = 0

be_trigger = be_label[be_label[:]!=0]
path = '/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger.npy'
np.save(path,be_trigger)

#%% 算每一个trail的数据互信息
for i in range(len(be_trigger)):
    cell_array = mat_data['spikes'][int(be_trigger[i]):int(be_trigger[i+1])]
    num_rows = len(cell_array)
    num_column = len(cell_array[0][0])
    neurons = np.zeros((num_rows, num_column))

    # 长度不一致需要截断处理然后再看

    for j in range(num_rows):
        neurons[j, :] = (cell_array[j][0]*10).astype(int)

    num_features = neurons.shape[0]
    mutinfo_d = np.zeros((num_features, num_features))

    # 计算每两line数据之间的互信息
    for m in range(num_features):
        for n in range(m + 1, num_features):
            mi = normalized_mutual_info_score(neurons[m, :], neurons[n, :])
            jaccard_distance = 1 - mi
            mutinfo_d[m, n] = jaccard_distance
            mutinfo_d[n, m] = jaccard_distance  # 因为互信息是对称的，可以减少计算量

import pickle
with open('matrix_list.pkl', 'wb') as file:
    pickle.dump(matrix_list, file)
#%% Muti dimension scale 降维后的低维表征

    # 使用numpy.save保存矩阵为文本文件
    # 使用MDS进行降维
    mds = MDS(n_components=3, random_state=42)
    mds_result = mds.fit_transform(mutinfo_d)

    # 可视化降维结果
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2])
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)

    if gene[int(be_trigger[i]),0] == 119:
        type = 'wt'
    else:
        type = 'AD'

    ax.set_title(type + '_Distance 3D Visualization')
    plt.savefig('/Users/sonmjack/Downloads/data_lab/'+type+f'_{i}_'+'.jpg')
    file_path = '/Users/sonmjack/Downloads/data_lab/'+type+ f'_{i}_' + '.npy'
    np.save(file_path, mutinfo_d)
    plt.show()

#%%  利用信息距离去建立拓扑图

    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    # 创建一个 211 x 211 的随机矩阵作为邻接矩阵,作为null model
    adj_matrix = np.random.randint(0, 2, size=(211, 211))
    #  自己真实的图
    mutinfo_d[(mutinfo_d <= 0.12) & (mutinfo_d >= -0.20)] = 0
    # 将邻接矩阵转换为 NetworkX 图
    #EC  是numpy的矩阵
    G = nx.Graph(EC)

    # 计算图中所有节点的度数
    degree = dict(G.degree())

    # 将度数按照从大到小排序
    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)

    # 打印排序后的节点度数
    print(sorted_degree)

    # 将节点度数作为横坐标，节点数量作为纵坐标画出度分布图
    # degrees = np.array([x[1] for x in sorted_degree], dtype=int)
    #
    # count = [degrees.tolist().count(x) for x in set(degrees)]
    # test = set(degrees)
    # test2 = count


    #degree_values = sorted(set(degree.values()))
    degree_values = degree.values()
    degree_hist = [list(degree.values()).count(x) / float(nx.number_of_nodes(G)) for x in degree_values]



    ax = sns.kdeplot(data=degree_values, label="premature infant",shade=True,clip=(0, None),bw_adjust=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.bar(set(degrees), count, width=0.8)
    #plt.bar(set(degrees), count, width=0.8)

    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution')
    plt.show()

#%%  central degree 分析， community






#假设mutual_information_matrix包含互信息矩阵






# 指定保存文件的路径和文件名


#%%


#%%

plt.figure(figsize=(10, 10))
plt.title("Schematic histogram of the distribution of data on the independent variable magnesium intake")
plt.subplot(211)
MI1 = df_under['MI(Magnesium intake) under 350']

# Plotting histograms
sns.histplot(MI1, bins=15, alpha=0.6,kde=True)
# Fitting a normal distribution curve using sns built-in parameters
plt.subplot(212)
MI2 = df_above['MI(Magnesium intake) over 350']
sns.histplot(MI2, bins=15, alpha=0.6,kde=True)

#%% time coding dedimension

#%%

# 创建散点图，使用指定颜色
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D

# 创建一个示例数据集，你需要替换成你的实际数据
# 这里假设data是一个1080x48的NumPy数组，包含0和1
neu_s= neu_data1.copy()
np.random.shuffle(neu_s)

pca = PCA (n_components=3)
#time
reduced_data = pca.fit_transform(neu_s)
#space

fig = plt.figure(figsize=(20,20), dpi=20)

# 创建一个三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
be_s_x= be_x_neu.copy()
np.random.shuffle(be_s_x)


# 绘制数据点
# ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=10,c=be_array_pi,cmap ='viridis')
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c = be_s_x, cmap='viridis',s=1)
#ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=10)
# 设置坐标轴标签
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
# ax.set_xticks(np.arange(-0.004, 0.006, 0.004))
# ax.set_yticks(np.arange(-0.004, 0.006, 0.002))
# ax.set_zticks(np.arange(-0.004, 0.006, 0.002))
# 显示图形
plt.show()
plt.savefig('/Users/sonmjack/Downloads/data_lab/PCAspace.pdf')

#%%
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


# 生成一个随机的183x183的矩阵，实际应用中你需要用你的数据替换这里的随机数据
mutinfo1 = np.load('/Users/sonmjack/Downloads/data_lab/mutual_infor_all.npy')
mutinfo2 = np.load('/Users/sonmjack/Downloads/data_lab/mutual_infor_all2.npy')



#%%
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


# 生成一个随机的183x183的矩阵，实际应用中你需要用你的数据替换这里的随机数据
mutinfo1 = np.load('/Users/sonmjack/Downloads/data_lab/mutual_infor_all.npy')
#mutinfo2 = np.load('/Users/sonmjack/Downloads/data_lab/mutual_infor_all2.npy')

# 使用MDS进行降维
mds = MDS(n_components=3, random_state=42)
mds_result = mds.fit_transform(mutinfo1)

# 可视化降维结果
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2])

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('MDS 3D Visualization')

plt.savefig('/Users/sonmjack/Downloads/data_lab/MDSspace1.pdf')

#%%


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建一个随机相关性矩阵（这里用随机数代替实际相关性值）
# 在你的应用程序中，你需要提供实际的相关性矩阵
correlation_matrix = 1-mutinfo

# 创建一个无向图
G = nx.Graph()

# 添加节点
for i in range(correlation_matrix.shape[0]):
    G.add_node(i)

# 添加边（根据相关性阈值）
threshold = 0.5 # 可根据需要调整相关性阈值
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if correlation_matrix[i, j] > threshold:
            G.add_edge(i, j)

# 绘制网络
pos = nx.spring_layout(G)  # 选择布局算法
nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=0.1)
plt.title("Correlation Network")
plt.savefig('/Users/sonmjack/Downloads/data_lab/graph1.pdf')



#%%

import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt

# 计算全局信号（按列求和）
global_signal1 = np.sum(neurons, axis=0)
nperseg = 1028
ftf1, ftt1, ftZ1 = signal.stft(global_signal1, nperseg=nperseg, fs=30.9, noverlap=nperseg-1)


#%%

plt.figure()
plt.pcolormesh(ftt1, ftf1, np.abs(ftZ1), cmap='hot_r')
plt.title('Short-Time Fourier-Transform')
plt.ylabel('Frequency (Hz)')

plt.savefig('/Users/sonmjack/Downloads/data_lab/Fre1.png')

#%%

import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt

# 计算全局信号（按列求和）
global_signal1 = np.sum(neurons, axis=0)
nperseg = 1028
ftf1, ftt1, ftZ1 = signal.stft(global_signal1, nperseg=nperseg, fs=30.9, noverlap=nperseg-1)
amp = 2 * np.sqrt(2)

#%%
plt.figure()
plt.pcolormesh(ftt1, ftf1, np.abs(ftZ1), cmap='hot_r',vmin=0, vmax=amp)
plt.title('Short-Time Fourier-Transform')
plt.ylabel('Frequency (Hz)')

plt.savefig('/Users/sonmjack/Downloads/data_lab/Fre2.pdf')

#%%


max_iterations = 1000#default is 5000.
output_dimension = 3 #here, we set as a variable for hypothesis testing below.

cebra_hybrid_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid = True)

#%%
cebra_hybrid_model.fit(neurons, be_array_pi)
cebra_hybrid = cebra_hybrid_model.transform(neurons)

#%%
def plot_hippocampus(ax, embedding, label, gray = False, idx_order = (0,1,2)):


    p = ax.scatter(embedding [:, 0], embedding [:, 1], embedding [:, 2],s=5,c=be_array_pi,cmap ='viridis')
    #ax.plot(embedding[:, 0], embedding[:, 1], embedding [:, 2],c='gray', linewidth=0.5)

    ax.grid(False)
    ax.set_box_aspect([np.ptp(i) for i in embedding.T])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    return p,ax

#%%
#matplotlib notebook
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,30))

# ax1 = plt.subplot(141, projection='3d')
# ax2 = plt.subplot(142, projection='3d')
# ax3 = plt.subplot(143, projection='3d')
ax4 = plt.subplot(111, projection='3d')

# ax1=plot_hippocampus(ax1, cebra_posdir3, hippocampus_pos.continuous_index)
# ax2=plot_hippocampus(ax2, cebra_posdir_shuffled3, hippocampus_pos.continuous_index)
# ax3=plot_hippocampus(ax3, cebra_time3, hippocampus_pos.continuous_index)
p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D)

# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax4.set_title('CEBRA-Hybrid')
fig.colorbar(p)
plt.savefig('/Users/sonmjack/Downloads/data_lab/cybrd.jpg')

#%% 原始图像
# fig = plt.figure(figsize=(9,3), dpi=600)
# plt.subplots_adjust(wspace = 0.3)
# ax = plt.subplot(121)
# data = neurons.T
# ax.imshow(neurons.T[:,0:1000], aspect = 'auto', cmap = 'gray_r')
# plt.ylabel('Neuron #')
# plt.xlabel('Time [s]')
#
# ax2 = plt.subplot(122)
#
#
# ax2.scatter(np.arange(29680), be_array_x, c = 'gray', s=1)
#
# # 这里可以再尝试下直接输入matrix
# plt.ylabel('Position [m]')
# plt.xlabel('Time [s]')
# plt.show()

#%%
# import numpy as np
# from scipy.stats import entropy, gaussian_kde
#
# # 生成模拟数据，这里假设有183个数据点，每个数据点有2000维特征,计算量太大了
# data = np.random.rand(183, 2000)
#
# # 初始化联合熵列表
# joint_entropies = []
#
# # 计算每对数据点之间的联合熵
# for i in range(data.shape[0]):
#     for j in range(i + 1, data.shape[0]):
#         # 使用核密度估计估算联合分布
#         kde = gaussian_kde(np.vstack((data[i], data[j])))
#
#         # 计算联合熵
#         joint_entropy = entropy(kde(np.vstack((data[i], data[j]))))
#         joint_entropies.append(joint_entropy)

#%%
cell_array = mat_data['spikes'][8803:9060]

be_array_x = be_data['x'][42,0]
be_array_y = be_data['y'][42,0]
be_array_pi = be_data['phi'][42,0]

cell_array = mat_data['spikes'][0:183]
be_array_x = be_data['x'][0,0]
be_array_y = be_data['y'][0,0]
be_array_pi = be_data['phi'][0,0]