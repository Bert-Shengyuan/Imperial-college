#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import seaborn as sns

import joblib as jl
from cebra.datasets import hippocampus
from cebra import CEBRA


from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import scipy.spatial.distance
from sklearn.metrics.cluster import adjusted_mutual_info_score

import numpy as np
from sklearn.manifold import MDS
import math

import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import mat73

#%% dataloader
neu_data  = scipy.io.loadmat('/Users/sonmjack/Downloads/Shuhan_new/neural_data_correcttrials.mat')
be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/Shuhan_new/positionx_data_correcttrials.mat')

neurons=neu_data['neural_data_correct']
#position_X
be_x = be_data['position_X'].reshape(-1,1)
be_label  = np.zeros((142712,2))

# %%基于position加trigger
for i in range(len(be_label)):
    if i == 142711:
        be_label[i, 0] = 2
        be_label[i, 1] = i
        break
    else:
        if be_x[i] > 4.5:

            be_label[i,0] = 1 #右转
            if be_x[i-1] <= 4.5: #trail开始
                be_label[i,1]  = i
            elif be_x[i+1] <= 4.5:
                    be_label[i, 1] = i #trail结束
        elif  be_x[i] <= -4.5:

            be_label[i,0] = 2 #左转

            if be_x[i-1] >= -4.5: #trail开始
                be_label[i,1]  = i
            elif be_x[i+1] >= -4.5:
                    be_label[i, 1] = i #trail结束
        else:
            be_label[i,0] = 0

be_trigger = be_label[be_label[:,1]!=0]
path = '/Users/sonmjack/Downloads/data_lab/shuhan_trigger.npy'
np.save(path,be_trigger)








#%% 决策后汇总
muti_list_after = []
neuron_after = np.zeros((211, 1))
be_after = np.array([])
#%%
for i in range(0, len(be_trigger), 2):
    if i == 210:
        break
    # 不同分类标准，其他代码不变，只需要改trigger 的索引
    neu_data = neurons[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), :]
    be_x_neu = be_x[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), 0]
    neuron_after = np.concatenate((neuron_after, neu_data.T),axis=1)
    be_after = np.concatenate((be_after, be_x_neu))
    # 创建一个空矩阵来存储互信息
    num_features = neu_data.shape[1]
    mutinfo_d = np.zeros((num_features, num_features))

    #计算每两line数据之间的互信息,计算量很大
    # for m in range(num_features):
    #     for n in range(m + 1, num_features):
    #         mi = mutual_info_regression(neu_data[:,m].reshape(-1, 1), np.ravel(neu_data[:,n]))
    #         mutinfo_d[m, n] = mi
    #         mutinfo_d[n, m] = mi  # 因为互信息是对称的，可以减少计算量
    #
    # muti_list_after.append(mutinfo_d)
neuron_after = neuron_after[:, 1:]

#%%
#after_tensor = np.array(muti_list_after)
#%%
plt.figure(figsize=(10, 5))
sns.heatmap(neuron_after)
plt.title("Neural activity of MS after decision")
plt.show()
#%%
plt.figure(figsize=(10, 5))
time_points = np.arange(len(be_after))
sns.lineplot(x=time_points, y=be_after)
plt.title("behaviour activity of MS after decision")
plt.show()
#%%
after_tensor = np.load('/Users/sonmjack/Downloads/Shuhan_new/after_tensor.npy')
average_tensor_after = np.mean(after_tensor, axis=0)
#%%
plt.figure(figsize=(13, 10))
sns.heatmap(average_tensor_after)
plt.title("mutual information of MS response (After decision)")
plt.show()
#%%
path = '/Users/sonmjack/Downloads/Shuhan_new/after_tensor.npy'
np.save(path,after_tensor)







#%% 决策前汇总
muti_list_pre = []
neuron_pre = np.zeros((211, 1))
be_pre = np.array([])
for i in range(1, len(be_trigger), 2):
    if i == 209:
        break
    # 不同分类标准，其他代码不变，只需要改trigger 的索引
    neu_data = neurons[int(be_trigger[i, 1]) + 1:int(be_trigger[(i + 1), 1])-1, :]
    be_x_neu = be_x[int(be_trigger[i, 1]) + 1:int(be_trigger[(i + 1), 1])-1, 0]
    neuron_pre = np.concatenate((neuron_pre,neu_data.T),axis=1)
    be_pre = np.concatenate((be_pre, be_x_neu))
    # 创建一个空矩阵来存储互信息
    num_features = neu_data.shape[1]
    mutinfo_d = np.zeros((num_features, num_features))

    # # 计算每两line数据之间的互信息,计算量很大
    # for m in range(num_features):
    #     for n in range(m + 1, num_features):
    #         mi = mutual_info_regression(neu_data[:,m].reshape(-1, 1), np.ravel(neu_data[:,n]))
    #         mutinfo_d[m, n] = mi
    #         mutinfo_d[n, m] = mi  # 因为互信息是对称的，可以减少计算量
    #
    # muti_list_pre.append(mutinfo_d)
neuron_pre = neuron_pre[:, 1:]








#%%
#pre_tensor = np.array(muti_list_pre)
#%%
plt.figure(figsize=(10, 5))
sns.heatmap(neuron_pre)
plt.title("Neural activity of MS before decision")
plt.show()
#%%
plt.figure(figsize=(10, 5))
time_points = np.arange(len(be_pre))
sns.lineplot(x=time_points, y=be_pre)
plt.title("behaviour activity of MS before decision")
plt.show()

be_max = max(be_pre)
#%%
pre_tensor = np.load('/Users/sonmjack/Downloads/Shuhan_new/pre_tensor.npy')
average_tensor_pre = np.mean(pre_tensor, axis=0)
#%%
plt.figure(figsize=(13, 10))
sns.heatmap(average_tensor_pre)
plt.title("mutual information of MS response (Before decision)")
plt.show()
#%%
path = '/Users/sonmjack/Downloads/Shuhan_new/pre_tensor.npy'
np.save(path,pre_tensor)




#%% 右决策后汇总
muti_list_right = []
neuron_right = np.zeros((211, 1))
be_right_x = np.array([])
be_right_y = np.array([])
# 替换成真实数据 position_y
be_y = np.random.uniform(-50, 50, 142712).reshape(142712,1)
#%%
for i in range(0, len(be_trigger), 2):
    if i == 210:
        break
    elif be_trigger[i, 0] == 1:
    # 不同分类标准，其他代码不变，只需要改trigger 的索引
        neu_data = neurons[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), :]
        be_x_neu = be_x[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), 0]
        be_y_neu = be_y[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), 0]
        neuron_right = np.concatenate((neuron_right, neu_data.T),axis=1)
        be_right_x = np.concatenate((be_right_x, be_x_neu))
        be_right_y = np.concatenate((be_right_y, be_y_neu))
        # 创建一个空矩阵来存储互信息
        num_features = neu_data.shape[1]
        mutinfo_d = np.zeros((num_features, num_features))

    #计算每两line数据之间的互信息,计算量很大
    # for m in range(num_features):
    #     for n in range(m + 1, num_features):
    #         mi = mutual_info_regression(neu_data[:,m].reshape(-1, 1), np.ravel(neu_data[:,n]))
    #         mutinfo_d[m, n] = mi
    #         mutinfo_d[n, m] = mi  # 因为互信息是对称的，可以减少计算量
    #
    # muti_list_right.append(mutinfo_d)
neuron_right =neuron_right[:, 1:]

#%%
# 替换成真实数据 position_y
be_y = np.random.uniform(-50, 50, 142712).reshape(142712,1)


#%% 左决策后汇总
muti_list_left = []
neuron_left = np.zeros((211, 1))
be_left_x = np.array([])
be_left_y = np.array([])
#%%
for i in range(0, len(be_trigger), 2):
    if i == 210:
        break
    elif be_trigger[i, 0] == 2:
    # 不同分类标准，其他代码不变，只需要改trigger 的索引
        neu_data = neurons[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), :]
        be_x_neu = be_x[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), 0]
        be_y_neu = be_y[int(be_trigger[i, 1]):int(be_trigger[(i + 1), 1]), 0]
        neuron_left = np.concatenate((neuron_left, neu_data.T),axis=1)
        be_left_x = np.concatenate((be_left_x, be_x_neu))
        be_left_y = np.concatenate((be_left_y, be_y_neu))
        # 创建一个空矩阵来存储互信息
        num_features = neu_data.shape[1]
        mutinfo_d = np.zeros((num_features, num_features))

    #计算每两line数据之间的互信息,计算量很大
    # for m in range(num_features):
    #     for n in range(m + 1, num_features):
    #         mi = mutual_info_regression(neu_data[:,m].reshape(-1, 1), np.ravel(neu_data[:,n]))
    #         mutinfo_d[m, n] = mi
    #         mutinfo_d[n, m] = mi  # 因为互信息是对称的，可以减少计算量
    #
    # muti_list_right.append(mutinfo_d)
neuron_left =neuron_left[:, 1:]

#%%
color_tensor = np.zeros([neuron_left.shape[0], 1000, 2])

top_n_indices = np.argpartition(neuron_left, -1000, axis=1)[:, -1000:]

for i in range(top_n_indices.shape[0]):
    for j in range(top_n_indices.shape[1]):
        color_tensor[i, j, 0] = be_left_x[top_n_indices[i, j]]
        color_tensor[i, j, 1] = be_left_y[top_n_indices[i, j]]

# 找到每一行最大的6个值对应的列索引
color_left = np.mean(color_tensor, axis=1)



#%%
import scipy.stats as stats
t_p = np.ravel(average_tensor_pre.copy())
t_a = np.ravel(average_tensor_after.copy())
t_all = np.concatenate([t_p,t_a])
threshold = 0.1*max(t_all)
t_p[(t_p <= threshold)] = 0
t_a[(t_a <= threshold)] = 0
t_p = t_p[t_p != 0]
t_a = t_a[t_a != 0]

plt.figure(figsize=(5, 5))
plt.title("mutual information of MS response")
# Plotting distribution
# ax = sns.kdeplot(t_p,bw_adjust=1,fill=True,label="Before_decision",common_norm=True)
# sns.kdeplot(t_a, bw_adjust=1,fill=True,label="After_decision",common_norm=True)
ax = sns.histplot(t_p,  element='poly', fill=True,label="Before_decision",color='skyblue')
sns.histplot(t_a, element='poly', fill=True,label="After decision",color='coral')
# tttest
t,p = stats.ttest_ind(t_p,t_a)
plt.text(0.25, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("mutual information")
plt.legend()
plt.show()

#%%
import scipy.stats as stats
t_p_n = neuron_pre
t_a_n = neuron_after
t_p_n[(t_p_n <= 0.1)] = 0
t_a_n[(t_a_n <= 0.1)] = 0
t_p_n = t_p_n[t_p_n != 0]
t_a_n = t_a_n[t_a_n != 0]

plt.figure(figsize=(5, 5))
plt.title("Neural activity of MS")
#ax = sns.kdeplot(t_p_n,bw_adjust=2,label="Before_decision",fill=True)
#sns.kdeplot(t_a_n, bw_adjust=2,label="After decision",fill=True)
# Plotting histograms
ax = sns.histplot(t_a_n, bins=15, alpha=0.5,label="After decision",color='coral',kde=True)
sns.histplot(t_p_n, bins=15, alpha=0.9,label="Before_decision",color='skyblue',kde=True)

# tttest
t,p = stats.ttest_ind(t_p_n,t_a_n)
plt.text(0.15, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("AP of neuron")
plt.legend()
plt.show()
#%%
max_iterations = 1000  # default is 5000.
output_dimension = 32  # here, we set as a variable for hypothesis testing below.

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
                           hybrid=True)  # hybrid = True

cebra_hybrid_model.fit(neu_data, be_x_neu)
cebra_hybrid = cebra_hybrid_model.transform(neu_data)
length = np.size(cebra_hybrid, 0)


def plot(ax, embedding, label):
    p = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, c=label, cmap='viridis')
    # ax.plot(embedding[:, 0], embedding[:, 1], embedding [:, 2],c='gray', linewidth=0.5)

    ax.grid(False)
    ax.set_box_aspect([np.ptp(i) for i in embedding.T])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    return p, ax


# %%
# matplotlib notebook
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='3d')
p, ax = plot(ax, cebra_hybrid, be_x_neu)

# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax.set_title('CEBRA-Hybrid')
fig.colorbar(p)
plt.show()
#%% MDS对神经元标色
# color_index = np.zeros([2,np.size(be_pre)])
# neuron_data_dict = {}

# for i in range(len(be_pre)):
#     R_n = np.argmax(neuron_pre[:,i])
#     color_index[0,i] = R_n
#     color_index[1,i] = be_pre[i]

#     if neuron_index in neuron_data_dict:
#         # 如果神经元索引已经在字典中，追加数据
#         neuron_data_dict[neuron_index].append(be_X)
#     else:
#         # 如果神经元索引不在字典中，创建一个新的键值对
#         neuron_data_dict[neuron_index] = [be_X]
#
# # 计算每个神经元的平均值
# neuron_averages = {}
# for neuron_index, data_list in neuron_data_dict.items():
#     neuron_averages[neuron_index] = np.mean(data_list)
#
# num_neurons = 211
# averages_matrix = np.zeros((num_neurons, 1))
#
# # 将neuron_averages中的数据填充到矩阵中
# for neuron_index, average in neuron_averages.items():
#     row_index = neuron_index - 1  # 字典的索引从1开始，所以需要减1来匹配行索引
#     averages_matrix[int(row_index), 0] = average

color_tensor = np.zeros([neuron_pre.shape[0],10])

top_n_indices = np.argpartition(neuron_pre, -10, axis=1)[:, -10:]

for i in range(top_n_indices.shape[0]):
    for j in range(top_n_indices.shape[1]):
        color_tensor[i,j] = be_pre[top_n_indices[i,j]]

# 找到每一行最大的6个值对应的列索引
color_means1 = np.mean(color_tensor, axis=1)

#%% Muti dimension scale 降维后的低维表征

    # 使用numpy.save保存矩阵为文本文件
    # 使用MDS进行降维
mds = MDS(n_components=3, random_state=42)
mds_result = mds.fit_transform(average_tensor_pre)

# 可视化降维结果
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2],c=color_means1)

ax.set_title('Modified locally linear embedding of neuron (before decision)')
fig.colorbar(p)
plt.show()

#%%
color_tensor = np.zeros([neuron_after.shape[0],10])

top_n_indices = np.argpartition(neuron_after, -10, axis=1)[:, -10:]

for i in range(top_n_indices.shape[0]):
    for j in range(top_n_indices.shape[1]):
        color_tensor[i,j] = be_after[top_n_indices[i,j]]

# 找到每一行最大的6个值对应的列索引
color_means2 = np.mean(color_tensor, axis=1)

mds = MDS(n_components=3, random_state=42)
mds_result = mds.fit_transform(average_tensor_after)

# 可视化降维结果
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2],c=color_means2)

ax.set_title('Modified locally linear embedding of neuron (after decision)')
fig.colorbar(p)
plt.show()








#%%
import networkx as nx
t_p_G = average_tensor_pre.copy()
t_p_G[(t_p_G >= threshold)] = 0
G1 = nx.Graph(t_p_G)
degrees = dict(G1.degree())
labels = list(degrees.keys())
degree_values_p = degrees.values()
#%%
plt.figure(figsize=(5, 5))
ax = sns.histplot(list(degree_values_p), bins=15, alpha=0.5,label="Before decision",color='skyblue',kde=True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Degree Distribution before decision (Weak connection)")
plt.xlabel("Degree")
plt.legend()
plt.show()
#%%
import networkx as nx
t_a_G = average_tensor_after.copy()
t_a_G[(t_a_G >= threshold)] = 0
G2 = nx.Graph(t_a_G)
degrees = dict(G2.degree())
degree_values_a = degrees.values()
#%%
plt.figure(figsize=(5, 5))
ax = sns.histplot(list(degree_values_a), bins=15, alpha=0.5,label="After decision",color='coral',kde=True)

t,p = stats.ttest_ind(list(degree_values_a),list(degree_values_p))
plt.text(0.45, 0.7, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Degree Distribution after decision (Weak connection)")
plt.xlabel("Degree")
plt.legend()
plt.show()
#%%
import networkx as nx
t_p_G = average_tensor_pre.copy()
t_p_G[(t_p_G <= threshold)] = 0
G1 = nx.Graph(t_p_G)
degrees = dict(G1.degree())
labels = list(degrees.keys())
degree_values_p = degrees.values()
#%%
plt.figure(figsize=(5, 5))
ax = sns.histplot(list(degree_values_p), bins=15, alpha=0.5,label="Before decision",color='skyblue',kde=True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Degree Distribution before decision ")
plt.xlabel("Degree")
plt.legend()
plt.show()

#%%
import networkx as nx
t_a_G = average_tensor_after.copy()
t_a_G[(t_a_G <= threshold)] = 0
G2 = nx.Graph(t_a_G)
degrees = dict(G2.degree())
degree_values_a = degrees.values()
#%%
plt.figure(figsize=(5, 5))
ax = sns.histplot(list(degree_values_a), bins=15, alpha=0.5,label="After decision",color='coral',kde=True)

t,p = stats.ttest_ind(list(degree_values_a),list(degree_values_p))
plt.text(0.45, 0.7, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Degree Distribution after decision")
plt.xlabel("Degree")
plt.legend()
plt.show()


#%%  群落检测然后画出来

# remove low-degree nodes
# low_degree = [n for n, d in G1.degree() if d < 50]
# G1.remove_nodes_from(low_degree)
#%%
# largest connected component
components = nx.connected_components(G1)
largest_component = max(components, key=len)
H1 = G1.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H1, k=100, endpoints=True)

# compute community structure
lpc = nx.community.label_propagation_communities(H1)
community_index1 = {n: i for i, com in enumerate(lpc) for n in com}

#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H1, k=0.15, seed=4572321)
node_color = [community_index1[n] for n in H1]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    H1,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Mutual information association network (Before decision)", font)
# Change font color for legend
font["color"] = "r"

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweenness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readability
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.show()

#%%
# largest connected component
components = nx.connected_components(G2)
largest_component = max(components, key=len)
H2 = G2.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H2, k=100, endpoints=True)

# compute community structure
lpc = nx.community.label_propagation_communities(H2)
community_index2 = {n: i for i, com in enumerate(lpc) for n in com}

#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H2, k=0.15, seed=4572321)
node_color = [community_index2[n] for n in H2]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    H2,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Mutual information association network (After decision)", font)
# Change font color for legend
font["color"] = "r"

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweenness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readability
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.show()