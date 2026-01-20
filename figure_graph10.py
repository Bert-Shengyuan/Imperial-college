#%%
from sklearn.manifold import MDS
import math
from sklearn.metrics import normalized_mutual_info_score
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import mat73
import pickle
import pandas as pd
import scipy.stats as stats

#%%
with open('/Users/sonmjack/Downloads/simon_paper/gene_list_age10.pkl', 'rb') as file:
    gene_list_10 = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_fam_age10.pkl', 'rb') as file:
    dy_list_fam1 = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_Nov_age10.pkl', 'rb') as file:
    dy_list_nov = pickle.load(file)

with open('/Users/sonmjack/Downloads/simon_paper/dynamic_list_famr2_age10.pkl', 'rb') as file:
    dy_list_famr2 = pickle.load(file)
with open('/Users/sonmjack/Downloads/simon_paper/fam1r2_neuron_list_age10.pkl', 'rb') as file:
    neuron_spike = pickle.load(file)
#%%
test = dy_list_famr2[4]
neuron = neuron_spike[4]
#%%
figure = plt.figure(figsize=(15, 10))

ax1 = plt.subplot(211)
plt.plot(neuron[1,:],label='Wild type, age>10, neuron NO.10')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel("Time series of spike trains")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax2 = plt.subplot(212)
plt.plot(neuron[24,:],label='Wild type, age>10, neuron NO.24',color = 'coral')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel("Time series of spike trains")
ax2.set_ylabel("Amplitude")
ax2.legend()

plt.show()
#%%
test_matrix = np.zeros((2,2))
test_matrix[0,0] = 0
test_matrix[1,1] = 0
test_matrix[0,1] = 0
test_matrix[1,0] = 0.67
row_labels = [10, 24]
col_labels = [10, 24]
# 绘制热图
plt.figure(figsize=(6, 4))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
ax = sns.heatmap(test_matrix, fmt="d", cmap =cmap, xticklabels=col_labels, yticklabels=row_labels, vmin=0, vmax=1)

ax.set_title("Weight matrix of N0.10 and No.24 (wild type, age>10)")
for i in range(len(test_matrix)):
    for j in range(len(test_matrix[i])):
        ax.text(j + 0.5, i + 0.5, str(test_matrix[i][j]), ha='center', va='center', color='black')

plt.show()


#%%

def average_clustering_coefficient_digraph(G):
    # 计算有向图的平均聚类系数
    clustering_coeffs = nx.clustering(G.to_undirected())
    avg_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs)
    return avg_clustering

def average_shortest_path_length_digraph(G):
    # 计算有向图的平均连接长度
    shortest_paths = list(nx.shortest_path_length(G).values())
    avg_shortest_path_length = sum(shortest_paths) / len(shortest_paths)
    return avg_shortest_path_length
def build_graph(g,label):
    dy_r = g.copy()
    t_p_G = dy_r
    threshold = (np.max(dy_r)-np.min(dy_r))*0.1+np.min(dy_r)
    if label == 'weak':
        t_p_G[(t_p_G >= threshold)] = 0
    if label == 'strong':
        t_p_G[(t_p_G <= threshold)] = 0

    G = nx.DiGraph(t_p_G)

    avg_path_length = average_shortest_path_length_digraph(G)
    avg_clustering = average_clustering_coefficient_digraph(G)

    adjacency_matrix = nx.to_numpy_array(G)
    adjacency_matrix_array = np.array(adjacency_matrix)

    return avg_clustering, avg_path_length, adjacency_matrix_array

def draw_pic(ax,degree,legend,color):
    degree = np.array(list(degree))
    sns.histplot(degree, bins=15, alpha=0.5, color=color, label=legend,kde=True,ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Degree")
    ax.legend()
    return ax
#%%
cluster_list_wt_fam = []
cluster_list_wt_famr2 = []
cluster_list_wt_nov = []

cluster_list_ad_fam = []
cluster_list_ad_famr2 = []
cluster_list_ad_nov = []

length_list_wt_fam = []
length_list_wt_famr2 = []
length_list_wt_nov = []

length_list_ad_fam = []
length_list_ad_famr2= []
length_list_ad_nov= []


link_list_wt_fam = []
link_list_wt_famr2 = []
link_list_wt_nov = []

link_list_ad_fam = []
link_list_ad_famr2= []
link_list_ad_nov= []


for i in range(len(dy_list_fam1)):
    avg_clustering1, avg_path_length1,link_matrix1 = build_graph(dy_list_fam1[i],'weak')
    avg_clustering2, avg_path_length2,link_matrix2 = build_graph(dy_list_nov[i],'weak')
    avg_clustering3, avg_path_length3,link_matrix3 = build_graph(dy_list_famr2[i], 'weak')
    if gene_list_10[i] == 119:
        type = 'wt'
        cluster_list_wt_fam.append(avg_clustering1)
        cluster_list_wt_nov.append(avg_clustering2)
        cluster_list_wt_famr2.append(avg_clustering3)

        length_list_wt_fam.append(avg_path_length1)
        length_list_wt_nov.append(avg_path_length2)
        length_list_wt_famr2.append(avg_path_length3)

        link_list_wt_fam.append(link_matrix1)
        link_list_wt_nov.append(link_matrix2)
        link_list_wt_famr2.append(link_matrix3)

    else:
        type = 'ad'
        cluster_list_ad_fam.append(avg_clustering1)
        cluster_list_ad_nov.append(avg_clustering2)
        cluster_list_ad_famr2.append(avg_clustering3)

        length_list_ad_fam.append(avg_path_length1)
        length_list_ad_nov.append(avg_path_length2)
        length_list_ad_famr2.append(avg_path_length3)

        link_list_ad_fam.append(link_matrix1)
        link_list_ad_nov.append(link_matrix2)
        link_list_ad_famr2.append(link_matrix3)


#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(cluster_list_ad_fam, cluster_list_ad_nov)
t, p2 = stats.ttest_ind(cluster_list_ad_fam, cluster_list_ad_famr2)
t, p3 = stats.ttest_ind(cluster_list_ad_nov, cluster_list_ad_famr2)

vector1 = np.array(cluster_list_ad_fam)
vector2 = np.array(cluster_list_ad_nov)
vector3 = np.array(cluster_list_ad_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("clustering coefficient in three environments (weak connection)")
plt.ylabel("Values of clustering coefficient")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.35, 0.1, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("5xFAD age > 6")
plt.show()

#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(cluster_list_wt_fam, cluster_list_wt_nov)
t, p2 = stats.ttest_ind(cluster_list_wt_fam, cluster_list_wt_famr2)
t, p3 = stats.ttest_ind(cluster_list_wt_nov, cluster_list_wt_famr2)

vector1 = np.array(cluster_list_wt_fam)
vector2 = np.array(cluster_list_wt_nov)
vector3 = np.array(cluster_list_wt_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("clustering coefficient in three environments (weak connection)")
plt.ylabel("Values of clustering coefficient")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

# plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.1, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("wild type age > 6")
plt.show()

#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(length_list_ad_fam, length_list_ad_nov)
t, p2 = stats.ttest_ind(length_list_ad_fam, length_list_ad_famr2)
t, p3 = stats.ttest_ind(length_list_ad_nov, length_list_ad_famr2)

vector1 = np.array(length_list_ad_fam)
vector2 = np.array(length_list_ad_nov)
vector3 = np.array(length_list_ad_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("geodesic path length in three environments (weak connection)")
plt.ylabel("Values of geodesic path length")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.4, 0.8, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("5xFAD age > 6")
plt.show()

#%%
import scipy.stats as stats
vector1 = np.array(length_list_wt_fam)
vector2 = np.array(length_list_wt_nov)
vector3 = np.array(length_list_wt_famr2)

t, p1 = stats.ttest_ind(vector1, vector2)
t, p2 = stats.ttest_ind(vector1, vector3)
t, p3 = stats.ttest_ind(vector2, vector3)


# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("geodesic path length in three environments (weak connection)")
plt.ylabel("Values of geodesic path length")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

# plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.4, 0.8, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("wild type age > 6")
plt.show()
#%%
    # fig1 = plt.figure(figsize=(10,15))
    # ax1 = plt.subplot(121)
    # ax1 = draw_pic(ax1, degree_values_p1,'Original data','skyblue')
    # ax2 = plt.subplot(122)
    # ax2 = draw_pic(ax2, degree_values_p2,'shuffled data','coral')
    # if gene_list_10[i] == 119:
    #     type = 'wt'
    # else:
    #     type = 'AD'
    # fig1.suptitle("Degree Distribution of original data (weak)" + f'-{type}-' + f'{i}')
    # #t, p = stats.ttest_ind(dy_list[i], dy_list_shuffled[i])
    # t, p = stats.ttest_ind(list(degree_values_p1), list(degree_values_p2))
    # plt.text(0.15, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
    # plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/graph/' + 'weak connection' + f'-{type}-' + f'{i}.jpg')
    # plt.close()

#%%
#%%
cluster_list_wt_fam = []
cluster_list_wt_famr2 = []
cluster_list_wt_nov = []

cluster_list_ad_fam = []
cluster_list_ad_famr2 = []
cluster_list_ad_nov = []

length_list_wt_fam = []
length_list_wt_famr2 = []
length_list_wt_nov = []

length_list_ad_fam = []
length_list_ad_famr2= []
length_list_ad_nov= []

link_list_wt_fam = []
link_list_wt_famr2 = []
link_list_wt_nov = []

link_list_ad_fam = []
link_list_ad_famr2= []
link_list_ad_nov= []

for i in range(len(dy_list_fam1)):
    avg_clustering1, avg_path_length1,link_matrix1 = build_graph(dy_list_fam1[i],'strong')
    avg_clustering2, avg_path_length2,link_matrix2 = build_graph(dy_list_nov[i],'strong')
    avg_clustering3, avg_path_length3,link_matrix3 = build_graph(dy_list_famr2[i], 'strong')
    if gene_list_10[i] == 119:
        type = 'wt'
        cluster_list_wt_fam.append(avg_clustering1)
        cluster_list_wt_nov.append(avg_clustering2)
        cluster_list_wt_famr2.append(avg_clustering3)

        length_list_wt_fam.append(avg_path_length1)
        length_list_wt_nov.append(avg_path_length2)
        length_list_wt_famr2.append(avg_path_length3)

        link_list_wt_fam.append(link_matrix1)
        link_list_wt_nov.append(link_matrix2)
        link_list_wt_famr2.append(link_matrix3)


    else:
        type = 'ad'
        cluster_list_ad_fam.append(avg_clustering1)
        cluster_list_ad_nov.append(avg_clustering2)
        cluster_list_ad_famr2.append(avg_clustering3)

        length_list_ad_fam.append(avg_path_length1)
        length_list_ad_nov.append(avg_path_length2)
        length_list_ad_famr2.append(avg_path_length3)

        link_list_ad_fam.append(link_matrix1)
        link_list_ad_famr2.append(link_matrix2)
        link_list_ad_nov.append(link_matrix3)

#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(cluster_list_ad_fam, cluster_list_ad_nov)
t, p2 = stats.ttest_ind(cluster_list_ad_fam, cluster_list_ad_famr2)
t, p3 = stats.ttest_ind(cluster_list_ad_nov, cluster_list_ad_famr2)

vector1 = np.array(cluster_list_ad_fam)
vector2 = np.array(cluster_list_ad_nov)
vector3 = np.array(cluster_list_ad_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("clustering coefficient in three environments")
plt.ylabel("Values of clustering coefficient")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("5xFAD age > 6")
plt.show()

#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(cluster_list_wt_fam, cluster_list_wt_nov)
t, p2 = stats.ttest_ind(cluster_list_wt_fam, cluster_list_wt_famr2)
t, p3 = stats.ttest_ind(cluster_list_wt_nov, cluster_list_wt_famr2)

vector1 = np.array(cluster_list_wt_fam)
vector2 = np.array(cluster_list_wt_nov)
vector3 = np.array(cluster_list_wt_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("clustering coefficient in three environments")
plt.ylabel("Values of clustering coefficient")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

# plt.text(0.25, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.6, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("wild type age > 6")
plt.show()

#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(length_list_ad_fam, length_list_ad_nov)
t, p2 = stats.ttest_ind(length_list_ad_fam, length_list_ad_famr2)
t, p3 = stats.ttest_ind(length_list_ad_nov, length_list_ad_famr2)

vector1 = np.array(length_list_ad_fam)
vector2 = np.array(length_list_ad_nov)
vector3 = np.array(length_list_ad_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("geodesic path length in three environments")
plt.ylabel("Values of geodesic path length")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.text(0.25, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.4, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.7, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("5xFAD age > 6")
plt.show()

#%%
import scipy.stats as stats
vector1 = np.array(length_list_wt_fam)
vector2 = np.array(length_list_wt_nov)
vector3 = np.array(length_list_wt_famr2)

t, p1 = stats.ttest_ind(vector1, vector2)
t, p2 = stats.ttest_ind(vector1, vector3)
t, p3 = stats.ttest_ind(vector2, vector3)


# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("geodesic path length in three environments")
plt.ylabel("Values of geodesic path length")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

# plt.text(0.25, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.4, 0.8, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.65, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("wild type age > 6")
plt.show()

#%%
A = link_list_wt_fam[0]
N = np.size(A,0)
Detail_Balance_A = []
Compone_strength_A = []


for i in range(N):
    for j in range(N):
        if (A[i, j] > 0):
            Compone_strength_A.append(A[i, j])
        if (i > j and (A[i, j] + A[j, i]) > 0):
            #             print(i,j,Be[i,j])
            Detail_Balance_A.append(abs(A[i, j] - A[j, i]) / (A[i, j] + A[j, i]))