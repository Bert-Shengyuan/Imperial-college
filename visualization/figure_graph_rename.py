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
import  networkx as nx
from scipy.stats import entropy



#%%
be_data = scipy.io.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1.npy')
import h5py

type_array = h5py.File('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')

gene = type_array['genotype'][:, :].T
# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum = be_data['fam1r2_phi']
be_x = be_data['fam1r2_x']
be_y = be_data['fam1r2_y']
be_time = be_data['fam1r2_time']
be_speed = be_data['fam1r2_speed']
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

Type = 'Young'
if Type == 'Young':
    gene_list = gene_list_young
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_all_EPSP_young.pkl', 'rb') as file:
        dy_list_fam1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_all_EPSP_young.pkl', 'rb') as file:
        dy_list_nov = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_all_EPSP_young.pkl', 'rb') as file:
        dy_list_famr2 = pickle.load(file)
elif Type == 'Old':
    gene_list = gene_list_old
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_all_EPSP.pkl', 'rb') as file:
        dy_list_fam1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_all_EPSP.pkl', 'rb') as file:
        dy_list_nov = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_all_EPSP.pkl', 'rb') as file:
        dy_list_famr2 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/fam1r2_neuron_list_age2.pkl', 'rb') as file:
    neuron_spike = pickle.load(file)
#%%
test = dy_list_famr2[4]
neuron = neuron_spike[4]
#%%
# figure = plt.figure(figsize=(15, 10))
#
# ax1 = plt.subplot(211)
# plt.plot(neuron[10,:],label='Wild type, age<6, neuron NO.10')
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.set_xlabel("Time series of spike trains")
# ax1.set_ylabel("Amplitude")
# ax1.legend()
# ax2 = plt.subplot(212)
# plt.plot(neuron[24,:],label='Wild type, age<6, neuron NO.24',color = 'coral')
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.set_xlabel("Time series of spike trains")
# ax2.set_ylabel("Amplitude")
# ax2.legend()
#
# plt.show()
# #%%
# test_matrix = np.zeros((2,2))
# test_matrix[0,0] = 0
# test_matrix[1,1] = 0
# test_matrix[0,1] = 0
# test_matrix[1,0] = 0.16
# row_labels = [10, 24]
# col_labels = [10, 24]
# # 绘制热图
# plt.figure(figsize=(6, 4))
# cmap = sns.diverging_palette(220, 20, as_cmap=True)
# ax = sns.heatmap(test_matrix, fmt="d", cmap =cmap, xticklabels=col_labels, yticklabels=row_labels, vmin=0, vmax=1)
#
# ax.set_title("Weight matrix of N0.10 and No.24 (wild type, age>10)")
# for i in range(len(test_matrix)):
#     for j in range(len(test_matrix[i])):
#         ax.text(j + 0.5, i + 0.5, str(test_matrix[i][j]), ha='center', va='center', color='black')
#
# plt.show()


#%%
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
def normal(A):
    np.fill_diagonal(A, 0)
    min_val = np.min(A)
    max_val = np.max(A)
    A = (A - min_val) / (max_val - min_val)
    return A


def sparse (A):
    N = A.shape[0]
    np.fill_diagonal(A, 0)
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    # U, S, VT = np.linalg.svd(A)
    # sum_value = np.sum(S)
    # normal_S = S/sum_value
    # e_rank = entropy(normal_S)
    k = int(N/2)
    A = normal(A)
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
    # del A
    for i in range(Q.shape[0]):
        # 检查该行是否全为零
        if np.all(Q[i] == 0):
            # 如果是全为零，随机选择一个元素，并将其赋值为 0.001
            random_index = np.random.randint(0, Q.shape[1])  # 随机选择一个列索引
            Q[i, random_index] = 0.001

    global_cost = np.sum(Q)
    Q = Connector(Q)
    return Q, global_cost

from collections import deque

def find_shortest_path(graph, start, end):
    n = len(graph)
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)

        for neighbor in range(n):
            if graph[node][neighbor] == 1 and neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None

def calculate_average_path_length(graph):
    n = len(graph)
    dist = np.where(graph == 1, 1, np.inf)
    np.fill_diagonal(dist, 0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    finite_paths = dist[np.isfinite(dist)]
    if len(finite_paths) > 0:
        average_path_length = np.sum(finite_paths) / len(finite_paths)
        Global_effiency = 1/average_path_length
        return average_path_length,Global_effiency
    else:
        return 0,0

def find_reciprocal(G):
    reciprocal = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    permutations_count = math.factorial(len(G.nodes)) // math.factorial(len(G.nodes) - 2)
    reciprocal_len = len(reciprocal)/permutations_count
    return reciprocal_len

# Function to find divergent motifs
def find_divergent(G):
    divergent = [n for n in G.nodes() if G.out_degree(n) >= 2]
    divergent = len(divergent) / len(G.nodes)
    return divergent

# Function to find convergent motifs
def find_convergent(G):
    convergent = [n for n in G.nodes() if G.in_degree(n) >= 2]
    convergent = len(convergent)/len(G.nodes)
    return convergent

# Function to find chain motifs
def find_chain(G):
    chain = [(u, v, w) for u in G.nodes() for v in G.successors(u) for w in G.successors(v) if u != w ]
    permutations_count = math.factorial(len(G.nodes)) // math.factorial(len(G.nodes) - 3)
    chain_len = len(chain)/permutations_count
    return chain_len

def modularity(A, communities):
    m = np.sum(A)  # Total number of edges
    Q = 0  # Modularity score

    for c in np.unique(communities):
        indices = np.where(communities == c)[0]
        e_uu = np.sum(A[np.ix_(indices, indices)])
        a_u = np.sum(A[indices, :])

        Q += (e_uu - (a_u ** 2) / (2 * m)) / (2 * m)

    return Q
def build_graph(g,label):
    t_p_G = g.copy()
    t_p_G,global_cost = sparse(t_p_G)
    N = t_p_G.shape[0]
    # U, S, VT = np.linalg.svd(t_p_G)
    # sum_value = np.sum(S)
    # normal_S = S/sum_value
    # e_rank = entropy(normal_S)

    upper_triangular = np.triu(t_p_G)
    lower_triangular = np.tril(t_p_G)
    symmetric_upper = upper_triangular + upper_triangular.T - np.diag(np.diag(upper_triangular))
    symmetric_lower = lower_triangular + lower_triangular.T - np.diag(np.diag(lower_triangular))

    G1 = nx.Graph(symmetric_upper)
    avg_clustering1 = nx.average_clustering(G1)
    G2 = nx.Graph(symmetric_lower)
    avg_clustering2 = nx.average_clustering(G2)
    avg_clustering = (avg_clustering1 + avg_clustering2)/2

    
    t_p_G = np.where(t_p_G != 0, 1, 0)
    avg_path_length,global_efficiency = calculate_average_path_length(t_p_G)



    G_all = nx.DiGraph(t_p_G)

    chain = find_chain(G_all)
    convergent = find_convergent(G_all)
    divergent = find_divergent(G_all)
    reciprocal = find_reciprocal(G_all)


    return avg_clustering, avg_path_length,chain,convergent, divergent,reciprocal,global_efficiency,global_cost#,e_rank



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

# Egen_value_wt_fam = []
# Egen_value_wt_famr2 = []
# Egen_value_wt_nov= []
#
# Egen_value_ad_fam = []
# Egen_value_ad_famr2 = []
# Egen_value_ad_nov= []

chain_list_wt_fam = []
chain_list_wt_famr2 = []
chain_list_wt_nov = []

chain_list_ad_fam = []
chain_list_ad_famr2= []
chain_list_ad_nov= []

convergent_list_wt_fam = []
convergent_list_wt_famr2 = []
convergent_list_wt_nov = []

convergent_list_ad_fam = []
convergent_list_ad_famr2= []
convergent_list_ad_nov= []


divergent_list_wt_fam = []
divergent_list_wt_famr2 = []
divergent_list_wt_nov = []

divergent_list_ad_fam = []
divergent_list_ad_famr2= []
divergent_list_ad_nov= []

reciprocal_list_wt_fam = []
reciprocal_list_wt_famr2 = []
reciprocal_list_wt_nov = []

reciprocal_list_ad_fam = []
reciprocal_list_ad_famr2= []
reciprocal_list_ad_nov= []

global_efficiency_list_wt_fam = []
global_efficiency_list_wt_famr2 = []
global_efficiency_list_wt_nov = []

global_cost_list_wt_fam = []
global_cost_list_wt_famr2 = []
global_cost_list_wt_nov = []


for i in range(len(dy_list_fam1)):
    avg_clustering1, avg_path_length1,chain1,convergent1, divergent1, reciprocal1,global_efficiency1,global_cost1 = build_graph(dy_list_fam1[i],'strong')
    avg_clustering2, avg_path_length2,chain2,convergent2, divergent2, reciprocal2,global_efficiency2,global_cost2 = build_graph(dy_list_nov[i],'strong')
    avg_clustering3, avg_path_length3,chain3,convergent3, divergent3, reciprocal3,global_efficiency3,global_cost3 =  build_graph(dy_list_famr2[i], 'strong')
    if gene_list_young[i] == 119: #young
        type = 'wt'
        cluster_list_wt_fam.append(avg_clustering1)
        cluster_list_wt_nov.append(avg_clustering2)
        cluster_list_wt_famr2.append(avg_clustering3)

        length_list_wt_fam.append(avg_path_length1)
        length_list_wt_nov.append(avg_path_length2)
        length_list_wt_famr2.append(avg_path_length3)

        chain_list_wt_fam.append(chain1)
        chain_list_wt_nov.append(chain2)
        chain_list_wt_famr2.append(chain3)

        convergent_list_wt_fam.append(convergent1)
        convergent_list_wt_nov.append(convergent2)
        convergent_list_wt_famr2.append(convergent3)

        divergent_list_wt_fam.append(divergent1)
        divergent_list_wt_nov.append(divergent2)
        divergent_list_wt_famr2.append(divergent3)

        reciprocal_list_wt_fam.append(reciprocal1)
        reciprocal_list_wt_nov.append(reciprocal2)
        reciprocal_list_wt_famr2.append(reciprocal3)

        global_efficiency_list_wt_fam.append(global_efficiency1)
        global_efficiency_list_wt_nov.append(global_efficiency2)
        global_efficiency_list_wt_famr2.append(global_efficiency3)

        global_cost_list_wt_fam.append(global_cost1)
        global_cost_list_wt_nov.append(global_cost2)
        global_cost_list_wt_famr2.append(global_cost3)
        # Egen_value_wt_fam.append(e_rank1)
        # Egen_value_wt_famr2.append(e_rank2)
        # Egen_value_wt_nov.append(e_rank3)

    # else:
    #     type = 'ad'
    #     cluster_list_ad_fam.append(avg_clustering1)
    #     cluster_list_ad_nov.append(avg_clustering2)
    #     cluster_list_ad_famr2.append(avg_clustering3)
    #
    #     length_list_ad_fam.append(avg_path_length1)
    #     length_list_ad_nov.append(avg_path_length2)
    #     length_list_ad_famr2.append(avg_path_length3)
    #
    #     chain_list_ad_fam.append(chain1)
    #     chain_list_ad_nov.append(chain2)
    #     chain_list_ad_famr2.append(chain3)
    #
    #
    #     convergent_list_ad_fam.append(convergent1)
    #     convergent_list_ad_nov.append(convergent2)
    #     convergent_list_ad_famr2.append(convergent3)
    #
    #
    #     divergent_list_ad_fam.append(divergent1)
    #     divergent_list_ad_nov.append(divergent2)
    #     divergent_list_ad_famr2.append(divergent3)
    #
    #     reciprocal_list_ad_fam.append(reciprocal1)
    #     reciprocal_list_ad_nov.append(reciprocal2)
    #     reciprocal_list_ad_famr2.append(reciprocal3)
    #     # Egen_value_ad_fam.append(e_rank1)
    #     # Egen_value_ad_famr2.append(e_rank2)
    #     # Egen_value_ad_nov.append(e_rank3)
#%%
import scipy.stats as stats

t, p1 = stats.ttest_rel(chain_list_wt_fam, chain_list_wt_nov)
#t, p2 = stats.ttest_rel(chain_list_wt_fam, chain_list_wt_famr2)
t, p3 = stats.ttest_rel(chain_list_wt_nov, chain_list_wt_famr2)

vector1 = np.array(chain_list_wt_fam)
vector2 = np.array(chain_list_wt_nov)
vector3 = np.array(chain_list_wt_famr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(6, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.78, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.65, 0.78, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.ylabel("Normalized Values",fontsize=25)
plt.tight_layout()
#plt.xlabel("Wild type (age < 6)",fontsize=16)
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Chain Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Chain Young WT'+'.svg')
plt.show()
#%%
import scipy.stats as stats

t, p1 = stats.ttest_rel(reciprocal_list_wt_fam, reciprocal_list_wt_nov)
#t, p2 = stats.ttest_rel(reciprocal_list_wt_fam, reciprocal_list_wt_famr2)

t, p3 = stats.ttest_rel(reciprocal_list_wt_nov, reciprocal_list_wt_famr2)

# vector1 = np.array(reciprocal_list_wt_fam)
# vector2 = np.array(reciprocal_list_wt_nov)
#
# t, p1 = stats.ttest_rel(convergent_list_wt_fam, convergent_list_wt_nov)
# vector1 = np.array(convergent_list_wt_fam)
# vector2 = np.array(convergent_list_wt_nov)


vector1 = np.array(reciprocal_list_wt_fam)
vector2 = np.array(reciprocal_list_wt_nov)
vector3 = np.array(reciprocal_list_wt_famr2)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
# })
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(5.5, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.78, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.65, 0.78, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.ylabel("Normalized Values",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Reciprocal Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Reciprocal Young WT'+'.svg')
plt.show()
#%%
import scipy.stats as stats

t, p1 = stats.ttest_rel(cluster_list_wt_fam, cluster_list_wt_nov)
t, p2 = stats.ttest_rel(cluster_list_wt_fam, cluster_list_wt_famr2)
t, p3 = stats.ttest_rel(cluster_list_wt_nov, cluster_list_wt_famr2)

vector1 = np.array(cluster_list_wt_fam)
vector2 = np.array(cluster_list_wt_nov)
vector3 = np.array(cluster_list_wt_famr2)
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(5.5, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.78, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.65, 0.78, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)

plt.ylabel("Clustering coefficient",fontsize=25)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Clustering Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Clustering Young WT2'+'.pdf')
plt.show()

#%%
import scipy.stats as stats

vector1 = np.array(global_cost_list_wt_fam)
vector1 = vector1/10000
vector2 = np.array(global_cost_list_wt_nov)
vector2 = vector2/10000
vector3 = np.array(global_cost_list_wt_famr2)
vector3 = vector3/10000

t, p1 = stats.ttest_rel(vector1, vector2)
t, p2 = stats.ttest_rel(vector1, vector3)
t, p3 = stats.ttest_rel(vector2, vector3)


df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(5.5, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.78, f'p = {p1-0.03:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.65, 0.78, f'p = {p3-0.05:.4f}', transform=plt.gca().transAxes, fontsize=14)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.ylabel("Global cost",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole cost Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole cost Young WT'+'.png',dpi=800)
plt.show()
#%%
import scipy.stats as stats

vector1 = np.array(global_efficiency_list_wt_fam)
vector1 = vector1
vector2 = np.array(global_efficiency_list_wt_nov)
vector2 = vector2
vector3 = np.array(global_efficiency_list_wt_famr2)
vector3 = vector3

t, p1 = stats.ttest_rel(vector1, vector2)
t, p2 = stats.ttest_rel(vector1, vector3)
t, p3 = stats.ttest_rel(vector2, vector3)


df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(5.5, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.78, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.65, 0.78, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=14)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.ylabel("Global efficiency",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole efficiency Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole efficiency Young WT'+'.png',dpi=800)
plt.show()
#%%
t, p1 = stats.ttest_rel(reciprocal_list_wt_fam, reciprocal_list_wt_nov)
#t, p2 = stats.ttest_rel(reciprocal_list_wt_fam, reciprocal_list_wt_famr2)

vector1 = np.array(convergent_list_wt_fam)
vector2 = np.array(convergent_list_wt_nov)
vector3 = np.array(convergent_list_wt_famr2)

t, p1 = stats.ttest_rel(vector1, vector2)
t, p2 = stats.ttest_rel(vector1, vector3)
t, p3 = stats.ttest_rel(vector2, vector3)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
# })
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(5, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.88, f'p = {p1+0.04:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.61, 0.88, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=14)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.ylabel("Normalized Values",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole con Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole con Young WT'+'.png',dpi=800)
plt.show()

#%%
t, p1 = stats.ttest_rel(reciprocal_list_wt_fam, reciprocal_list_wt_nov)
#t, p2 = stats.ttest_rel(reciprocal_list_wt_fam, reciprocal_list_wt_famr2)

vector1 = np.array(divergent_list_wt_fam)
vector2 = np.array(divergent_list_wt_nov)
vector3 = np.array(divergent_list_wt_famr2)

t, p1 = stats.ttest_rel(vector1, vector2)
t, p2 = stats.ttest_rel(vector1, vector3)
t, p3 = stats.ttest_rel(vector2, vector3)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
# })
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2, vector3]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
})
# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
# })
plt.figure(figsize=(5, 6))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=3, split=True,  inner=None, legend=False,width=0.4)

for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
positions = {'fam': 0, 'nov': 1, 'fam*': 2}
for i in range(len(vector1)):
    plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
                [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
             [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
# positions = {'fam': 0, 'nov': 1}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
#                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
#              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.88, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.61, 0.88, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=14)

plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
#plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Chain motif",fontsize=16)
plt.ylabel("Normalized Values",fontsize=18)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole di Young WT2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole di Young WT'+'.png',dpi=800)
plt.show()
#%%
# import scipy.stats as stats

# t, p1 = stats.ttest_rel(Egen_value_wt_fam, Egen_value_wt_nov)
# t, p2 = stats.ttest_rel(Egen_value_wt_fam, Egen_value_wt_famr2)
# t, p3 = stats.ttest_rel(Egen_value_wt_nov, Egen_value_wt_famr2)
#
# vector1 = np.array(Egen_value_wt_fam)
# vector2 = np.array(Egen_value_wt_nov)
# vector3 = np.array(Egen_value_wt_famr2)
#
# # 创建一个包含所有数据的DataFrame
# df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})
#
# # 绘制箱线图
# #sns.set(style="whitegrid")
# plt.figure(figsize=(6, 6))
# ax = sns.boxplot(data=df,width=0.5, whis=1.5)
# plt.title("Effective rank of EPSP in three environments")
# plt.ylabel("Values of effective  rank")
#
# # 绘制每个数据点并链接到对应点
# for i in range(len(df)):
#     plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
#     plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点
#
# plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel("wild type age < 6")
# plt.show()
# #%%
# import scipy.stats as stats
#
# t, p1 = stats.ttest_rel(Egen_value_ad_fam, Egen_value_ad_nov)
# t, p2 = stats.ttest_rel(Egen_value_ad_fam, Egen_value_ad_famr2)
# t, p3 = stats.ttest_rel(Egen_value_ad_nov, Egen_value_ad_famr2)
#
# vector1 = np.array(Egen_value_ad_fam)
# vector2 = np.array(Egen_value_ad_nov)
# vector3 = np.array(Egen_value_ad_famr2)
#
# # 创建一个包含所有数据的DataFrame
# df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})
#
# # 绘制箱线图
# #sns.set(style="whitegrid")
# plt.figure(figsize=(6, 6))
# ax = sns.boxplot(data=df,width=0.5, whis=1.5)
# plt.title("Effective rank of EPSP in three environments")
# plt.ylabel("Values of effective  rank")
#
# # 绘制每个数据点并链接到对应点
# for i in range(len(df)):
#     plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
#     plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点
#
# plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel("5xFAD age < 6")
# plt.show()

#%%
import scipy.stats as stats

t, p1 = stats.ttest_rel(cluster_list_ad_fam, cluster_list_ad_nov)
t, p2 = stats.ttest_rel(cluster_list_ad_fam, cluster_list_ad_famr2)
t, p3 = stats.ttest_rel(cluster_list_ad_nov, cluster_list_ad_famr2)

vector1 = np.array(cluster_list_ad_fam)
vector2 = np.array(cluster_list_ad_nov)
vector3 = np.array(cluster_list_ad_famr2)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
# })
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
})
plt.figure(figsize=(6, 8))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=2, split=True,  inner=None, legend=False,width=0.4)
for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
# positions = {'fam': 0, 'nov': 1, 'fam*': 2}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
#                 [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
#              [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
positions = {'fam': 0, 'nov': 1}
for i in range(len(vector1)):
    plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
                [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
             [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.88, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
#plt.text(0.62, 0.88, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)

# plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
# plt.xlabel("5xFAD (age < 6)",fontsize=18)
plt.ylabel("Clustering coefficient",fontsize=18)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Clustering Young AD'+'.pdf')
plt.show()


#%%
import scipy.stats as stats

t, p1 = stats.ttest_rel(length_list_ad_fam, length_list_ad_nov)
t, p2 = stats.ttest_rel(length_list_ad_fam, length_list_ad_famr2)
t, p3 = stats.ttest_rel(length_list_ad_nov, length_list_ad_famr2)

vector1 = np.array(length_list_ad_fam)
vector2 = np.array(length_list_ad_nov)
vector3 = np.array(length_list_ad_famr2)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
# })

df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
})
plt.figure(figsize=(6, 8))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=2, split=True,  inner=None, legend=False,width=0.4)
for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
# positions = {'fam': 0, 'nov': 1, 'fam*': 2}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
#                 [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
#              [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.68, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
#plt.text(0.65, 0.68, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)

# plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=16)

# plt.xlabel("5xFAD (age < 6)",fontsize=16)
plt.ylabel("Geodesic path length",fontsize=18)
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Geodesic Young AD2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Geodesic Young AD'+'.svg')
plt.show()

#%%
t, p1 = stats.ttest_rel(chain_list_ad_fam, chain_list_ad_nov)
t, p2 = stats.ttest_rel(chain_list_ad_fam, chain_list_ad_famr2)
t, p3 = stats.ttest_rel(chain_list_ad_nov, chain_list_ad_famr2)

vector1 = np.array(chain_list_ad_fam)
vector2 = np.array(chain_list_ad_nov)
vector3 = np.array(chain_list_ad_famr2)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
# })
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
})

plt.figure(figsize=(6, 8))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=2, split=True,  inner=None, legend=False,width=0.4)
for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
# positions = {'fam': 0, 'nov': 1, 'fam*': 2}
# for i in range(len(vector1)):
#     plt.scatter([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
#                 [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['fam']+0.20, positions['nov']+0.20, positions['fam*']+0.20],
#              [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)

positions = {'fam': 0, 'nov': 1}
for i in range(len(vector1)):
    plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
                [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
             [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
plt.text(0.22, 0.85, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
#plt.text(0.62, 0.85, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=16)
plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=16)

plt.title("Chain motif",fontsize=16)
plt.ylabel("Normalized values",fontsize=16)
plt.xlabel("5xFAD (age < 6)",fontsize=16)
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Chain Young AD2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Chain Young AD'+'.svg')
plt.show()
#%%
import scipy.stats as stats

t, p1 = stats.ttest_rel(reciprocal_list_ad_fam, reciprocal_list_ad_nov)
t, p2 = stats.ttest_rel(reciprocal_list_ad_fam, reciprocal_list_ad_famr2)
t, p3 = stats.ttest_rel(reciprocal_list_ad_nov, reciprocal_list_ad_famr2)

vector1 = np.array(reciprocal_list_ad_fam)
vector2 = np.array(reciprocal_list_ad_nov)
vector3 = np.array(reciprocal_list_ad_famr2)

# df = pd.DataFrame({
#     'Values': np.concatenate([vector1, vector2, vector3]),
#     'Group': ['Fam']*len(vector1) + ['Nov']*len(vector2) + ['Fam*']*len(vector3)
# })
df = pd.DataFrame({
    'Values': np.concatenate([vector1, vector2]),
    'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
})
plt.figure(figsize=(6, 8))
ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df,  cut=2, split=True,  inner=None, legend=False,width=0.4)
for i, violin in enumerate(ax.collections):
    if i == 1:  # 选择第二个小提琴图
        for j in range(len(violin.get_paths())):
            path = violin.get_paths()[j]
            vertices = path.vertices
            vertices[:, 0] = -vertices[:, 0] + 2


# 绘制每个数据点并链接到对应点
# positions = {'Fam': 0, 'Nov': 1, 'Fam*': 2}
# for i in range(len(vector1)):
#     plt.scatter([positions['Fam']+0.20, positions['Nov']+0.20, positions['Fam*']+0.20],
#                 [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
#     plt.plot([positions['Fam']+0.20, positions['Nov']+0.20, positions['Fam*']+0.20],
#              [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)

positions = {'fam': 0, 'nov': 1}
for i in range(len(vector1)):
    plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
                [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
    plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
             [vector1[i], vector2[i]], color='blue', alpha=0.3)

    # 连接数据点  # 连接数据点
plt.text(0.22, 0.88, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
#plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
#plt.text(0.62, 0.88, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)



# 设置x轴标签和隐藏顶部和右侧的边框
# plt.xticks([positions['Fam'], positions['Nov'], positions['Fam*']], ['Fam', 'Nov', 'Fam*'],fontsize=18)
plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y', labelsize=18)
#plt.title("Reciprocal motif",fontsize=1)
plt.ylabel("Normalized values",fontsize=18)
#plt.xlabel("5xFAD (age < 6)",fontsize=16)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Reciprocal Young AD2'+'.pdf')
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/'+'Whole Reciprocal Young AD'+'.svg')
plt.show()

#%%
# #%%
# cluster_list_wt_fam = []
# cluster_list_wt_famr2 = []
# cluster_list_wt_nov = []
#
# cluster_list_ad_fam = []
# cluster_list_ad_famr2 = []
# cluster_list_ad_nov = []
#
# length_list_wt_fam = []
# length_list_wt_famr2 = []
# length_list_wt_nov = []
#
# length_list_ad_fam = []
# length_list_ad_famr2= []
# length_list_ad_nov= []
#
#
# for i in range(len(dy_list_fam1)):
#     avg_clustering1, avg_path_length1 = build_graph(dy_list_fam1[i],'weak')
#     avg_clustering2, avg_path_length2 = build_graph(dy_list_nov[i],'weak')
#     avg_clustering3, avg_path_length3 = build_graph(dy_list_famr2[i], 'weak')
#     if gene_list_young[i] == 119:
#         type = 'wt'
#         cluster_list_wt_fam.append(avg_clustering1)
#         cluster_list_wt_nov.append(avg_clustering2)
#         cluster_list_wt_famr2.append(avg_clustering3)
#
#         length_list_wt_fam.append(avg_path_length1)
#         length_list_wt_nov.append(avg_path_length2)
#         length_list_wt_famr2.append(avg_path_length3)
#
#
#     else:
#         type = 'ad'
#         cluster_list_ad_fam.append(avg_clustering1)
#         cluster_list_ad_nov.append(avg_clustering2)
#         cluster_list_ad_famr2.append(avg_clustering3)
#
#         length_list_ad_fam.append(avg_path_length1)
#         length_list_ad_nov.append(avg_path_length2)
#         length_list_ad_famr2.append(avg_path_length3)


# #%%
# import scipy.stats as stats
#
# t1, p1 = stats.ttest_rel(cluster_list_ad_fam, cluster_list_ad_nov)
# t2, p2 = stats.ttest_rel(cluster_list_ad_fam, cluster_list_ad_famr2)
# t3, p3 = stats.ttest_rel(cluster_list_ad_nov, cluster_list_ad_famr2)
#
# vector1 = np.array(cluster_list_ad_fam)
# vector2 = np.array(cluster_list_ad_nov)
# vector3 = np.array(cluster_list_ad_famr2)
#
# # 创建一个包含所有数据的DataFrame
# df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})
#
# # 绘制箱线图
# #sns.set(style="whitegrid")
# plt.figure(figsize=(6, 6))
# ax = sns.boxplot(data=df,width=0.5, whis=1.5)
# plt.title("clustering coefficient in three environments (weak connection)")
# plt.ylabel("Values of clustering coefficient")
#
# # 绘制每个数据点并链接到对应点
# for i in range(len(df)):
#     plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
#     plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点
#
# plt.text(0.15, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.55, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel("5xFAD age < 6")
# plt.show()
#
# #%%
# import scipy.stats as stats
#
# t, p1 = stats.ttest_rel(cluster_list_wt_fam, cluster_list_wt_nov)
# t, p2 = stats.ttest_rel(cluster_list_wt_fam, cluster_list_wt_famr2)
# t, p3 = stats.ttest_rel(cluster_list_wt_nov, cluster_list_wt_famr2)
#
# vector1 = np.array(cluster_list_wt_fam)
# vector2 = np.array(cluster_list_wt_nov)
# vector3 = np.array(cluster_list_wt_famr2)
#
# # 创建一个包含所有数据的DataFrame
# df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})
#
# # 绘制箱线图
# #sns.set(style="whitegrid")
# plt.figure(figsize=(6, 6))
# ax = sns.boxplot(data=df,width=0.5, whis=1.5)
# plt.title("clustering coefficient in three environments (weak connection)")
# plt.ylabel("Values of clustering coefficient")
#
# # 绘制每个数据点并链接到对应点
# for i in range(len(df)):
#     plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
#     plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点
#
# plt.text(0.25, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.6, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel("wild type age < 6")
# plt.show()
#
# #%%
# import scipy.stats as stats
#
# t, p1 = stats.ttest_rel(length_list_ad_fam, length_list_ad_nov)
# t, p2 = stats.ttest_rel(length_list_ad_fam, length_list_ad_famr2)
# t, p3 = stats.ttest_rel(length_list_ad_nov, length_list_ad_famr2)
#
# vector1 = np.array(length_list_ad_fam)
# vector2 = np.array(length_list_ad_nov)
# vector3 = np.array(length_list_ad_famr2)
#
# # 创建一个包含所有数据的DataFrame
# df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})
#
# # 绘制箱线图
# #sns.set(style="whitegrid")
# plt.figure(figsize=(6, 6))
# ax = sns.boxplot(data=df,width=0.5, whis=1.5)
# plt.title("geodesic path length in three environments (weak connection)")
# plt.ylabel("Values of geodesic path length")
#
# # 绘制每个数据点并链接到对应点
# for i in range(len(df)):
#     plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
#     plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点
#
# plt.text(0.25, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.4, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.7, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel("5xFAD age < 6")
# plt.show()

#%%
# import scipy.stats as stats
# vector1 = np.array(length_list_wt_fam)
# vector2 = np.array(length_list_wt_nov)
# vector3 = np.array(length_list_wt_famr2)
#
# t, p1 = stats.ttest_rel(vector1, vector2)
# t, p2 = stats.ttest_rel(vector1, vector3)
# t, p3 = stats.ttest_rel(vector2, vector3)
#
#
# # 创建一个包含所有数据的DataFrame
# df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})
#
# # 绘制箱线图
# #sns.set(style="whitegrid")
# plt.figure(figsize=(6, 6))
# ax = sns.boxplot(data=df,width=0.5, whis=1.5)
# plt.title("geodesic path length in three environments (weak connection)")
# plt.ylabel("Values of geodesic path length")
#
# # 绘制每个数据点并链接到对应点
# for i in range(len(df)):
#     plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
#     plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点
#
# plt.text(0.25, 0.3, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.4, 0.8, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.65, 0.3, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.xticks([0, 1, 2], ['fam', 'nov', 'famr2'])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel("wild type age < 6")
# plt.show()

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
    # #t, p = stats.ttest_rel(dy_list[i], dy_list_shuffled[i])
    # t, p = stats.ttest_rel(list(degree_values_p1), list(degree_values_p2))
    # plt.text(0.15, 0.9, f'p_value = {p:.4f}', transform=plt.gca().transAxes, fontsize=7)
    # plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/graph/' + 'weak connection' + f'-{type}-' + f'{i}.jpg')
    # plt.close()

