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
import h5py

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

type_array = h5py.File('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')
be_data = scipy.io.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1.npy')

gene = type_array['genotype'][:, :].T

# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum_fam1 = be_data['fam1_phi']
be_phi_sum_nov = be_data['nov_phi']
be_phi_sum_fam1r2 = be_data['fam1r2_phi']

be_x_sum_fam1 = be_data['fam1_x']
be_x_sum_nov = be_data['nov_x']
be_x_sum_fam1r2 = be_data['fam1r2_x']

be_y_sum_fam1 = be_data['fam1_y']
be_y_sum_nov = be_data['nov_y']
be_y_sum_fam1r2 = be_data['fam1r2_y']

be_speed_sum_fam1 = be_data['fam1_speed']
be_speed_sum_nov = be_data['nov_speed']
be_speed_sum_fam1r2 = be_data['fam1r2_speed']

be_phi_list_young_fam1 = []
be_phi_list_young_fam1r2 = []
be_phi_list_young_nov = []

be_x_list_young_fam1 = []
be_x_list_young_fam1r2 = []
be_x_list_young_nov = []

be_y_list_young_fam1 = []
be_y_list_young_fam1r2 = []
be_y_list_young_nov = []

be_speed_list_young_fam1 = []
be_speed_list_young_nov = []
be_speed_list_young_fam1r2 = []

gene_list_young = []
for i in range(10, 46, 2):  # 0, len(mat_trigger), 2
    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_phi_list_young_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
        be_phi_list_young_nov.append(be_phi_sum_nov[int(i / 2), 0])
        be_phi_list_young_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

        be_x_list_young_fam1.append(be_x_sum_fam1[int(i / 2), 0])
        be_x_list_young_nov.append(be_x_sum_nov[int(i / 2), 0])
        be_x_list_young_fam1r2.append(be_x_sum_fam1r2[int(i / 2), 0])

        be_y_list_young_fam1.append(be_y_sum_fam1[int(i / 2), 0])
        be_y_list_young_nov.append(be_y_sum_nov[int(i / 2), 0])
        be_y_list_young_fam1r2.append(be_y_sum_fam1r2[int(i / 2), 0])

        gene_list_young.append(mat_trigger[i, 1])

be_phi_list_old_fam1 = []
be_phi_list_old_nov = []
be_phi_list_old_fam1r2 = []

be_x_list_old_fam1 = []
be_x_list_old_fam1r2 = []
be_x_list_old_nov = []

be_y_list_old_fam1 = []
be_y_list_old_fam1r2 = []
be_y_list_old_nov = []

gene_list_old = []
for i in range(0, 10, 2):  # 0, len(mat_trigger), 2
    be_phi_list_old_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
    be_phi_list_old_nov.append(be_phi_sum_nov[int(i / 2), 0])
    be_phi_list_old_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

    be_x_list_old_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
    be_x_list_old_nov.append(be_phi_sum_nov[int(i / 2), 0])
    be_x_list_old_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

    be_y_list_old_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
    be_y_list_old_nov.append(be_phi_sum_nov[int(i / 2), 0])
    be_y_list_old_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])
    gene_list_old.append(mat_trigger[i, 1])
del be_data, be_phi_sum_fam1, be_phi_sum_nov, be_phi_sum_fam1r2
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
    mask = pickle.load(file)

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
    Q = nx.to_networkx_graph(Q, create_using=nx.DiGraph())
    return Q


def normal(A):
    np.fill_diagonal(A, 0)
    min_val = np.min(A)
    max_val = np.max(A)
    A = (A - min_val) / (max_val - min_val)
    return A


def sparse(A):
    N = A.shape[0]
    np.fill_diagonal(A, 0)
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    # U, S, VT = np.linalg.svd(A)
    # sum_value = np.sum(S)
    # normal_S = S/sum_value
    # e_rank = entropy(normal_S)
    k = int(N / 2)
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
    Q = Connector(Q)
    return Q


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


def calculate_average_path_length(graph,size):
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
        local_effiency = 1 / average_path_length
        return average_path_length/size, local_effiency/size
    else:
        return 0, 0

def find_reciprocal(G,size):
    reciprocal = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    permutations_count = math.factorial(len(G.nodes)) // math.factorial(len(G.nodes) - 2)
    reciprocal_len = len(reciprocal) / permutations_count
    return reciprocal_len/size


# Function to find divergent motifs
def find_divergent(G,size):
    divergent = [n for n in G.nodes() if G.out_degree(n) >= 2]
    divergent = len(divergent) / len(G.nodes)
    return divergent/size


# Function to find convergent motifs
def find_convergent(G,size):
    convergent = [n for n in G.nodes() if G.in_degree(n) >= 2]
    convergent = len(convergent) / len(G.nodes)
    return convergent/size


# Function to find chain motifs
def find_chain(G,size):
    chain = [(u, v, w) for u in G.nodes() for v in G.successors(u) for w in G.successors(v) if u != w]
    permutations_count = math.factorial(len(G.nodes)) // math.factorial(len(G.nodes) - 3)
    chain_len = len(chain) / permutations_count
    return chain_len/size


def modularity(A, communities):
    m = np.sum(A)  # Total number of edges
    Q = 0  # Modularity score

    for c in np.unique(communities):
        indices = np.where(communities == c)[0]
        e_uu = np.sum(A[np.ix_(indices, indices)])
        a_u = np.sum(A[indices, :])

        Q += (e_uu - (a_u ** 2) / (2 * m)) / (2 * m)

    return Q


def build_graph(g, Community):
    t_p_G = g.copy()
    N = t_p_G.shape[0]
    t_p_G = sparse(t_p_G)
    size = Community.shape[0]
    D = t_p_G.copy()

    B = D.copy()
    Between = D.copy()
    Within = D.copy()
    Di_Between = []
    Di_Within = []

    Community = Community.tolist()

    for node in B.nodes:
        B.nodes[node]["community"] = Community[node]

    ######## Creating a subgraph using the node atterbute ######################################################

    for x in range(max(Community) + 1):
        selected_nodes = [n for n, v in B.nodes(data=True) if v["community"] == x]
        # print (selected_nodes)
        globals()['G%s' % x] = B.subgraph(selected_nodes)

    ######## Creating the network between and within the communities ##########################################

    for x in range(max(Community) + 1):
        List_edge = list(globals()['G%s' % x].edges())
        Between.remove_edges_from(List_edge)

    List_edge1 = list(Between.edges())
    Within.remove_edges_from(List_edge1)


    ######## Ploting the Histogram ##############################################################################
    Wi = nx.to_numpy_array(Within)
    Be = nx.to_numpy_array(Between)

    local_cost_W = np.sum(Wi)/size
    local_cost_B = np.sum(Be)/size

    upper_triangular = np.triu(Wi)
    lower_triangular = np.tril(Wi)
    symmetric_upper = upper_triangular + upper_triangular.T - np.diag(np.diag(upper_triangular))
    symmetric_lower = lower_triangular + lower_triangular.T - np.diag(np.diag(lower_triangular))

    G1 = nx.Graph(symmetric_upper)
    avg_clustering1 = nx.average_clustering(G1)
    G2 = nx.Graph(symmetric_lower)
    avg_clustering2 = nx.average_clustering(G2)
    avg_clustering_W = (avg_clustering1 + avg_clustering2) / 2/size

    Wi = np.where(Wi >= 0.01, 1, 0)
    avg_path_length_W,local_efficiency_W = calculate_average_path_length(Wi,size)

    G_all = nx.DiGraph(Wi)

    chain_W = find_chain(G_all,size)
    convergent_W = find_convergent(G_all,size)
    divergent_W = find_divergent(G_all,size)
    reciprocal_W = find_reciprocal(G_all,size)

    # between
    upper_triangular = np.triu(Be)
    lower_triangular = np.tril(Be)
    symmetric_upper = upper_triangular + upper_triangular.T - np.diag(np.diag(upper_triangular))
    symmetric_lower = lower_triangular + lower_triangular.T - np.diag(np.diag(lower_triangular))

    G1 = nx.Graph(symmetric_upper)
    avg_clustering1 = nx.average_clustering(G1)
    G2 = nx.Graph(symmetric_lower)
    avg_clustering2 = nx.average_clustering(G2)
    avg_clustering_B = (avg_clustering1 + avg_clustering2) / 2/size

    Be = np.where(Be >= 0.01, 1, 0)
    avg_path_length_B,local_efficiency_B = calculate_average_path_length(Be,size)

    G_all = nx.DiGraph(Be)

    chain_B = find_chain(G_all,size)
    convergent_B = find_convergent(G_all,size)
    divergent_B = find_divergent(G_all,size)
    reciprocal_B = find_reciprocal(G_all,size)


    return (avg_clustering_W, avg_path_length_W, chain_W, convergent_W, divergent_W, reciprocal_W, local_efficiency_W, local_cost_W,
            avg_clustering_B, avg_path_length_B, chain_B, convergent_B, divergent_B, reciprocal_B, local_efficiency_B, local_cost_B)





def draw_pic(ax, degree, legend, color):
    degree = np.array(list(degree))
    sns.histplot(degree, bins=15, alpha=0.5, color=color, label=legend, kde=True, ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Degree")
    ax.legend()
    return ax


Type = 'Young'
if Type == 'Young':
    gene_list = gene_list_young
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_all_EPSP_young.pkl', 'rb') as file:
        dy_list_fam1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_all_EPSP_young.pkl', 'rb') as file:
        dy_list_nov = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_all_EPSP_young.pkl', 'rb') as file:
        dy_list_famr2 = pickle.load(file)

cluster_number = []
Color_Code2 = ['white', 'darkblue', 'red', "orange", "pink", "olive", "cyan", "yellow", "green", "brown", "gray",
               "tomato", 'limegreen']

list_color = []
for color in mcolors.CSS4_COLORS:
    list_color.append(color)
random.shuffle(list_color)
Color_Code1 = [*Color_Code2, *list_color]
Color_Code = [*Color_Code1, *list_color]
z = 0
v = 0

#%%
cluster_list_wt_fam = []
cluster_list_wt_famr2 = []
cluster_list_wt_nov = []

cluster_list_be_fam = []
cluster_list_be_famr2 = []
cluster_list_be_nov = []

length_list_wt_fam = []
length_list_wt_famr2 = []
length_list_wt_nov = []

length_list_be_fam = []
length_list_be_famr2= []
length_list_be_nov= []

# Egen_value_wt_fam = []
# Egen_value_wt_famr2 = []
# Egen_value_wt_nov= []
#
# Egen_value_be_fam = []
# Egen_value_be_famr2 = []
# Egen_value_be_nov= []

chain_list_wt_fam = []
chain_list_wt_famr2 = []
chain_list_wt_nov = []

chain_list_be_fam = []
chain_list_be_famr2= []
chain_list_be_nov= []

convergent_list_wt_fam = []
convergent_list_wt_famr2 = []
convergent_list_wt_nov = []

convergent_list_be_fam = []
convergent_list_be_famr2= []
convergent_list_be_nov= []


divergent_list_wt_fam = []
divergent_list_wt_famr2 = []
divergent_list_wt_nov = []

divergent_list_be_fam = []
divergent_list_be_famr2= []
divergent_list_be_nov= []

reciprocal_list_wt_fam = []
reciprocal_list_wt_famr2 = []
reciprocal_list_wt_nov = []

reciprocal_list_be_fam = []
reciprocal_list_be_famr2= []
reciprocal_list_be_nov= []

local_efficiency_list_wt_fam = []
local_efficiency_list_wt_famr2 = []
local_efficiency_list_wt_nov = []

local_efficiency_list_be_fam = []
local_efficiency_list_be_famr2 = []
local_efficiency_list_be_nov = []

local_cost_list_wt_fam = []
local_cost_list_wt_famr2 = []
local_cost_list_wt_nov = []

local_cost_list_be_fam = []
local_cost_list_be_famr2 = []
local_cost_list_be_nov = []



for index in range(len(dy_list_fam1)):  #(9,10)# len(dy_list)
    plt.close()
    if gene_list_young[index] == 119:
        type = 'wild type'

        with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
            all_results_fam1 = pickle.load(file)
        with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
            all_results_nov = pickle.load(file)
        with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_Signal_Markov_' + str(z) + '.pkl',
                  'rb') as file:
            all_results_famr2 = pickle.load(file)


        closest_number = min(all_results_fam1['selected_partitions'], key= lambda x: abs(x - 133))
        Community = all_results_fam1['community_id'][closest_number]
        dy_matrix = dy_list_fam1[index]

        (avg_clustering_W, avg_path_length_W, chain_W, convergent_W, divergent_W, reciprocal_W, local_efficiency_W,
         local_cost_W,
         avg_clustering_B, avg_path_length_B, chain_B, convergent_B, divergent_B, reciprocal_B, local_efficiency_B,
         local_cost_B) = build_graph(dy_matrix,Community)

        cluster_list_wt_fam.append(avg_clustering_W)
        cluster_list_be_fam.append(avg_clustering_B)

        length_list_wt_fam.append(avg_path_length_W)
        length_list_be_fam.append(avg_path_length_B)

        chain_list_wt_fam.append(chain_W)
        chain_list_be_fam.append(chain_B)

        convergent_list_wt_fam.append(convergent_W)
        convergent_list_be_fam.append(convergent_B)

        divergent_list_wt_fam.append(divergent_W)
        divergent_list_be_fam.append(divergent_B)

        reciprocal_list_wt_fam.append(reciprocal_W)
        reciprocal_list_be_fam.append(reciprocal_B)

        local_efficiency_list_wt_fam.append(local_efficiency_W)
        local_efficiency_list_be_fam.append(local_efficiency_B)

        local_cost_list_wt_fam.append(local_cost_W)
        local_cost_list_be_fam.append(local_cost_B)

        #%%
        closest_number = min(all_results_famr2['selected_partitions'], key=lambda x: abs(x - 133))
        Community = all_results_famr2['community_id'][closest_number]
        dy_matrix = dy_list_famr2[index]

        (avg_clustering_W, avg_path_length_W, chain_W, convergent_W, divergent_W, reciprocal_W, local_efficiency_W,
         local_cost_W,
         avg_clustering_B, avg_path_length_B, chain_B, convergent_B, divergent_B, reciprocal_B, local_efficiency_B,
         local_cost_B) = build_graph(dy_matrix,Community)

        cluster_list_wt_famr2.append(avg_clustering_W)
        cluster_list_be_famr2.append(avg_clustering_B)

        length_list_wt_famr2.append(avg_path_length_W)
        length_list_be_famr2.append(avg_path_length_B)

        chain_list_wt_famr2.append(chain_W)
        chain_list_be_famr2.append(chain_B)

        convergent_list_wt_famr2.append(convergent_W)
        convergent_list_be_famr2.append(convergent_B)

        divergent_list_wt_famr2.append(divergent_W)
        divergent_list_be_famr2.append(divergent_B)

        reciprocal_list_wt_famr2.append(reciprocal_W)
        reciprocal_list_be_famr2.append(reciprocal_B)

        local_efficiency_list_wt_famr2.append(local_efficiency_W)
        local_efficiency_list_be_famr2.append(local_efficiency_B)

        local_cost_list_wt_famr2.append(local_cost_W)
        local_cost_list_be_famr2.append(local_cost_B)

        #%%
        selected_partitions = all_results_nov['selected_partitions']
        Community = all_results_nov['community_id'][199]
        closest_number = 133

        dy_matrix = dy_list_nov[index]

        (avg_clustering_W, avg_path_length_W, chain_W, convergent_W, divergent_W, reciprocal_W, local_efficiency_W,
         local_cost_W,
         avg_clustering_B, avg_path_length_B, chain_B, convergent_B, divergent_B, reciprocal_B, local_efficiency_B,
         local_cost_B) = build_graph(dy_matrix, Community)

        cluster_list_wt_nov.append(avg_clustering_W)
        cluster_list_be_nov.append(avg_clustering_B)

        length_list_wt_nov.append(avg_path_length_W)
        length_list_be_nov.append(avg_path_length_B)

        chain_list_wt_nov.append(chain_W)
        chain_list_be_nov.append(chain_B)

        convergent_list_wt_nov.append(convergent_W)
        convergent_list_be_nov.append(convergent_B)

        divergent_list_wt_nov.append(divergent_W)
        divergent_list_be_nov.append(divergent_B)

        reciprocal_list_wt_nov.append(reciprocal_W)
        reciprocal_list_be_nov.append(reciprocal_B)

        local_efficiency_list_wt_nov.append(local_efficiency_W)
        local_efficiency_list_be_nov.append(local_efficiency_B)

        local_cost_list_wt_nov.append(local_cost_W)
        local_cost_list_be_nov.append(local_cost_B)
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]
        z = z + 1

#%%
def prepare_data_B(data_B, data_W,group_name,data_type,value1=0,value2=10,value3 = 0,value4 = 10):
    #dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))

    data_B = np.array(data_B)
    data_W = np.array(data_W)
    if data_type == 'Local cost':
        data_B = np.array(data_B)/1000
        data_W = np.array(data_W)/1000
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ f'{data_type}_B': data_B.reshape(-1, 1).flatten(),
                       f'{data_type}_W': data_W.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    df = df[df[f'{data_type}_W'] > value1]
    df = df[df[f'{data_type}_W'] < value2]
    df = df[df[f'{data_type}_B'] > value3]
    df = df[df[f'{data_type}_B'] < value4]
    return df

#%%
df1 = prepare_data_B(cluster_list_be_fam,cluster_list_wt_fam, 'Fam','Clustering coefficient')
df2 = prepare_data_B(cluster_list_be_nov,cluster_list_wt_nov, 'Nov','Clustering coefficient')
df3 = prepare_data_B(cluster_list_be_famr2,cluster_list_wt_famr2, 'Fam*','Clustering coefficient')

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = 'Clustering coefficient_W', y = 'Clustering coefficient_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.5)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel('Clustering coefficient (Within subgraph)',fontsize=15)
plt.ylabel("Clustering coefficient (Between subgraph)",fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole Clustering coefficient wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()


#%%
type = 'local_efficiency'
df1 = prepare_data_B(local_efficiency_list_be_fam,local_efficiency_list_wt_fam, 'Fam',type)
df2 = prepare_data_B(local_efficiency_list_be_nov,local_efficiency_list_wt_nov, 'Nov',type)
df3 = prepare_data_B(local_efficiency_list_be_famr2,local_efficiency_list_wt_famr2, 'Fam*',type)
df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = f'{type}_W', y = f'{type}_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.05)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel(f'{type} (Within subgraph)',fontsize=15)
plt.ylabel(f'{type} (Between subgraph)',fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + f'{type} wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()

#%%
type = 'Local cost'
df1 = prepare_data_B(local_cost_list_be_fam,local_cost_list_wt_fam, 'Fam',type)
df2 = prepare_data_B(local_cost_list_be_nov,local_cost_list_wt_nov, 'Nov',type)
df3 = prepare_data_B(local_cost_list_be_famr2,local_cost_list_wt_famr2, 'Fam*',type)

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = f'{type}_W', y = f'{type}_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.05)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel(f'{type} (Within subgraph)',fontsize=15)
plt.ylabel(f'{type} (Between subgraph)',fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + f'{type} wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()

#%%
def prepare_data_B_R(data_B, data_W,group_name,data_type,value1=0,value2=10,value3 = 0,value4 = 10):
    #dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))
    data_B = np.array(data_B)
    data_W = np.array(data_W)
    if data_type == 'Local cost':
        data_B = np.array(data_B)/1000
        data_W = np.array(data_W)/1000
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ f'{data_type}_B': data_B.reshape(-1, 1).flatten(),
                       f'{data_type}_W': data_W.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    df = df[df[f'{data_type}_W'] > value1]
    df = df[df[f'{data_type}_W'] < value2]
    df = df[df[f'{data_type}_B'] > value3]
    df = df[df[f'{data_type}_B'] < value4]
    return df

type = 'Reciprocal'
# df1 = prepare_data_B_R(reciprocal_list_be_fam,reciprocal_list_wt_fam, 'Fam',type,0.1,10,0,0.18)
# df2 = prepare_data_B_R(reciprocal_list_be_nov,reciprocal_list_wt_nov, 'Nov',type)
# df3 = prepare_data_B_R(reciprocal_list_be_famr2,reciprocal_list_wt_famr2, 'Fam*',type,0,10,0.12,10)
df1 = prepare_data_B_R(reciprocal_list_be_fam,reciprocal_list_wt_fam, 'Fam',type)
df2 = prepare_data_B_R(reciprocal_list_be_nov,reciprocal_list_wt_nov, 'Nov',type)
df3 = prepare_data_B_R(reciprocal_list_be_famr2,reciprocal_list_wt_famr2, 'Fam*',type)
df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = f'{type}_W', y = f'{type}_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.05)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel(f'{type} (Within subgraph)',fontsize=15)
plt.ylabel(f'{type} (Between subgraph)',fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + f'{type} wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()

#%%
type = 'Chain'
df1 = prepare_data_B(chain_list_be_fam,chain_list_wt_fam, 'Fam',type)
df2 = prepare_data_B(chain_list_be_nov,chain_list_wt_nov, 'Nov',type)
df3 = prepare_data_B(chain_list_be_famr2,chain_list_wt_famr2, 'Fam*',type)

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = f'{type}_W', y = f'{type}_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.05)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel(f'{type} (Within subgraph)',fontsize=15)
plt.ylabel(f'{type} (Between subgraph)',fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + f'{type} wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()

#%%
type = 'Convergent'
df1 = prepare_data_B(convergent_list_be_fam,convergent_list_wt_fam, 'Fam',type)
df2 = prepare_data_B(convergent_list_be_nov,convergent_list_wt_nov, 'Nov',type)
df3 = prepare_data_B(convergent_list_be_famr2,convergent_list_wt_famr2, 'Fam*',type)

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = f'{type}_W', y = f'{type}_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.05)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel(f'{type} (Within subgraph)',fontsize=15)
plt.ylabel(f'{type} (Between subgraph)',fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + f'{type} wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()

#%%
type = 'Divergent'
df1 = prepare_data_B(divergent_list_be_fam,divergent_list_wt_fam, 'Fam',type)
df2 = prepare_data_B(divergent_list_be_nov,divergent_list_wt_nov, 'Nov',type)
df3 = prepare_data_B(divergent_list_be_famr2,divergent_list_wt_famr2, 'Fam*',type)

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = f'{type}_W', y = f'{type}_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.05)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel(f'{type} (Within subgraph)',fontsize=15)
plt.ylabel(f'{type} (Between subgraph)',fontsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + f'{type} wt' + '.png',dpi =800)
#plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()