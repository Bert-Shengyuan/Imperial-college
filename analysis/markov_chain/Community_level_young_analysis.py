import numpy as np
import matplotlib.colors as mcolors
import random
import pickle
import matplotlib.pyplot as plt
from pygenstability.optimal_scales import identify_optimal_scales
plt.rcParams.update(plt.rcParamsDefault)
import numpy as np
import mat73
import scipy.io
import h5py
import networkx as nx
import pygenstability as pgs
import pandas as pd
import seaborn as sns
be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')
import h5py

type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')

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

#print('Attention! It is fam1')

if env == 'fam1':
    if Type == 'Young':
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_EPSP_young.pkl', 'rb') as file:
            dy_list = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_spike.pkl', 'rb') as file:
            All_spike = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_laps.pkl', 'rb') as file:
            All_laps = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
            All_mask = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
            All_df_f = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_index.pkl', 'rb') as file:
            All_spike_index = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
            All_tuning_curve = pickle.load(file)
        mouse_speed_list = be_speed_list_young_fam1
        mouse_position_list = be_phi_list_young_fam1

    elif Type == 'Old':
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_EPSP.pkl', 'rb') as file:
            dy_list = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_spike.pkl', 'rb') as file:
            All_spike =  pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_laps.pkl', 'rb') as file:
            All_laps = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
            All_mask = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
            All_df_f = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_index.pkl', 'rb') as file:
            All_spike_index = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
            All_tuning_curve = pickle.load(file)
elif env == 'nov':
    if Type == 'Young':
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_all_EPSP_young.pkl', 'rb') as file:
            dy_list = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_spike.pkl', 'rb') as file:
            All_spike = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_all_laps.pkl', 'rb') as file:
            All_laps = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
            All_mask = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_df_f.pkl', 'rb') as file:
            All_df_f = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_index.pkl', 'rb') as file:
            All_spike_index = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
            All_tuning_curve = pickle.load(file)
        mouse_speed_list = be_speed_list_young_nov
        mouse_position_list = be_phi_list_young_nov

    elif Type == 'Old':
        with open('/Users/sonmjack/Downloads/age10 result_nov/nov_all_EPSP.pkl', 'rb') as file:
            dy_list = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_nov/nov_S_spike.pkl', 'rb') as file:
            All_spike = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_nov/nov_all_laps.pkl', 'rb') as file:
            All_laps = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
            All_mask = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_nov/nov_S_df_f.pkl', 'rb') as file:
            All_df_f = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_nov/nov_S_index.pkl', 'rb') as file:
            All_spike_index = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
            All_tuning_curve = pickle.load(file)

elif env == 'fam1r2':
    if Type == 'Young':
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_all_EPSP_young.pkl', 'rb') as file:
            dy_list = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_spike.pkl', 'rb') as file:
            All_spike = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_all_laps.pkl', 'rb') as file:
            All_laps = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
            All_mask = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
            All_df_f = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_index.pkl', 'rb') as file:
            All_spike_index = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
            All_tuning_curve = pickle.load(file)
        mouse_speed_list = be_speed_list_young_fam1r2
        mouse_position_list = be_phi_list_young_fam1r2

    elif Type == 'Old':
        with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_all_EPSP.pkl', 'rb') as file:
            dy_list = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_S_spike.pkl', 'rb') as file:
            All_spike = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_all_laps.pkl', 'rb') as file:
            All_laps = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
            All_mask = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
            All_df_f = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_S_index.pkl', 'rb') as file:
            All_spike_index = pickle.load(file)
        with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
            All_tuning_curve = pickle.load(file)
        mouse_speed_list = be_speed_list_old_fam1r2
        mouse_position_list = be_phi_list_old_fam1r2

def Reverse(lst):
    new_lst = lst[::-1]
    return new_lst

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
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    k = int(N/2)

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
            random_index = np.random.randint(0, Q.shape[1]-1)-1  # 随机选择一个列索引
            Q[i, random_index] = 0.001
            Q[random_index,i] = 0.001
    return Q


def asymmetry(Q1,Q2):
    Q1 = np.array(Q1)
    Q2 = np.array(Q2)
    N1 = np.size(Q1,0)
    N2 = np.size(Q2, 0)
    Detail_Balance_Q = []
    Compone_strength_Q = []

    for i in range(N1):
        for j in range(N2):
            if (Q1[i, j] + Q2[j, i]) > 0:
                Detail_Balance_Q.append(abs(Q1[i, j] - Q2[j, i]) / (Q1[i, j] + Q2[j, i]))
    if Detail_Balance_Q == []:
        Detail_Balance_Q_mean = 0
    else:
        Detail_Balance_Q_mean = np.mean(np.array(Detail_Balance_Q))

    return Detail_Balance_Q_mean


def PRA_partition(Q, D, all_results):
    N = len(Q)
    Detail_Balance_Q = []
    for i in range(N):
        for j in range(N):
            if (i > j and (Q[i, j] + Q[j, i]) > 0):
                Detail_Balance_Q.append(abs(Q[i, j] - Q[j, i]) / (Q[i, j] + Q[j, i]))

                #######################################################
    selected_partitions = all_results['selected_partitions']
    #selected_partitions = all_results["community_id"][0::10]

    Di_Between = []
    Di_Within = []
    #     print(Repository.Reverse(selected_partitions))
    for t_opt in selected_partitions:

        B = D.copy()
        Between = D.copy()
        Within = D.copy()

        #Community = t_opt
        Community = all_results['community_id'][t_opt]
        Community = Community.tolist()

        ######## set node attrbute #################################################################################
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

        Detail_Balance_Be = []
        Detail_Balance_Wi = []

        for i in range(N):
            for j in range(N):
                if (i > j and (Be[i, j] + Be[j, i]) > 0):
                    #             print(i,j,Be[i,j])
                    Detail_Balance_Be.append(abs(Be[i, j] - Be[j, i]) / (Be[i, j] + Be[j, i]))
                if (i > j and (Wi[i, j] + Wi[j, i]) > 0):
                    #             print(i,j,Wi[i,j])`
                    Detail_Balance_Wi.append(abs(Wi[i, j] - Wi[j, i]) / (Wi[i, j] + Wi[j, i]))
                    #             print(abs(A[i,j]-A[j,i])/(A[i,j]+A[j,i])

        ###############  Normalization of Histogram ########################################
        if np.isnan(np.mean(Detail_Balance_Wi)):
            #Di_Between.append(np.mean(Detail_Balance_Be) / np.mean(Detail_Balance_Q))
            Di_Between.append(np.mean(Detail_Balance_Be))
            Di_Within.append(0.0)
        else:
            # Di_Between.append(np.mean(Detail_Balance_Be) / np.mean(Detail_Balance_Q))
            # Di_Within.append(np.mean(Detail_Balance_Wi) / np.mean(Detail_Balance_Q))
            Di_Between.append(np.mean(Detail_Balance_Be))
            Di_Within.append(np.mean(Detail_Balance_Wi))

    #         print(np.mean(Detail_Balance_Be))
    return Di_Between, Di_Within
All_c_asy_list_WT_B = []
All_c_asy_list_WT_W = []
All_c_asy_list_AD_B = []
All_c_asy_list_AD_W = []
z = 0
v = 0
for index in range(len(gene_list_young)):#len(dy_list)
    if gene_list_young[index] == 119:
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
                all_results = pickle.load(file)

        elif env =="nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
                all_results = pickle.load(file)

        elif env =="fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_'+ str(z) +'.pkl', 'rb') as file:
                all_results = pickle.load(file)
        # c_asy_list = []
        # for k in range(len(community_ids)):
        #     community = community_ids[k]
        #     size = np.max(community)
        #
        #     adj_df = pd.DataFrame(dy_matrix)
        #     labels_df = pd.DataFrame({'community': community})
        #     # 按社区标签对点进行排序
        #     sorted_indices = labels_df.sort_values(by='community').index
        #     sorted_adj_df = adj_df.iloc[sorted_indices, sorted_indices]
        #
        #     # 计算社区之间的平均链接强度
        #     unique_communities = np.unique(community)
        #     n_communities = len(unique_communities)
        #     community_strength = np.zeros((n_communities, n_communities))
        #     asy_list = []
        #     for community_i in unique_communities:
        #         for community_j in unique_communities[community_i + 1:]:
        #             members_i = sorted_indices[community[sorted_indices] == community_i]
        #             members_j = sorted_indices[community[sorted_indices] == community_j]
        #             community_matrix1 = sorted_adj_df.loc[members_i, members_j]
        #             community_matrix2 = sorted_adj_df.loc[members_j, members_i]
        #             asy = asymmetry(community_matrix1, community_matrix2)
        #             asy_list.append(asy)
        #             # community_strength[i, j] = community_matrix.mean()
        #     c_asy = np.mean(np.array(asy_list))
        #     c_asy_list.append(c_asy)
        # All_c_asy_list_WT.append(c_asy_list)
        community_ids = all_results["community_id"][0::10]
        dy_matrix = dy_list[index]
        dy_matrix = normal(dy_matrix)
        dy_matrix = sparse(dy_matrix)
        D = nx.to_networkx_graph(dy_matrix, create_using=nx.DiGraph())
        #all_results= identify_optimal_scales(all_results,kernel_size=6,window_size=6)

        Di_Between, Di_Within = PRA_partition(dy_matrix, D, all_results)
        All_c_asy_list_WT_B.append(Di_Between)
        All_c_asy_list_WT_W.append(Di_Within)
        z = z + 1
        print('Fininsh WT' + str(index))
    else:
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_AD'+ str(v) +'.pkl', 'rb') as file:
                all_results = pickle.load(file)

        elif env =="nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_AD'+ str(v) +'.pkl', 'rb') as file:
                all_results = pickle.load(file)

        elif env =="fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_AD'+ str(v) +'.pkl', 'rb') as file:
                all_results = pickle.load(file)

        dy_matrix = dy_list[index]
        dy_matrix = normal(dy_matrix)
        dy_matrix = sparse(dy_matrix)
        D = nx.to_networkx_graph(dy_matrix, create_using=nx.DiGraph())
        all_results = identify_optimal_scales(all_results,kernel_size=6,window_size=6)

        Di_Between, Di_Within = PRA_partition(dy_matrix, D, all_results)
        # c_asy_list = []
        # for k in range(len(community_ids)):
        #     community = community_ids[k]
        #     size = np.max(community)
        #
        #     adj_df = pd.DataFrame(dy_matrix)
        #     labels_df = pd.DataFrame({'community': community})
        #     # 按社区标签对点进行排序
        #     sorted_indices = labels_df.sort_values(by='community').index
        #     sorted_adj_df = adj_df.iloc[sorted_indices, sorted_indices]
        #
        #     # 计算社区之间的平均链接强度
        #     unique_communities = np.unique(community)
        #     n_communities = len(unique_communities)
        #     community_strength = np.zeros((n_communities, n_communities))
        #     asy_list = []
        #     for community_i in unique_communities:
        #         for community_j in unique_communities[community_i + 1:]:
        #             members_i = sorted_indices[community[sorted_indices] == community_i]
        #             members_j = sorted_indices[community[sorted_indices] == community_j]
        #             community_matrix1 = sorted_adj_df.loc[members_i, members_j]
        #             community_matrix2 = sorted_adj_df.loc[members_j, members_i]
        #             asy = asymmetry(community_matrix1, community_matrix2)
        #             asy_list.append(asy)
        #             # community_strength[i, j] = community_matrix.mean()
        #     c_asy = np.mean(np.array(asy_list))
        #     c_asy_list.append(c_asy)
        All_c_asy_list_AD_B.append(Di_Between)
        All_c_asy_list_AD_W.append(Di_Within)
        v = v + 1
        print('Fininsh AD'+str(index))

if env == 'fam1':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_c_asy_list_B.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_WT_B, file)
elif env == 'nov':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_c_asy_list_B.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_WT_B, file)
elif env == 'fam1r2':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_c_asy_list_B.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_WT_B, file)

if env == 'fam1':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_c_asy_list_W.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_WT_W, file)
elif env == 'nov':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_c_asy_list_W.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_WT_W, file)
elif env == 'fam1r2':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_c_asy_list_W.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_WT_W, file)

if env == 'fam1':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_AD_c_asy_list_B.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_AD_B, file)
elif env == 'nov':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_AD_c_asy_list_B.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_AD_B, file)
elif env == 'fam1r2':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_AD_c_asy_list_B.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_AD_B, file)

if env == 'fam1':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_AD_c_asy_list_W.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_AD_W, file)
elif env == 'nov':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_AD_c_asy_list_W.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_AD_W, file)
elif env == 'fam1r2':
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_AD_c_asy_list_W.pkl', 'wb') as file:
        pickle.dump(All_c_asy_list_AD_W, file)
#%%
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_c_asy_list.pkl', 'rb') as file:
        All_c_asy_list_fam1 = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_c_asy_list.pkl', 'rb') as file:
        All_c_asy_list_nov = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_c_asy_list.pkl', 'rb') as file:
        All_c_asy_list_fam1r2 = pickle.load(file)
#%%
def prepare_data(data, group_name):
    dimensions = np.tile(np.arange(-1.5,1.5, 3/20), len(data))
    df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry': data.reshape(-1,1).flatten()})
    df['Group'] = group_name
    return df
df1 = prepare_data(np.array(All_c_asy_list_fam1), 'Fam')
df2 = prepare_data(np.array(All_c_asy_list_nov), 'Nov')
df3 = prepare_data(np.array(All_c_asy_list_fam1r2), 'Fam*')

df = pd.concat([df1, df2, df3])

fig, ax = plt.subplots()
sns.lineplot(data=df, x='Scale (log10(t))',
             y='Asymmetry', hue='Group', style='Group', markers=True, dashes=False)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Wild type (age < 6)",fontsize=13)
plt.xlabel('Scale (log10(t))',fontsize=13)
plt.ylabel("Asymmetry",fontsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.tick_params(axis='x', labelsize=13)
plt.savefig('/Users/sonmjack/Downloads/figure_compare/'+'Whole asy corr WT'+'.pdf')
#plt.savefig('/Users/sonmjack/Downloads/figure_compare/'+'Whole asy corr WT'+'.svg')
plt.show()
#%%

with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_c_asy_list_W.pkl', 'rb') as file:
    All_c_asy_list_fam1_W = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_c_asy_list_W.pkl', 'rb') as file:
    All_c_asy_list_nov_W = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_c_asy_list_W.pkl', 'rb') as file:
    All_c_asy_list_fam1r2_W = pickle.load(file)
#%%
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_c_asy_list_B.pkl', 'rb') as file:
    All_c_asy_list_fam1_B = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_c_asy_list_B.pkl', 'rb') as file:
    All_c_asy_list_nov_B = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_c_asy_list_B.pkl', 'rb') as file:
    All_c_asy_list_fam1r2_B = pickle.load(file)
def prepare_data_B(data1, data2,group_name):
    dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))
    data_B = []
    data_W = []
    for row in data1:
        data_B.extend(row)
    for row in data2:
        data_W.extend(row)
    data_B = np.array(data_B)
    data_W = np.array(data_W)
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ 'Asymmetry_B': data_B.reshape(-1, 1).flatten(),
                       'Asymmetry_W': data_W.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    df = df[df['Asymmetry_W'] > 0.2]
    return df
def prepare_data_B1(data1, data2,group_name):
    dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))
    data_B = []
    data_W = []
    for row in data1:
        data_B.extend(row)
    for row in data2:
        data_W.extend(row)
    data_B = np.array(data_B)
    data_W = np.array(data_W)
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ 'Asymmetry_B': data_B.reshape(-1, 1).flatten(),
                       'Asymmetry_W': data_W.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    df = df[df['Asymmetry_W'] <= 0.56]
    return df

def prepare_data_B3(data1, data2,group_name):
    dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))
    data_B = []
    data_W = []
    for row in data1:
        data_B.extend(row)
    for row in data2:
        data_W.extend(row)
    data_B = np.array(data_B)
    data_W = np.array(data_W)
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ 'Asymmetry_B': data_B.reshape(-1, 1).flatten(),
                       'Asymmetry_W': data_W.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    df = df[df['Asymmetry_W'] >= 0.46]
    return df
# df1 = prepare_data_B(np.array(All_c_asy_list_fam1_B),np.array(All_c_asy_list_fam1_W), 'Fam')
# df2 = prepare_data_B(np.array(All_c_asy_list_nov_B),np.array(All_c_asy_list_nov_W), 'Nov')
# df3 = prepare_data_B(np.array(All_c_asy_list_fam1r2_B),np.array(All_c_asy_list_fam1r2_W), 'Fam*')

df1 = prepare_data_B1(All_c_asy_list_fam1_B,All_c_asy_list_fam1_W, 'Fam')
df2 = prepare_data_B(All_c_asy_list_nov_B,All_c_asy_list_nov_W, 'Nov')
df3 = prepare_data_B3(All_c_asy_list_fam1r2_B,All_c_asy_list_fam1r2_W, 'Fam*')

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = 'Asymmetry_W', y = 'Asymmetry_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.1)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel('Asymmetry (Within subgraph)',fontsize=18)
plt.ylabel("Asymmetry (Between subgraph)",fontsize=18)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole asy corr WT' + '.pdf')
#plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()

#%%
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_AD_c_asy_list_W.pkl', 'rb') as file:
    All_c_asy_list_fam1_W = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_AD_c_asy_list_W.pkl', 'rb') as file:
    All_c_asy_list_nov_W = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_AD_c_asy_list_W.pkl', 'rb') as file:
    All_c_asy_list_fam1r2_W = pickle.load(file)

with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1_Signal_All_AD_c_asy_list_B.pkl', 'rb') as file:
    All_c_asy_list_fam1_B = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_nov_Signal_All_AD_c_asy_list_B.pkl', 'rb') as file:
    All_c_asy_list_nov_B = pickle.load(file)
with open('/Users/sonmjack/Downloads/figure_compare/asy/Young_wild_fam1r2_Signal_All_AD_c_asy_list_B.pkl', 'rb') as file:
    All_c_asy_list_fam1r2_B = pickle.load(file)
def prepare_data_B(data1, data2,group_name):
    dimensions = np.tile(np.arange(-1.5, 1.5, 3 / 20), len(data1))
    data_B = []
    data_W = []
    for row in data1:
        data_B.extend(row)
    for row in data2:
        data_W.extend(row)
    data_B = np.array(data_B)
    data_W = np.array(data_W)
    #df = pd.DataFrame({'Scale (log10(t))': dimensions, 'Asymmetry_B': data1.reshape(-1, 1).flatten(),'Asymmetry_W': data2.reshape(-1, 1).flatten()})
    df = pd.DataFrame({ 'Asymmetry_B': data_B.reshape(-1, 1).flatten(),
                       'Asymmetry_W': data_W.reshape(-1, 1).flatten()})
    df['Group'] = group_name
    # df = df[df['Asymmetry_W'] <= 1]
    df = df[df['Asymmetry_W'] > 0]
    return df

df1 = prepare_data_B(All_c_asy_list_fam1_B,All_c_asy_list_fam1_W, 'Fam')
df2 = prepare_data_B(All_c_asy_list_nov_B,All_c_asy_list_nov_W, 'Nov')
df3 = prepare_data_B(All_c_asy_list_fam1r2_B,All_c_asy_list_fam1r2_W, 'Fam*')

df = pd.concat([df1, df2, df3])


ax = sns.jointplot(data =df , x = 'Asymmetry_W', y = 'Asymmetry_B', hue="Group")
#ax = sns.lmplot(data=df, x='Asymmetry_W', y='Asymmetry_B', hue='Group',legend= False,order =2)
ax.plot_joint(sns.kdeplot, hue='Group', zorder=0, levels=2,thresh=.1)
#plt.legend(fontsize=18, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.legend(fontsize=12, title_fontsize=12, loc='upper left')
#sns.regplot(data=df, x='Asymmetry_W',y='Asymmetry_B', order =1.5)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.title("5xFAD (age < 6)",fontsize=20)
plt.xlabel('Asymmetry (Within subgraph)',fontsize=18)
plt.ylabel("Asymmetry (Between subgraph)",fontsize=18)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole asy corr AD_J' + '.pdf')
#plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole asy corr AD' + '.svg')
plt.show()
#%%


#%%
#
#
# for i in range(len(All_c_asy_list_fam1_B)):
#     y = All_c_asy_list_fam1_B[i]
#     x = All_c_asy_list_fam1_W[i]
#     Color=[i+1 for i in range(len(x))]
#     plt.scatter(x, y, c=Color)
#
#
# for i in range(len(All_c_asy_list_nov_B)):
#     y = All_c_asy_list_nov_B[i]
#     x = All_c_asy_list_nov_W[i]
#     Color=[i+1 for i in range(len(x))]
#     plt.scatter(x, y, c=Color)
#
#
# for i in range(len(All_c_asy_list_fam1r2_B)):
#     y = All_c_asy_list_fam1r2_B[i]
#     x = All_c_asy_list_fam1r2_W[i]
#     Color=[i+1 for i in range(len(x))]
#     plt.scatter(x, y, c=Color)
# plt.show()

#%%
# def seperate_speed_neuron(spike_data,mouse_speed,threshold):
#     indices_greater_than_10 = np.where(mouse_speed > threshold)[0]
#     indices_less_than_10 = np.where(mouse_speed <= threshold)[0]
#
#     neural_data_greater = spike_data[:, indices_greater_than_10]
#     neural_data_less = spike_data[:, indices_less_than_10]
#
#     return neural_data_greater,neural_data_less
# def calculate_firing_counts(spike_data, threshold):
#     spike_data[np.where(spike_data != 0)] = 1
#     firing_counts = np.sum(spike_data >= threshold, axis=1)
#
#     return firing_counts
# #%%
# for index in range(len(be_speed_list_young_nov)):
#     if gene_list_young[index] == 119:
#
#         #change here!!!!
#         mouse_speed = mouse_speed_list[index]
#         mouse_speed[np.where(mouse_speed >= 400)] = 400
#
#         mouse_position = mouse_position_list[index]
#
#
#         spike_data = All_spike[index]
#         neuron_counts_greater, neuron_counts_less= seperate_speed_neuron(spike_data,mouse_speed,20)
#
#         firing_counts_greater = calculate_firing_counts(neuron_counts_greater,1)
#         firing_counts_less = calculate_firing_counts(neuron_counts_less,1)
#
#         sorted_indices_greater = np.argsort(firing_counts_greater)
#         sorted_indices_less = np.argsort(firing_counts_less)
#
#         sorted_firing_counts_greater = firing_counts_greater[sorted_indices_greater]
#         sorted_firing_counts_less = firing_counts_less[sorted_indices_less]
#
#         labels = [f'Neuron {i+1}' for i in range(len(firing_counts_greater))]
#         y = np.arange(len(labels))  # the label locations
#         y1= sorted_indices_less
#         y2 = sorted_indices_greater
#
#         fig, ax = plt.subplots(figsize=(6, 30))
#
#         # 绘制条形图
#         # ax.barh(y, firing_counts_greater[sorted_indices_less], height=0.4, label='Velocity > 20', color='blue', align='center')
#         # ax.barh(y, -sorted_firing_counts_less, height=0.4, label='Velocity <= 20', color='red', align='center')
#
#         ax.barh(y, sorted_firing_counts_greater, height=0.4, label='Velocity > 20', color='blue', align='center')
#         ax.barh(y, -sorted_firing_counts_less, height=0.4, label='Velocity <= 20', color='red', align='center')
#
#         # 添加标签
#         ax.set_yticks(y)
#         ax.set_yticklabels(y1)
#         ax.set_xlabel('Firing Counts')
#         ax.set_title('Pyramidal Chart of Neuron Firing Counts')
#         ax.legend()
#
#         #
#         # ax2 = ax.twinx()
#         # ax2.set_yticks(y)
#         # ax2.set_yticklabels(y2)
#         # ax2.set_ylim(ax.get_ylim())
#         # 显示图表
#         # plt.savefig('/Users/sonmjack/Downloads/figure_compare/speed/' + str(index) + '_' + env + '_' + Type + '_Speed tune neuron''.jpg')
#         plt.savefig('/Users/sonmjack/Downloads/figure_compare/speed/'+str(index)+'_reindex'+env+'_'+Type+'_Speed tune neuron''.jpg')
#         plt.close()
# #%%
# import seaborn as sns
# plt.close()
# plt.figure(figsize=(10, 6))
# sns.heatmap(All_tuning_curve[0], cmap='viridis')
# plt.show()
# #%%
# fig, axes = plt.subplots(2, 1, figsize=(27, 6), gridspec_kw={'height_ratios': [3, 3]})
# axes[0].plot(mouse_position)
# axes[1].plot(mouse_speed)
# # axes[0].plot(mouse_position[int(spike_data.shape[1]/3):int(spike_data.shape[1]*2/3)])
# # axes[1].plot(mouse_speed[int(spike_data.shape[1]/3):int(spike_data.shape[1]*2/3)])
# plt.show()

