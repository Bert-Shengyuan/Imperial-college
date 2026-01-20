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

        be_speed_list_young_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
        be_speed_list_young_nov.append(be_speed_sum_nov[int(i / 2), 0])
        be_speed_list_young_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

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

be_speed_list_old_fam1 = []
be_speed_list_old_nov = []
be_speed_list_old_fam1r2 = []

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

    be_speed_list_old_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
    be_speed_list_old_nov.append(be_speed_sum_nov[int(i / 2), 0])
    be_speed_list_old_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

    gene_list_old.append(mat_trigger[i, 1])
del be_data, be_phi_sum_fam1, be_phi_sum_nov, be_phi_sum_fam1r2
#%%
with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
    mask = pickle.load(file)

env = 'nov'
#env = 'fam1'
#env = 'fam1r2'
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
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)
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
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)
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
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)

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


def sparse(A):
    N = A.shape[0]
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    k = int(N / 2)

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
            # random_index = np.random.randint(0, Q.shape[1]-1)-1  # 随机选择一个列索引
            Q[i, i - 1] = 0.001
            Q[i - 1, i] = 0.001
    return Q


import plotly.graph_objects as go  # pragma: no cover
from plotly.offline import plot


def plot_sankey(
        all_results,
        optimal_scales=True,
        live=False,
        filename='',
        scale_index=None,
):  # pragma: no cover
    """Plot Sankey diagram of communities accros scale (plotly only).
    Args:
        all_results (dict): results from run function
        optimal_scales (bool): use optimal scales or not
        live (bool): if True, interactive figure will appear in browser
        filename (str): filename to save the plot
        scale_index (bool): plot scale of indices
    """
    sources = []
    targets = []
    values = []
    shift = 0

    # if not scale_index:
    all_results["community_id_reduced"] = all_results["community_id"][0::10]

    community_ids = all_results["community_id_reduced"]

    for i in range(len(community_ids) - 1):
        community_source = np.array(community_ids[i])
        community_target = np.array(community_ids[i + 1])
        source_ids = set(community_source)
        target_ids = set(community_target)
        #         print(target_ids)
        for source in source_ids:
            for target in target_ids:
                value = sum(community_target[community_source == source] == target)
                #                 print(community_target[community_source == source] == target)
                if value > 0:
                    values.append(value)
                    sources.append(source + shift)
                    targets.append(target + len(source_ids) + shift)
        shift += len(source_ids)

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 1,
                    "thickness": 1,
                    "line": {"color": "black", "width": 0.0},
                },
                link={"source": sources, "target": targets, "value": values},
            )
        ],
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        paper_bgcolor='white',  # 设置图表的背景颜色为白色
        plot_bgcolor='white')

    Scale_enumerate = [r'$S_{%s}$' % (len(community_ids) - i) for i in range(len(community_ids))]
    for x_coordinate, column_name in enumerate(Scale_enumerate):
        fig.add_annotation(
            x=x_coordinate,  # Plotly recognizes 0-5 to be the x range.

            y=1.075,  # y value above 1 means above all nodes
            xref="x",
            yref="paper",
            text=column_name,  # Text
            showarrow=False,
            font=dict(
                family="Tahoma",
                size=16,
                color="black"
            ),
            align="left",
        )
    fig.write_image(filename)
#%%
cluster_number = []
Color_Code2 = ['white', 'darkblue', 'red', "orange", "pink", "olive", "cyan", "yellow", "green", "brown", "gray",
               "tomato", 'limegreen']

list_color = []
for color in mcolors.CSS4_COLORS:
    list_color.append(color)
random.shuffle(list_color)
Color_Code1 = [*Color_Code2, *list_color]
Color_Code = [*Color_Code1, *list_color]
z = 5
v = 0
Time_list_matrix = []
for index in range(9,10):  # len(dy_list)
    plt.close()
    if gene_list_young[index] == 119:
        Spike_train = neuron_spike[index][:,
                      int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)]
        mask_index = mask[index]
        Time_train = neuron_spike[index]
        tuning_curve = All_tuning_curve[index]
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_markov.pdf')
            plt.close()

        elif env == "nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_markov.pdf')
            plt.close()
        elif env == "fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_' + str(z) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.close()

        if env == 'fam1' or env == 'fam1r2':
            closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            Community = all_results['community_id'][closest_number]

        elif env == "nov":
            selected_partitions = all_results['selected_partitions']
            Community = all_results['community_id'][199]
            closest_number = 133
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]

        non_zero_indices_per_row = []
        for row in Spike_train:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))

        neuralData = []
        ColorCode = []
        C_tuning_curve = []
        it = 0

        tuning_curve_df = pd.DataFrame(tuning_curve)
        labels_df = pd.DataFrame({'community': Community})
        time_list_df = pd.DataFrame(Time_train)

        # 按社区标签对神经元进行排序
        sorted_indices = labels_df.sort_values(by='community').index
        sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]

        sorted_time_list_df = time_list_df.iloc[sorted_indices]
        # 按community分组
        grouped1 = sorted_tuning_curve_df.groupby(labels_df['community'])

        community_max_indices = {}
        for community, group in grouped1:
            mean_tuning_curve = group.mean(axis=0)
            max_index = np.argmax(mean_tuning_curve)
            community_max_indices[community] = max_index

        # 根据平均tuning curve最大值的index对community进行排序
        sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

        # 根据排序后的community重新排列神经元
        sorted_tuning_curve_list = []
        time_new_list = []
        for community in sorted_communities:
            sorted_tuning_curve_list.append(tuning_curve_df[labels_df['community'] == community])
            time_new_list.append(time_list_df[labels_df['community'] == community])
            sorted_tuning_curve_list.append(pd.DataFrame(np.full((3, tuning_curve.shape[1]), np.nan)))
        # 将所有community的tuning curve拼接在一起

        C_tuning_curve = pd.concat(sorted_tuning_curve_list, axis=0).values
        Time_list_matrix.append(time_new_list)
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [6, 3]})
        sns.heatmap(C_tuning_curve, cmap='viridis', vmax=1, cbar=False, ax=axes[0])
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)
        if env == 'fam1':
            angle_fam1 = be_phi_list_young_fam1[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_tuning.pdf')
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_tuning.png')
            plt.close()
        elif env == "nov":
            angle_nov = be_phi_list_young_nov[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_nov, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_tuning.pdf')
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_tuning.png')
            plt.close()
        elif env == "fam1r2":
            angle_fam1r2 = be_phi_list_young_fam1r2[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1r2, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels, rotation=0)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(index) + '_young_fam1r2_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(index) + '_young_fam1r2_tuning.png')
            plt.close()

        Nodes = range(len(non_zero_indices_per_row))
        for m, j in zip(Nodes, Nodes):
            for i in range(max(Community) + 1):
                if Community[j] == i:
                    non_zero_indices_per_row[int(m)].sort()
                    neuralData.append(non_zero_indices_per_row[int(m)])
                    ColorCode.append(Color_Code[i + 1])
                    # A_ordered_row[it,:]=Q[j,:]
                    it += 1
        # it=0
        # for i in range(max(Community)+1):
        #     for m,j in zip(Nodes,range(N)):
        #         if Community[j]==i:
        #             A_ordered[:,it]=A_ordered_row[:,j]
        #             it+=1
        fig, axes = plt.subplots(2, 1, figsize=(27, 9), gridspec_kw={'height_ratios': [6, 3]})
        axes[0].eventplot(neuralData, color=ColorCode)
        axes[0].invert_yaxis()

        axes[0].set_title("Reordered neural spike trains " + f'N-{max(Community) + 1}',fontsize=20)
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)


        # max_in_sublists = [max(sublist) for sublist in neuralData if sublist]
        #
        # # 找到最大值中的最大值
        # overall_max = max(max_in_sublists)
        # nearest_5000_multiple = round(overall_max / 10000) * 10000
        # axes[0].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
        # plt.show()

        if env == 'fam1':
            axes[1].plot(be_phi_list_young_fam1[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_list.png')
            plt.close()
        elif env == "nov":
            axes[1].plot(be_phi_list_young_nov[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_list.png')
            plt.close()
        elif env == "fam1r2":
            axes[1].plot(be_phi_list_young_fam1r2[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/neuron_list_' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/neuron_list_' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        from matplotlib.colors import BoundaryNorm

        community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

        # 更新labels_df中的community标签
        labels_df['community'] = labels_df['community'].map(community_mapping)
        Sort_community = labels_df['community'].values
        pic = np.zeros((512, 512))
        for i in range(mask_index.shape[2]):
            index_row = np.where(mask_index[:, :, i] == True)[0]
            index_con = np.where(mask_index[:, :, i] == True)[1]
            for j in range(len(index_row)):
                pic[index_row[j], index_con[j]] = Sort_community[i]
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        import seaborn as sns

        #cmap = plt.get_cmap('tab10')

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
        bounds = np.linspace(-0.5, 11.5, 13)
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(pic, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Scale= {closest_number*1.5/200-0.5:.2f}, N='+str(max(Community) + 1),fontsize=20)
        plt.tight_layout()
        # plt.title('Community beasd on real data')
        # plt.show()
        if env == 'fam1':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()

        elif env == "nov":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()
        elif env == "fam1r2":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.png')
            plt.close()

        if env == 'fam1':
            name = '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_sankey.png'
        elif env == "nov":
            name = '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_snakey.png'
        elif env == "fam1r2":
            name = '/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/snakey' + str(
                index) + '_young_fam1r2.png'

        plot_sankey(
            all_results,
            optimal_scales=True,
            live=False,
            filename=name,
            scale_index=None,
        )

        z = z + 1
    else:
        Spike_train = neuron_spike[index][:,
                      int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)]
        mask_index = mask[index]
        tuning_curve = All_tuning_curve[index]
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_AD' + str(v) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov.png')
            plt.close()

        elif env == "nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_AD' + str(v) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov.png')
            plt.close()
        elif env == "fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_AD' + str(v) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        if env == 'fam1' or env == 'fam1r2':
            closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            Community = all_results['community_id'][closest_number]

        elif env == "nov":
            selected_partitions = all_results['selected_partitions']
            Community = all_results['community_id'][199]
            closest_number = 133
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]

        non_zero_indices_per_row = []
        for row in Spike_train:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))

        neuralData = []
        ColorCode = []
        C_tuning_curve = []
        it = 0

        tuning_curve_df = pd.DataFrame(tuning_curve)
        labels_df = pd.DataFrame({'community': Community})
        time_list_df = pd.DataFrame(Time_train)

        # 按社区标签对神经元进行排序
        sorted_indices = labels_df.sort_values(by='community').index
        sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]

        sorted_time_list_df = time_list_df.iloc[sorted_indices]
        # 按community分组
        grouped1 = sorted_tuning_curve_df.groupby(labels_df['community'])

        community_max_indices = {}
        for community, group in grouped1:
            mean_tuning_curve = group.mean(axis=0)
            max_index = np.argmax(mean_tuning_curve)
            community_max_indices[community] = max_index

        # 根据平均tuning curve最大值的index对community进行排序
        sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

        # 根据排序后的community重新排列神经元
        sorted_tuning_curve_list = []
        time_new_list = []
        for community in sorted_communities:
            sorted_tuning_curve_list.append(tuning_curve_df[labels_df['community'] == community])
            time_new_list.append(time_list_df[labels_df['community'] == community])
            sorted_tuning_curve_list.append(pd.DataFrame(np.full((3, tuning_curve.shape[1]), np.nan)))
        # 将所有community的tuning curve拼接在一起

        C_tuning_curve = pd.concat(sorted_tuning_curve_list, axis=0).values

        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [6, 3]})
        sns.heatmap(C_tuning_curve, cmap='viridis', vmax=1, cbar=False, ax=axes[0])
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)
        if env == 'fam1':
            angle_fam1 = be_phi_list_young_fam1[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_tuning.png')
            plt.close()
        elif env == "nov":
            angle_nov = be_phi_list_young_nov[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_nov, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_tuning.png')
            plt.close()
        elif env == "fam1r2":
            angle_fam1r2 = be_phi_list_young_fam1r2[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1r2, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels, rotation=0)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_tuning.png')
            plt.close()

        Nodes = range(len(non_zero_indices_per_row))
        for m, j in zip(Nodes, Nodes):
            for i in range(max(Community) + 1):
                if Community[j] == i:
                    non_zero_indices_per_row[int(m)].sort()
                    neuralData.append(non_zero_indices_per_row[int(m)])
                    ColorCode.append(Color_Code[i + 1])
                    # A_ordered_row[it,:]=Q[j,:]
                    it += 1
        # it=0
        # for i in range(max(Community)+1):
        #     for m,j in zip(Nodes,range(N)):
        #         if Community[j]==i:
        #             A_ordered[:,it]=A_ordered_row[:,j]
        #             it+=1
        fig, axes = plt.subplots(2, 1, figsize=(27, 9), gridspec_kw={'height_ratios': [6, 3]})
        axes[0].eventplot(neuralData, color=ColorCode)
        axes[0].invert_yaxis()

        axes[0].set_title("Reordered neural spike trains " + f'N-{max(Community) + 1}',fontsize=20)
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)

        # max_in_sublists = [max(sublist) for sublist in neuralData if sublist]
        #
        # # 找到最大值中的最大值
        # overall_max = max(max_in_sublists)
        # nearest_5000_multiple = round(overall_max / 10000) * 10000
        # axes[0].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
        # plt.show()

        if env == 'fam1':
            axes[1].plot(be_phi_list_young_fam1[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_list.png')
            plt.close()
        elif env == "nov":
            axes[1].plot(be_phi_list_young_nov[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_list.png')
            plt.close()
        elif env == "fam1r2":
            axes[1].plot(be_phi_list_young_fam1r2[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/neuron_list_' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/neuron_list_' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        from matplotlib.colors import BoundaryNorm

        community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

        # 更新labels_df中的community标签
        labels_df['community'] = labels_df['community'].map(community_mapping)
        Sort_community = labels_df['community'].values

        pic = np.zeros((512, 512))
        for i in range(mask_index.shape[2]):
            index_row = np.where(mask_index[:, :, i] == True)[0]
            index_con = np.where(mask_index[:, :, i] == True)[1]
            for j in range(len(index_row)):
                pic[index_row[j], index_con[j]] = Sort_community[i]
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        import seaborn as sns

        #cmap = plt.get_cmap('tab10')

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
        bounds = np.linspace(-0.5, 11.5, 13)
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(pic, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Scale= {closest_number*1.5/200-0.5:.2f}, N='+str(max(Community) + 1),fontsize=20)
        plt.tight_layout()
        plt.title('Community beasd on real data')
        # plt.show()
        if env == 'fam1':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()

        elif env == "nov":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_mask.png')
            plt.close()
        elif env == "fam1r2":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.png')
            plt.close()

        if env == 'fam1':
            name = '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_sankey.png'
        elif env == "nov":
            name = '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_snakey.png'
        elif env == "fam1r2":
            name = '/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/snakey' + str(
                index) + '_young_fam1r2.png'

        plot_sankey(
            all_results,
            optimal_scales=True,
            live=False,
            filename=name,
            scale_index=None,
        )

        v = v + 1

#%%
from cebra import CEBRA

max_iterations = 5000#default is 5000.
output_dimension = 32 #here, we set as a variable for hypothesis testing below.
neu_data = np.array(Time_list_matrix[0][0]).T+0.01
#neu_data = neu_data[0:22200,:]

be_phi = be_phi_list_young_nov[9]#[0:22200,:]
bin_edges = np.linspace(0, 360, 11)  # 11 edges for 10 bins
# Create labels for each bin
labels = np.arange(1, 11)  # Laimport numpy as np
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

        be_speed_list_young_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
        be_speed_list_young_nov.append(be_speed_sum_nov[int(i / 2), 0])
        be_speed_list_young_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

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

be_speed_list_old_fam1 = []
be_speed_list_old_nov = []
be_speed_list_old_fam1r2 = []

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

    be_speed_list_old_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
    be_speed_list_old_nov.append(be_speed_sum_nov[int(i / 2), 0])
    be_speed_list_old_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

    gene_list_old.append(mat_trigger[i, 1])
del be_data, be_phi_sum_fam1, be_phi_sum_nov, be_phi_sum_fam1r2
#%%
with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
    mask = pickle.load(file)

env = 'nov'
#env = 'fam1'
#env = 'fam1r2'
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
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)
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
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)
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
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)

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


def sparse(A):
    N = A.shape[0]
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    k = int(N / 2)

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
            # random_index = np.random.randint(0, Q.shape[1]-1)-1  # 随机选择一个列索引
            Q[i, i - 1] = 0.001
            Q[i - 1, i] = 0.001
    return Q


import plotly.graph_objects as go  # pragma: no cover
from plotly.offline import plot


def plot_sankey(
        all_results,
        optimal_scales=True,
        live=False,
        filename='',
        scale_index=None,
):  # pragma: no cover
    """Plot Sankey diagram of communities accros scale (plotly only).
    Args:
        all_results (dict): results from run function
        optimal_scales (bool): use optimal scales or not
        live (bool): if True, interactive figure will appear in browser
        filename (str): filename to save the plot
        scale_index (bool): plot scale of indices
    """
    sources = []
    targets = []
    values = []
    shift = 0

    # if not scale_index:
    all_results["community_id_reduced"] = all_results["community_id"][0::10]

    community_ids = all_results["community_id_reduced"]

    for i in range(len(community_ids) - 1):
        community_source = np.array(community_ids[i])
        community_target = np.array(community_ids[i + 1])
        source_ids = set(community_source)
        target_ids = set(community_target)
        #         print(target_ids)
        for source in source_ids:
            for target in target_ids:
                value = sum(community_target[community_source == source] == target)
                #                 print(community_target[community_source == source] == target)
                if value > 0:
                    values.append(value)
                    sources.append(source + shift)
                    targets.append(target + len(source_ids) + shift)
        shift += len(source_ids)

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 1,
                    "thickness": 1,
                    "line": {"color": "black", "width": 0.0},
                },
                link={"source": sources, "target": targets, "value": values},
            )
        ],
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        paper_bgcolor='white',  # 设置图表的背景颜色为白色
        plot_bgcolor='white')

    Scale_enumerate = [r'$S_{%s}$' % (len(community_ids) - i) for i in range(len(community_ids))]
    for x_coordinate, column_name in enumerate(Scale_enumerate):
        fig.add_annotation(
            x=x_coordinate,  # Plotly recognizes 0-5 to be the x range.

            y=1.075,  # y value above 1 means above all nodes
            xref="x",
            yref="paper",
            text=column_name,  # Text
            showarrow=False,
            font=dict(
                family="Tahoma",
                size=16,
                color="black"
            ),
            align="left",
        )
    fig.write_image(filename)
#%%
cluster_number = []
Color_Code2 = ['white', 'darkblue', 'red', "orange", "pink", "olive", "cyan", "yellow", "green", "brown", "gray",
               "tomato", 'limegreen']

list_color = []
for color in mcolors.CSS4_COLORS:
    list_color.append(color)
random.shuffle(list_color)
Color_Code1 = [*Color_Code2, *list_color]
Color_Code = [*Color_Code1, *list_color]
z = 5
v = 0
Time_list_matrix = []
for index in range(9,10):  # len(dy_list)
    plt.close()
    if gene_list_young[index] == 119:
        Spike_train = neuron_spike[index][:,
                      int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)]
        mask_index = mask[index]
        Time_train = neuron_spike[index]
        tuning_curve = All_tuning_curve[index]
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_markov.pdf')
            plt.close()

        elif env == "nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_markov.pdf')
            plt.close()
        elif env == "fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_' + str(z) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.close()

        if env == 'fam1' or env == 'fam1r2':
            closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            Community = all_results['community_id'][closest_number]

        elif env == "nov":
            selected_partitions = all_results['selected_partitions']
            Community = all_results['community_id'][199]
            closest_number = 133
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]

        non_zero_indices_per_row = []
        for row in Spike_train:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))

        neuralData = []
        ColorCode = []
        C_tuning_curve = []
        it = 0

        tuning_curve_df = pd.DataFrame(tuning_curve)
        labels_df = pd.DataFrame({'community': Community})
        time_list_df = pd.DataFrame(Time_train)

        # 按社区标签对神经元进行排序
        sorted_indices = labels_df.sort_values(by='community').index
        sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]

        sorted_time_list_df = time_list_df.iloc[sorted_indices]
        # 按community分组
        grouped1 = sorted_tuning_curve_df.groupby(labels_df['community'])

        community_max_indices = {}
        for community, group in grouped1:
            mean_tuning_curve = group.mean(axis=0)
            max_index = np.argmax(mean_tuning_curve)
            community_max_indices[community] = max_index

        # 根据平均tuning curve最大值的index对community进行排序
        sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

        # 根据排序后的community重新排列神经元
        sorted_tuning_curve_list = []
        time_new_list = []
        for community in sorted_communities:
            sorted_tuning_curve_list.append(tuning_curve_df[labels_df['community'] == community])
            time_new_list.append(time_list_df[labels_df['community'] == community])
            sorted_tuning_curve_list.append(pd.DataFrame(np.full((3, tuning_curve.shape[1]), np.nan)))
        # 将所有community的tuning curve拼接在一起

        C_tuning_curve = pd.concat(sorted_tuning_curve_list, axis=0).values
        Time_list_matrix.append(time_new_list)
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [6, 3]})
        sns.heatmap(C_tuning_curve, cmap='viridis', vmax=1, cbar=False, ax=axes[0])
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)
        if env == 'fam1':
            angle_fam1 = be_phi_list_young_fam1[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_tuning.pdf')
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_tuning.png')
            plt.close()
        elif env == "nov":
            angle_nov = be_phi_list_young_nov[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_nov, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_tuning.pdf')
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_tuning.png')
            plt.close()
        elif env == "fam1r2":
            angle_fam1r2 = be_phi_list_young_fam1r2[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1r2, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels, rotation=0)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(index) + '_young_fam1r2_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(index) + '_young_fam1r2_tuning.png')
            plt.close()

        Nodes = range(len(non_zero_indices_per_row))
        for m, j in zip(Nodes, Nodes):
            for i in range(max(Community) + 1):
                if Community[j] == i:
                    non_zero_indices_per_row[int(m)].sort()
                    neuralData.append(non_zero_indices_per_row[int(m)])
                    ColorCode.append(Color_Code[i + 1])
                    # A_ordered_row[it,:]=Q[j,:]
                    it += 1
        # it=0
        # for i in range(max(Community)+1):
        #     for m,j in zip(Nodes,range(N)):
        #         if Community[j]==i:
        #             A_ordered[:,it]=A_ordered_row[:,j]
        #             it+=1
        fig, axes = plt.subplots(2, 1, figsize=(27, 9), gridspec_kw={'height_ratios': [6, 3]})
        axes[0].eventplot(neuralData, color=ColorCode)
        axes[0].invert_yaxis()

        axes[0].set_title("Reordered neural spike trains " + f'N-{max(Community) + 1}',fontsize=20)
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)


        # max_in_sublists = [max(sublist) for sublist in neuralData if sublist]
        #
        # # 找到最大值中的最大值
        # overall_max = max(max_in_sublists)
        # nearest_5000_multiple = round(overall_max / 10000) * 10000
        # axes[0].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
        # plt.show()

        if env == 'fam1':
            axes[1].plot(be_phi_list_young_fam1[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_list.png')
            plt.close()
        elif env == "nov":
            axes[1].plot(be_phi_list_young_nov[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_list.png')
            plt.close()
        elif env == "fam1r2":
            axes[1].plot(be_phi_list_young_fam1r2[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/neuron_list_' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/neuron_list_' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        from matplotlib.colors import BoundaryNorm

        community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

        # 更新labels_df中的community标签
        labels_df['community'] = labels_df['community'].map(community_mapping)
        Sort_community = labels_df['community'].values
        pic = np.zeros((512, 512))
        for i in range(mask_index.shape[2]):
            index_row = np.where(mask_index[:, :, i] == True)[0]
            index_con = np.where(mask_index[:, :, i] == True)[1]
            for j in range(len(index_row)):
                pic[index_row[j], index_con[j]] = Sort_community[i]
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        import seaborn as sns

        #cmap = plt.get_cmap('tab10')

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
        bounds = np.linspace(-0.5, 11.5, 13)
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(pic, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Scale= {closest_number*1.5/200-0.5:.2f}, N='+str(max(Community) + 1),fontsize=20)
        plt.tight_layout()
        # plt.title('Community beasd on real data')
        # plt.show()
        if env == 'fam1':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()

        elif env == "nov":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()
        elif env == "fam1r2":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.png')
            plt.close()

        if env == 'fam1':
            name = '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_sankey.png'
        elif env == "nov":
            name = '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_snakey.png'
        elif env == "fam1r2":
            name = '/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/snakey' + str(
                index) + '_young_fam1r2.png'

        plot_sankey(
            all_results,
            optimal_scales=True,
            live=False,
            filename=name,
            scale_index=None,
        )

        z = z + 1
    else:
        Spike_train = neuron_spike[index][:,
                      int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)]
        mask_index = mask[index]
        tuning_curve = All_tuning_curve[index]
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_AD' + str(v) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov.png')
            plt.close()

        elif env == "nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_AD' + str(v) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov.png')
            plt.close()
        elif env == "fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_AD' + str(v) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        if env == 'fam1' or env == 'fam1r2':
            closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            Community = all_results['community_id'][closest_number]

        elif env == "nov":
            selected_partitions = all_results['selected_partitions']
            Community = all_results['community_id'][199]
            closest_number = 133
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]

        non_zero_indices_per_row = []
        for row in Spike_train:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))

        neuralData = []
        ColorCode = []
        C_tuning_curve = []
        it = 0

        tuning_curve_df = pd.DataFrame(tuning_curve)
        labels_df = pd.DataFrame({'community': Community})
        time_list_df = pd.DataFrame(Time_train)

        # 按社区标签对神经元进行排序
        sorted_indices = labels_df.sort_values(by='community').index
        sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]

        sorted_time_list_df = time_list_df.iloc[sorted_indices]
        # 按community分组
        grouped1 = sorted_tuning_curve_df.groupby(labels_df['community'])

        community_max_indices = {}
        for community, group in grouped1:
            mean_tuning_curve = group.mean(axis=0)
            max_index = np.argmax(mean_tuning_curve)
            community_max_indices[community] = max_index

        # 根据平均tuning curve最大值的index对community进行排序
        sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

        # 根据排序后的community重新排列神经元
        sorted_tuning_curve_list = []
        time_new_list = []
        for community in sorted_communities:
            sorted_tuning_curve_list.append(tuning_curve_df[labels_df['community'] == community])
            time_new_list.append(time_list_df[labels_df['community'] == community])
            sorted_tuning_curve_list.append(pd.DataFrame(np.full((3, tuning_curve.shape[1]), np.nan)))
        # 将所有community的tuning curve拼接在一起

        C_tuning_curve = pd.concat(sorted_tuning_curve_list, axis=0).values

        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [6, 3]})
        sns.heatmap(C_tuning_curve, cmap='viridis', vmax=1, cbar=False, ax=axes[0])
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)
        if env == 'fam1':
            angle_fam1 = be_phi_list_young_fam1[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_tuning.png')
            plt.close()
        elif env == "nov":
            angle_nov = be_phi_list_young_nov[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_nov, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_tuning.png')
            plt.close()
        elif env == "fam1r2":
            angle_fam1r2 = be_phi_list_young_fam1r2[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1r2, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels, rotation=0)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_tuning.png')
            plt.close()

        Nodes = range(len(non_zero_indices_per_row))
        for m, j in zip(Nodes, Nodes):
            for i in range(max(Community) + 1):
                if Community[j] == i:
                    non_zero_indices_per_row[int(m)].sort()
                    neuralData.append(non_zero_indices_per_row[int(m)])
                    ColorCode.append(Color_Code[i + 1])
                    # A_ordered_row[it,:]=Q[j,:]
                    it += 1
        # it=0
        # for i in range(max(Community)+1):
        #     for m,j in zip(Nodes,range(N)):
        #         if Community[j]==i:
        #             A_ordered[:,it]=A_ordered_row[:,j]
        #             it+=1
        fig, axes = plt.subplots(2, 1, figsize=(27, 9), gridspec_kw={'height_ratios': [6, 3]})
        axes[0].eventplot(neuralData, color=ColorCode)
        axes[0].invert_yaxis()

        axes[0].set_title("Reordered neural spike trains " + f'N-{max(Community) + 1}',fontsize=20)
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)

        # max_in_sublists = [max(sublist) for sublist in neuralData if sublist]
        #
        # # 找到最大值中的最大值
        # overall_max = max(max_in_sublists)
        # nearest_5000_multiple = round(overall_max / 10000) * 10000
        # axes[0].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
        # plt.show()

        if env == 'fam1':
            axes[1].plot(be_phi_list_young_fam1[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_list.png')
            plt.close()
        elif env == "nov":
            axes[1].plot(be_phi_list_young_nov[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_list.png')
            plt.close()
        elif env == "fam1r2":
            axes[1].plot(be_phi_list_young_fam1r2[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/neuron_list_' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/neuron_list_' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        from matplotlib.colors import BoundaryNorm

        community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

        # 更新labels_df中的community标签
        labels_df['community'] = labels_df['community'].map(community_mapping)
        Sort_community = labels_df['community'].values

        pic = np.zeros((512, 512))
        for i in range(mask_index.shape[2]):
            index_row = np.where(mask_index[:, :, i] == True)[0]
            index_con = np.where(mask_index[:, :, i] == True)[1]
            for j in range(len(index_row)):
                pic[index_row[j], index_con[j]] = Sort_community[i]
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        import seaborn as sns

        #cmap = plt.get_cmap('tab10')

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
        bounds = np.linspace(-0.5, 11.5, 13)
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(pic, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Scale= {closest_number*1.5/200-0.5:.2f}, N='+str(max(Community) + 1),fontsize=20)
        plt.tight_layout()
        plt.title('Community beasd on real data')
        # plt.show()
        if env == 'fam1':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()

        elif env == "nov":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_mask.png')
            plt.close()
        elif env == "fam1r2":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.png')
            plt.close()

        if env == 'fam1':
            name = '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_sankey.png'
        elif env == "nov":
            name = '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_snakey.png'
        elif env == "fam1r2":
            name = '/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/snakey' + str(
                index) + '_young_fam1r2.png'

        plot_sankey(
            all_results,
            optimal_scales=True,
            live=False,
            filename=name,
            scale_index=None,
        )

        v = v + 1

#%%
from cebra import CEBRA

max_iterations = 5000#default is 5000.
output_dimension = 32 #here, we set as a variable for hypothesis testing below.
neu_data = np.array(Time_list_matrix[0][0]).T+0.01
#neu_data = neu_data[0:22200,:]

be_phi = be_phi_list_young_nov[9]#[0:22200,:]
bin_edges = np.linspace(0, 360, 11)  # 11 edges for 10 bins
# Create labels for each bin
labels = np.arange(1, 11)  # Labels from 1 to 10
# Assign labels to each element based on the bin it falls into
label_vector = np.digitize(be_phi, bin_edges)
# Map the labels to the desired range (1 to 10)
label_vector2 = labels[label_vector - 1]
#%%
be_2D = np.column_stack([be_x_list_young_nov[9], be_y_list_young_nov[9]])#[0:22200,:]

# plt.figure(figsize=(20, 4))
# #plt.plot(neuron_spike[9][1,:]+0.1)
# plt.plot(Time_train[1,:])
# plt.show()


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
                        time_offsets=10)
                        #hybrid = True)# hybrid = True

#cebra_hybrid_model.fit(neu_data, be_2D)
cebra_hybrid_model.fit(neu_data, label_vector2)
cebra_hybrid = cebra_hybrid_model.transform(neu_data)

#%%

def plot_hippocampus(ax, embedding, be_phi, elev, azim,roll,gray = False, idx_order = (0,1,2),xlabel='X', ylabel='Y', zlabel='Z'):
    p = ax.scatter(embedding [:, 0], embedding [:, 1], embedding [:, 2],s=5,c=be_phi,cmap ='viridis')
    #ax.plot(embedding[:, 0], embedding[:, 1], embedding [:, 2],c='gray', linewidth=0.5)

    ax.grid(False)
    ax.set_box_aspect([np.ptp(i) for i in embedding.T])
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.view_init(elev=elev, azim=azim,roll=roll)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return p,ax

#matplotlib notebook
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,20))

# ax1 = plt.subplot(141, projection='3d')
# ax2 = plt.subplot(142, projection='3d')
# ax3 = plt.subplot(143, projection='3d')
ax4 = plt.subplot(111, projection='3d')

# ax1=plot_hippocampus(ax1, cebra_posdir3, hippocampus_pos.continuous_index)
# ax2=plot_hippocampus(ax2, cebra_posdir_shuffled3, hippocampus_pos.continuous_index)
# ax3=plot_hippocampus(ax3, cebra_time3, hippocampus_pos.continuous_index)
p, ax4= plot_hippocampus(ax4, cebra_hybrid, be_phi, elev=0, azim=45, roll=0)
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_phi)
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D[:,0])
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D[:,1])
# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax4.set_title('CEBRA-Hybrid')
fig.colorbar(p)
plt.show()

#%%
import cebra.datasets
hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
Te_b = hippocampus_pos.continuous_index.numpy()
def split_data(data, test_ratio):

    split_idx = int(len(data)* (1-test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]
    label_train = data.continuous_index[:split_idx]
    label_test = data.continuous_index[split_idx:]

    return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()

neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2)

#%%
monkey_pos = cebra.datasets.init('area2-bump-pos-active')
Te_M = monkey_pos.continuous_index.numpy()
#%%
monkey_target = cebra.datasets.init('area2-bump-target-active')
Te_D= monkey_target.discrete_index.numpy()
TE_neural = monkey_target.neural.numpy()

#%%
plt.figure(figsize=(20, 4))
plt.plot(TE_neural[:,1])
plt.show()

#%%
direction_trial = (monkey_target.discrete_index == 1).numpy()
cebra_target_model = CEBRA(model_architecture='offset10-model',
                           batch_size=512,
                           learning_rate=0.0001,
                           temperature=1,
                           output_dimension=3,
                           max_iterations=max_iterations,
                           distance='cosine',
                           conditional='time_delta',
                           device='cuda_if_available',
                           verbose=True,
                           time_offsets=10)
cebra_target_model.fit(monkey_target.neural,
                       monkey_target.discrete_index.numpy())
cebra_target = cebra_target_model.transform(monkey_target.neural)
#%%
fig = plt.figure(figsize=(4, 2), dpi=300)
plt.suptitle('CEBRA-behavior trained with target label',
             fontsize=5)
ax = plt.subplot(121, projection = '3d')
ax.set_title('All trials embedding', fontsize=5, y=-0.1)
x = ax.scatter(cebra_target[:, 0],
               cebra_target[:, 1],
               cebra_target[:, 2],
               c=monkey_target.discrete_index,
               cmap=plt.cm.hsv,
               s=0.01)
ax.axis('off')

ax = plt.subplot(122,projection = '3d')
ax.set_title('direction-averaged embedding', fontsize=5, y=-0.1)
for i in range(8):
    direction_trial = (monkey_target.discrete_index == i)
    trial_avg = cebra_target[direction_trial, :].reshape(-1, 600, 3).mean(axis=0)
    trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    ax.scatter(trial_avg_normed[:, 0],
               trial_avg_normed[:, 1],
               trial_avg_normed[:, 2],
               color=plt.cm.hsv(1 / 8 * i),
               s=0.01)
ax.axis('off')
plt.show()
#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建PCA对象,指定降维后的维度为3
pca = PCA(n_components=3)

# 对矩阵进行PCA降维
matrix_100_3 = pca.fit_transform(neu_data)

# 创建一个新的图形和3D坐标轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plt.scatter
# 绘制散点图
ax.scatter(matrix_100_3[:, 0], matrix_100_3[:, 1], matrix_100_3[:, 2], c=be_phi, marker='o',s = 0.1)

# 设置坐标轴标签
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# 显示图形
plt.show()
bels from 1 to 10
# Assign labels to each element based on the bin it falls into
label_vector = np.digitize(be_phi, bin_edges)
# Map the labels to the desired range (1 to 10)
label_vector2 = labels[label_vector import numpy as np
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

        be_speed_list_young_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
        be_speed_list_young_nov.append(be_speed_sum_nov[int(i / 2), 0])
        be_speed_list_young_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

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

be_speed_list_old_fam1 = []
be_speed_list_old_nov = []
be_speed_list_old_fam1r2 = []

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

    be_speed_list_old_fam1.append(be_speed_sum_fam1[int(i / 2), 0])
    be_speed_list_old_nov.append(be_speed_sum_nov[int(i / 2), 0])
    be_speed_list_old_fam1r2.append(be_speed_sum_fam1r2[int(i / 2), 0])

    gene_list_old.append(mat_trigger[i, 1])
del be_data, be_phi_sum_fam1, be_phi_sum_nov, be_phi_sum_fam1r2
#%%
with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_all_Smask.pkl', 'rb') as file:
    mask = pickle.load(file)

env = 'nov'
#env = 'fam1'
#env = 'fam1r2'
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
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)
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
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)
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
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
        time_list = pickle.load(file)

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


def sparse(A):
    N = A.shape[0]
    # print(max(A[:,4]))
    # A=np.where(A > 0.09, 1, 0)
    k = int(N / 2)

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
            # random_index = np.random.randint(0, Q.shape[1]-1)-1  # 随机选择一个列索引
            Q[i, i - 1] = 0.001
            Q[i - 1, i] = 0.001
    return Q


import plotly.graph_objects as go  # pragma: no cover
from plotly.offline import plot


def plot_sankey(
        all_results,
        optimal_scales=True,
        live=False,
        filename='',
        scale_index=None,
):  # pragma: no cover
    """Plot Sankey diagram of communities accros scale (plotly only).
    Args:
        all_results (dict): results from run function
        optimal_scales (bool): use optimal scales or not
        live (bool): if True, interactive figure will appear in browser
        filename (str): filename to save the plot
        scale_index (bool): plot scale of indices
    """
    sources = []
    targets = []
    values = []
    shift = 0

    # if not scale_index:
    all_results["community_id_reduced"] = all_results["community_id"][0::10]

    community_ids = all_results["community_id_reduced"]

    for i in range(len(community_ids) - 1):
        community_source = np.array(community_ids[i])
        community_target = np.array(community_ids[i + 1])
        source_ids = set(community_source)
        target_ids = set(community_target)
        #         print(target_ids)
        for source in source_ids:
            for target in target_ids:
                value = sum(community_target[community_source == source] == target)
                #                 print(community_target[community_source == source] == target)
                if value > 0:
                    values.append(value)
                    sources.append(source + shift)
                    targets.append(target + len(source_ids) + shift)
        shift += len(source_ids)

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 1,
                    "thickness": 1,
                    "line": {"color": "black", "width": 0.0},
                },
                link={"source": sources, "target": targets, "value": values},
            )
        ],
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        paper_bgcolor='white',  # 设置图表的背景颜色为白色
        plot_bgcolor='white')

    Scale_enumerate = [r'$S_{%s}$' % (len(community_ids) - i) for i in range(len(community_ids))]
    for x_coordinate, column_name in enumerate(Scale_enumerate):
        fig.add_annotation(
            x=x_coordinate,  # Plotly recognizes 0-5 to be the x range.

            y=1.075,  # y value above 1 means above all nodes
            xref="x",
            yref="paper",
            text=column_name,  # Text
            showarrow=False,
            font=dict(
                family="Tahoma",
                size=16,
                color="black"
            ),
            align="left",
        )
    fig.write_image(filename)
#%%
cluster_number = []
Color_Code2 = ['white', 'darkblue', 'red', "orange", "pink", "olive", "cyan", "yellow", "green", "brown", "gray",
               "tomato", 'limegreen']

list_color = []
for color in mcolors.CSS4_COLORS:
    list_color.append(color)
random.shuffle(list_color)
Color_Code1 = [*Color_Code2, *list_color]
Color_Code = [*Color_Code1, *list_color]
z = 5
v = 0
Time_list_matrix = []
for index in range(9,10):  # len(dy_list)
    plt.close()
    if gene_list_young[index] == 119:
        Spike_train = neuron_spike[index][:,
                      int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)]
        mask_index = mask[index]
        Time_train = neuron_spike[index]
        tuning_curve = All_tuning_curve[index]
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_markov.pdf')
            plt.close()

        elif env == "nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_' + str(z) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_markov.pdf')
            plt.close()
        elif env == "fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_' + str(z) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.close()

        if env == 'fam1' or env == 'fam1r2':
            closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            Community = all_results['community_id'][closest_number]

        elif env == "nov":
            selected_partitions = all_results['selected_partitions']
            Community = all_results['community_id'][199]
            closest_number = 133
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]

        non_zero_indices_per_row = []
        for row in Spike_train:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))

        neuralData = []
        ColorCode = []
        C_tuning_curve = []
        it = 0

        tuning_curve_df = pd.DataFrame(tuning_curve)
        labels_df = pd.DataFrame({'community': Community})
        time_list_df = pd.DataFrame(Time_train)

        # 按社区标签对神经元进行排序
        sorted_indices = labels_df.sort_values(by='community').index
        sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]

        sorted_time_list_df = time_list_df.iloc[sorted_indices]
        # 按community分组
        grouped1 = sorted_tuning_curve_df.groupby(labels_df['community'])

        community_max_indices = {}
        for community, group in grouped1:
            mean_tuning_curve = group.mean(axis=0)
            max_index = np.argmax(mean_tuning_curve)
            community_max_indices[community] = max_index

        # 根据平均tuning curve最大值的index对community进行排序
        sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

        # 根据排序后的community重新排列神经元
        sorted_tuning_curve_list = []
        time_new_list = []
        for community in sorted_communities:
            sorted_tuning_curve_list.append(tuning_curve_df[labels_df['community'] == community])
            time_new_list.append(time_list_df[labels_df['community'] == community])
            sorted_tuning_curve_list.append(pd.DataFrame(np.full((3, tuning_curve.shape[1]), np.nan)))
        # 将所有community的tuning curve拼接在一起

        C_tuning_curve = pd.concat(sorted_tuning_curve_list, axis=0).values
        Time_list_matrix.append(time_new_list)
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [6, 3]})
        sns.heatmap(C_tuning_curve, cmap='viridis', vmax=1, cbar=False, ax=axes[0])
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)
        if env == 'fam1':
            angle_fam1 = be_phi_list_young_fam1[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_tuning.pdf')
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(index) + '_young_fam1_tuning.png')
            plt.close()
        elif env == "nov":
            angle_nov = be_phi_list_young_nov[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_nov, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_tuning.pdf')
            plt.savefig(
                '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_tuning.png')
            plt.close()
        elif env == "fam1r2":
            angle_fam1r2 = be_phi_list_young_fam1r2[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1r2, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels, rotation=0)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(index) + '_young_fam1r2_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/' + str(index) + '_young_fam1r2_tuning.png')
            plt.close()

        Nodes = range(len(non_zero_indices_per_row))
        for m, j in zip(Nodes, Nodes):
            for i in range(max(Community) + 1):
                if Community[j] == i:
                    non_zero_indices_per_row[int(m)].sort()
                    neuralData.append(non_zero_indices_per_row[int(m)])
                    ColorCode.append(Color_Code[i + 1])
                    # A_ordered_row[it,:]=Q[j,:]
                    it += 1
        # it=0
        # for i in range(max(Community)+1):
        #     for m,j in zip(Nodes,range(N)):
        #         if Community[j]==i:
        #             A_ordered[:,it]=A_ordered_row[:,j]
        #             it+=1
        fig, axes = plt.subplots(2, 1, figsize=(27, 9), gridspec_kw={'height_ratios': [6, 3]})
        axes[0].eventplot(neuralData, color=ColorCode)
        axes[0].invert_yaxis()

        axes[0].set_title("Reordered neural spike trains " + f'N-{max(Community) + 1}',fontsize=20)
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)


        # max_in_sublists = [max(sublist) for sublist in neuralData if sublist]
        #
        # # 找到最大值中的最大值
        # overall_max = max(max_in_sublists)
        # nearest_5000_multiple = round(overall_max / 10000) * 10000
        # axes[0].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
        # plt.show()

        if env == 'fam1':
            axes[1].plot(be_phi_list_young_fam1[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_list.png')
            plt.close()
        elif env == "nov":
            axes[1].plot(be_phi_list_young_nov[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_list.png')
            plt.close()
        elif env == "fam1r2":
            axes[1].plot(be_phi_list_young_fam1r2[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/neuron_list_' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/neuron_list_' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        from matplotlib.colors import BoundaryNorm

        community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

        # 更新labels_df中的community标签
        labels_df['community'] = labels_df['community'].map(community_mapping)
        Sort_community = labels_df['community'].values
        pic = np.zeros((512, 512))
        for i in range(mask_index.shape[2]):
            index_row = np.where(mask_index[:, :, i] == True)[0]
            index_con = np.where(mask_index[:, :, i] == True)[1]
            for j in range(len(index_row)):
                pic[index_row[j], index_con[j]] = Sort_community[i]
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        import seaborn as sns

        #cmap = plt.get_cmap('tab10')

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
        bounds = np.linspace(-0.5, 11.5, 13)
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(pic, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Scale= {closest_number*1.5/200-0.5:.2f}, N='+str(max(Community) + 1),fontsize=20)
        plt.tight_layout()
        # plt.title('Community beasd on real data')
        # plt.show()
        if env == 'fam1':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()

        elif env == "nov":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_nov_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()
        elif env == "fam1r2":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.png')
            plt.close()

        if env == 'fam1':
            name = '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig/' + str(
                index) + '_young_fam1_sankey.png'
        elif env == "nov":
            name = '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig/' + str(index) + '_young_nov_snakey.png'
        elif env == "fam1r2":
            name = '/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig/snakey' + str(
                index) + '_young_fam1r2.png'

        plot_sankey(
            all_results,
            optimal_scales=True,
            live=False,
            filename=name,
            scale_index=None,
        )

        z = z + 1
    else:
        Spike_train = neuron_spike[index][:,
                      int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)]
        mask_index = mask[index]
        tuning_curve = All_tuning_curve[index]
        type = 'wild type'
        if env == 'fam1':
            with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_Signal_Markov_AD' + str(v) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov.png')
            plt.close()

        elif env == "nov":
            with open('/Users/sonmjack/Downloads/age2 result_nov/nov_Signal_Markov_AD' + str(v) + '.pkl', 'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov.png')
            plt.close()
        elif env == "fam1r2":
            with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_Signal_Markov_AD' + str(v) + '.pkl',
                      'rb') as file:
                all_results = pickle.load(file)
            _ = pgs.plot_scan(all_results)
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        if env == 'fam1' or env == 'fam1r2':
            closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            Community = all_results['community_id'][closest_number]

        elif env == "nov":
            selected_partitions = all_results['selected_partitions']
            Community = all_results['community_id'][199]
            closest_number = 133
            # if selected_partitions == []:
            #     Community = all_results['community_id'][199]
            # else:
            #     closest_number = min(all_results['selected_partitions'], key=lambda x: abs(x - 133))
            #     Community = all_results['community_id'][closest_number]

        non_zero_indices_per_row = []
        for row in Spike_train:
            # 找到每行中不为0的元素的列索引
            non_zero_indices = np.where(row != 0)[0]
            # 添加到列表中
            non_zero_indices_per_row.append(list(non_zero_indices))

        neuralData = []
        ColorCode = []
        C_tuning_curve = []
        it = 0

        tuning_curve_df = pd.DataFrame(tuning_curve)
        labels_df = pd.DataFrame({'community': Community})
        time_list_df = pd.DataFrame(Time_train)

        # 按社区标签对神经元进行排序
        sorted_indices = labels_df.sort_values(by='community').index
        sorted_tuning_curve_df = tuning_curve_df.iloc[sorted_indices]

        sorted_time_list_df = time_list_df.iloc[sorted_indices]
        # 按community分组
        grouped1 = sorted_tuning_curve_df.groupby(labels_df['community'])

        community_max_indices = {}
        for community, group in grouped1:
            mean_tuning_curve = group.mean(axis=0)
            max_index = np.argmax(mean_tuning_curve)
            community_max_indices[community] = max_index

        # 根据平均tuning curve最大值的index对community进行排序
        sorted_communities = sorted(community_max_indices.keys(), key=lambda x: community_max_indices[x])

        # 根据排序后的community重新排列神经元
        sorted_tuning_curve_list = []
        time_new_list = []
        for community in sorted_communities:
            sorted_tuning_curve_list.append(tuning_curve_df[labels_df['community'] == community])
            time_new_list.append(time_list_df[labels_df['community'] == community])
            sorted_tuning_curve_list.append(pd.DataFrame(np.full((3, tuning_curve.shape[1]), np.nan)))
        # 将所有community的tuning curve拼接在一起

        C_tuning_curve = pd.concat(sorted_tuning_curve_list, axis=0).values

        fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [6, 3]})
        sns.heatmap(C_tuning_curve, cmap='viridis', vmax=1, cbar=False, ax=axes[0])
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)
        if env == 'fam1':
            angle_fam1 = be_phi_list_young_fam1[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_tuning.png')
            plt.close()
        elif env == "nov":
            angle_nov = be_phi_list_young_nov[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_nov, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_tuning.png')
            plt.close()
        elif env == "fam1r2":
            angle_fam1r2 = be_phi_list_young_fam1r2[index].reshape(-1)
            bins = np.linspace(0, 360, num=101)
            even_labels = [str(i) for i in range(0, 100, 2)]
            angle_categories = pd.cut(angle_fam1r2, bins, right=False, labels=[str(i) for i in range(100)])
            angle_distribution = pd.value_counts(angle_categories, normalize=True).sort_index()

            df = pd.DataFrame({'Angle Range': angle_distribution.index, 'Percentage': angle_distribution.values})
            observed_frequencies = pd.value_counts(angle_categories, sort=False).sort_index()
            sns.barplot(x='Angle Range', y='Percentage', data=df, ax=axes[1])
            x_labels = [label if int(label) % 2 == 0 else '' for label in df['Angle Range']]
            # axes[1].set_xticklabels(x_labels, rotation=90)
            axes[1].set_xticklabels(x_labels, rotation=0)
            #axes[1].set_title('Distribution of Angles of '+str(index),fontsize=20)
            axes[1].set_ylabel('Percentage',fontsize=20)
            axes[1].set_xlabel('Position (cm)',fontsize=20)
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_tuning.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1r2_tuning.png')
            plt.close()

        Nodes = range(len(non_zero_indices_per_row))
        for m, j in zip(Nodes, Nodes):
            for i in range(max(Community) + 1):
                if Community[j] == i:
                    non_zero_indices_per_row[int(m)].sort()
                    neuralData.append(non_zero_indices_per_row[int(m)])
                    ColorCode.append(Color_Code[i + 1])
                    # A_ordered_row[it,:]=Q[j,:]
                    it += 1
        # it=0
        # for i in range(max(Community)+1):
        #     for m,j in zip(Nodes,range(N)):
        #         if Community[j]==i:
        #             A_ordered[:,it]=A_ordered_row[:,j]
        #             it+=1
        fig, axes = plt.subplots(2, 1, figsize=(27, 9), gridspec_kw={'height_ratios': [6, 3]})
        axes[0].eventplot(neuralData, color=ColorCode)
        axes[0].invert_yaxis()

        axes[0].set_title("Reordered neural spike trains " + f'N-{max(Community) + 1}',fontsize=20)
        axes[0].set_ylabel('Rearranged Neuron ID', fontsize=20)

        # max_in_sublists = [max(sublist) for sublist in neuralData if sublist]
        #
        # # 找到最大值中的最大值
        # overall_max = max(max_in_sublists)
        # nearest_5000_multiple = round(overall_max / 10000) * 10000
        # axes[0].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
        # plt.show()

        if env == 'fam1':
            axes[1].plot(be_phi_list_young_fam1[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_list.png')
            plt.close()
        elif env == "nov":
            axes[1].plot(be_phi_list_young_nov[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_list.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_list.png')
            plt.close()
        elif env == "fam1r2":
            axes[1].plot(be_phi_list_young_fam1r2[index][
                         int(neuron_spike[index].shape[1] / 3):int(neuron_spike[index].shape[1] * 2 / 3)])
            # axes[1].set_xticks(np.arange(0,nearest_5000_multiple, 10000), [str(int(i*10000/30)) for i in range(0, round(overall_max / 10000))],fontsize=13)
            axes[1].set_xlabel('Time', fontsize=20)
            axes[1].set_ylabel('Position°', fontsize=20)
            axes[1].invert_yaxis()
            plt.tight_layout()
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/neuron_list_' + str(
                index) + '_young_fam1r2_markov.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/neuron_list_' + str(
                index) + '_young_fam1r2_markov.png')
            plt.close()

        from matplotlib.colors import BoundaryNorm

        community_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_communities, start=1)}

        # 更新labels_df中的community标签
        labels_df['community'] = labels_df['community'].map(community_mapping)
        Sort_community = labels_df['community'].values

        pic = np.zeros((512, 512))
        for i in range(mask_index.shape[2]):
            index_row = np.where(mask_index[:, :, i] == True)[0]
            index_con = np.where(mask_index[:, :, i] == True)[1]
            for j in range(len(index_row)):
                pic[index_row[j], index_con[j]] = Sort_community[i]
        # cmap = sns.diverging_palette(220, 20, as_cmap=True)
        import seaborn as sns

        #cmap = plt.get_cmap('tab10')

        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("custom_cmap", Color_Code2, N=len(Color_Code2))
        bounds = np.linspace(-0.5, 11.5, 13)
        norm = BoundaryNorm(bounds, cmap.N)
        sns.heatmap(pic, cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Scale= {closest_number*1.5/200-0.5:.2f}, N='+str(max(Community) + 1),fontsize=20)
        plt.tight_layout()
        plt.title('Community beasd on real data')
        # plt.show()
        if env == 'fam1':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_markov_neuron_mask.png')
            plt.close()

        elif env == "nov":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_mask.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_markov_neuron_mask.png')
            plt.close()
        elif env == "fam1r2":
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.pdf')
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/mask_' + str(
                index) + '_young_fam1r2_markov_neuron.png')
            plt.close()

        if env == 'fam1':
            name = '/Users/sonmjack/Downloads/age2 result_fam1/Signal_Markov_fig_AD/' + str(
                index) + '_young_fam1_sankey.png'
        elif env == "nov":
            name = '/Users/sonmjack/Downloads/age2 result_nov/Signal_Markov_fig_AD/' + str(
                index) + '_young_nov_snakey.png'
        elif env == "fam1r2":
            name = '/Users/sonmjack/Downloads/age2 result_fam1r2/Signal_Markov_fig_AD/snakey' + str(
                index) + '_young_fam1r2.png'

        plot_sankey(
            all_results,
            optimal_scales=True,
            live=False,
            filename=name,
            scale_index=None,
        )

        v = v + 1

#%%
from cebra import CEBRA
#%%
max_iterations = 5000#default is 5000.
output_dimension = 32 #here, we set as a variable for hypothesis testing below.
neu_data = np.array(Time_list_matrix[0][0]).T+0.01
#neu_data = neu_data[0:22200,:]

be_phi = be_phi_list_young_nov[9]#[0:22200,:]
bin_edges = np.linspace(0, 360, 11)  # 11 edges for 10 bins
# Create labels for each bin
labels = np.arange(1, 11)  # Labels from 1 to 10
# Assign labels to each element based on the bin it falls into
label_vector = np.digitize(be_phi, bin_edges)
# Map the labels to the desired range (1 to 10)
label_vector2 = labels[label_vector - 1]
be_2D = np.column_stack([be_x_list_young_nov[9], be_y_list_young_nov[9]])#[0:22200,:]

# plt.figure(figsize=(20, 4))
# #plt.plot(neuron_spike[9][1,:]+0.1)
# plt.plot(Time_train[1,:])
# plt.show()


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
                        time_offsets=10)
                        #hybrid = True)# hybrid = True

#cebra_hybrid_model.fit(neu_data, be_2D)
cebra_hybrid_model.fit(neu_data, label_vector2)
cebra_hybrid = cebra_hybrid_model.transform(neu_data)

#%%

def plot_hippocampus(ax, embedding, be_phi, elev, azim,roll,gray = False, idx_order = (0,1,2),xlabel='X', ylabel='Y', zlabel='Z'):
    p = ax.scatter(embedding [:, 0], embedding [:, 1], embedding [:, 2],s=5,c=be_phi,cmap ='viridis')
    #ax.plot(embedding[:, 0], embedding[:, 1], embedding [:, 2],c='gray', linewidth=0.5)

    ax.grid(False)
    ax.set_box_aspect([np.ptp(i) for i in embedding.T])
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.view_init(elev=elev, azim=azim,roll=roll)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return p,ax

#matplotlib notebook
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,20))

# ax1 = plt.subplot(141, projection='3d')
# ax2 = plt.subplot(142, projection='3d')
# ax3 = plt.subplot(143, projection='3d')
ax4 = plt.subplot(111, projection='3d')

# ax1=plot_hippocampus(ax1, cebra_posdir3, hippocampus_pos.continuous_index)
# ax2=plot_hippocampus(ax2, cebra_posdir_shuffled3, hippocampus_pos.continuous_index)
# ax3=plot_hippocampus(ax3, cebra_time3, hippocampus_pos.continuous_index)
p, ax4= plot_hippocampus(ax4, cebra_hybrid, be_phi, elev=0, azim=45, roll=0)
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_phi)
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D[:,0])
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D[:,1])
# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax4.set_title('CEBRA-Hybrid')
fig.colorbar(p)
plt.show()

#%%
import cebra.datasets
#%%
hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
Te_b = hippocampus_pos.continuous_index.numpy()
def split_data(data, test_ratio):

    split_idx = int(len(data)* (1-test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]
    label_train = data.continuous_index[:split_idx]
    label_test = data.continuous_index[split_idx:]

    return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()

neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2)

#%%
monkey_pos = cebra.datasets.init('area2-bump-pos-active')
Te_M = monkey_pos.continuous_index.numpy()
#%%
monkey_target = cebra.datasets.init('area2-bump-target-active')
Te_D= monkey_target.discrete_index.numpy()
TE_neural = monkey_target.neural.numpy()

#%%
plt.figure(figsize=(20, 4))
plt.plot(TE_neural[:,1])
plt.show()

#%%
direction_trial = (monkey_target.discrete_index == 1).numpy()
cebra_target_model = CEBRA(model_architecture='offset10-model',
                           batch_size=512,
                           learning_rate=0.0001,
                           temperature=1,
                           output_dimension=3,
                           max_iterations=max_iterations,
                           distance='cosine',
                           conditional='time_delta',
                           device='cuda_if_available',
                           verbose=True,
                           time_offsets=10)
cebra_target_model.fit(monkey_target.neural,
                       monkey_target.discrete_index.numpy())
cebra_target = cebra_target_model.transform(monkey_target.neural)
#%%
fig = plt.figure(figsize=(4, 2), dpi=300)
plt.suptitle('CEBRA-behavior trained with target label',
             fontsize=5)
ax = plt.subplot(121, projection = '3d')
ax.set_title('All trials embedding', fontsize=5, y=-0.1)
x = ax.scatter(cebra_target[:, 0],
               cebra_target[:, 1],
               cebra_target[:, 2],
               c=monkey_target.discrete_index,
               cmap=plt.cm.hsv,
               s=0.01)
ax.axis('off')

ax = plt.subplot(122,projection = '3d')
ax.set_title('direction-averaged embedding', fontsize=5, y=-0.1)
for i in range(8):
    direction_trial = (monkey_target.discrete_index == i)
    trial_avg = cebra_target[direction_trial, :].reshape(-1, 600, 3).mean(axis=0)
    trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    ax.scatter(trial_avg_normed[:, 0],
               trial_avg_normed[:, 1],
               trial_avg_normed[:, 2],
               color=plt.cm.hsv(1 / 8 * i),
               s=0.01)
ax.axis('off')
plt.show()
#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建PCA对象,指定降维后的维度为3
pca = PCA(n_components=3)

# 对矩阵进行PCA降维
matrix_100_3 = pca.fit_transform(neu_data)

# 创建一个新的图形和3D坐标轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plt.scatter
# 绘制散点图
ax.scatter(matrix_100_3[:, 0], matrix_100_3[:, 1], matrix_100_3[:, 2], c=be_phi, marker='o',s = 0.1)

# 设置坐标轴标签
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# 显示图形
plt.show()
- 1]
#%%
be_2D = np.column_stack([be_x_list_young_nov[9], be_y_list_young_nov[9]])#[0:22200,:]

# plt.figure(figsize=(20, 4))
# #plt.plot(neuron_spike[9][1,:]+0.1)
# plt.plot(Time_train[1,:])
# plt.show()


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
                        time_offsets=10)
                        #hybrid = True)# hybrid = True

#cebra_hybrid_model.fit(neu_data, be_2D)
cebra_hybrid_model.fit(neu_data, label_vector2)
cebra_hybrid = cebra_hybrid_model.transform(neu_data)

#%%

def plot_hippocampus(ax, embedding, be_phi, elev, azim,roll,gray = False, idx_order = (0,1,2),xlabel='X', ylabel='Y', zlabel='Z'):
    p = ax.scatter(embedding [:, 0], embedding [:, 1], embedding [:, 2],s=5,c=be_phi,cmap ='viridis')
    #ax.plot(embedding[:, 0], embedding[:, 1], embedding [:, 2],c='gray', linewidth=0.5)

    ax.grid(False)
    ax.set_box_aspect([np.ptp(i) for i in embedding.T])
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.view_init(elev=elev, azim=azim,roll=roll)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return p,ax

#matplotlib notebook
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,20))

# ax1 = plt.subplot(141, projection='3d')
# ax2 = plt.subplot(142, projection='3d')
# ax3 = plt.subplot(143, projection='3d')
ax4 = plt.subplot(111, projection='3d')

# ax1=plot_hippocampus(ax1, cebra_posdir3, hippocampus_pos.continuous_index)
# ax2=plot_hippocampus(ax2, cebra_posdir_shuffled3, hippocampus_pos.continuous_index)
# ax3=plot_hippocampus(ax3, cebra_time3, hippocampus_pos.continuous_index)
p, ax4= plot_hippocampus(ax4, cebra_hybrid, be_phi, elev=0, azim=45, roll=0)
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_phi)
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D[:,0])
#p,ax4=plot_hippocampus(ax4, cebra_hybrid, be_2D[:,1])
# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax4.set_title('CEBRA-Hybrid')
fig.colorbar(p)
plt.show()

#%%
import cebra.datasets
hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
Te_b = hippocampus_pos.continuous_index.numpy()
def split_data(data, test_ratio):

    split_idx = int(len(data)* (1-test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]
    label_train = data.continuous_index[:split_idx]
    label_test = data.continuous_index[split_idx:]

    return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()

neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2)

#%%
monkey_pos = cebra.datasets.init('area2-bump-pos-active')
Te_M = monkey_pos.continuous_index.numpy()
#%%
monkey_target = cebra.datasets.init('area2-bump-target-active')
Te_D= monkey_target.discrete_index.numpy()
TE_neural = monkey_target.neural.numpy()

#%%
plt.figure(figsize=(20, 4))
plt.plot(TE_neural[:,1])
plt.show()

#%%
direction_trial = (monkey_target.discrete_index == 1).numpy()
cebra_target_model = CEBRA(model_architecture='offset10-model',
                           batch_size=512,
                           learning_rate=0.0001,
                           temperature=1,
                           output_dimension=3,
                           max_iterations=max_iterations,
                           distance='cosine',
                           conditional='time_delta',
                           device='cuda_if_available',
                           verbose=True,
                           time_offsets=10)
cebra_target_model.fit(monkey_target.neural,
                       monkey_target.discrete_index.numpy())
cebra_target = cebra_target_model.transform(monkey_target.neural)
#%%
fig = plt.figure(figsize=(4, 2), dpi=300)
plt.suptitle('CEBRA-behavior trained with target label',
             fontsize=5)
ax = plt.subplot(121, projection = '3d')
ax.set_title('All trials embedding', fontsize=5, y=-0.1)
x = ax.scatter(cebra_target[:, 0],
               cebra_target[:, 1],
               cebra_target[:, 2],
               c=monkey_target.discrete_index,
               cmap=plt.cm.hsv,
               s=0.01)
ax.axis('off')

ax = plt.subplot(122,projection = '3d')
ax.set_title('direction-averaged embedding', fontsize=5, y=-0.1)
for i in range(8):
    direction_trial = (monkey_target.discrete_index == i)
    trial_avg = cebra_target[direction_trial, :].reshape(-1, 600, 3).mean(axis=0)
    trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    ax.scatter(trial_avg_normed[:, 0],
               trial_avg_normed[:, 1],
               trial_avg_normed[:, 2],
               color=plt.cm.hsv(1 / 8 * i),
               s=0.01)
ax.axis('off')
plt.show()
#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建PCA对象,指定降维后的维度为3
pca = PCA(n_components=3)

# 对矩阵进行PCA降维
matrix_100_3 = pca.fit_transform(neu_data)

# 创建一个新的图形和3D坐标轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plt.scatter
# 绘制散点图
ax.scatter(matrix_100_3[:, 0], matrix_100_3[:, 1], matrix_100_3[:, 2], c=be_phi, marker='o',s = 0.1)

# 设置坐标轴标签
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# 显示图形
plt.show()
