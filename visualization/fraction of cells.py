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
#%%
be_data = scipy.io.loadmat('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_trackdata.mat')
mat_trigger = np.load('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/shengyuan_trigger_fam1.npy')
import h5py

type_array = h5py.File('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/data_fam1novfam1_timeseries.mat')

gene = type_array['genotype'][:, :].T
# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum = be_data['fam1_phi']
be_x = be_data['fam1_x']
be_y = be_data['fam1_y']
be_time = be_data['fam1_time']
be_speed = be_data['fam1_speed']
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
#%%
Type = 'Young'
if Type == 'Young':
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve2 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve3 = pickle.load(file)
elif Type == 'Old':
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve2 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve3 = pickle.load(file)
average_tuning_curve = []

def average_tune(All_tuning_curve,type):
    average_tuning_curve = []
    for index in range(len(All_tuning_curve)):
        if type == 'wild type':
            if gene_list[index] == 119:
                print('Finished ' + f'{index}')
                tuning_curve = np.mean(All_tuning_curve[index],axis=0)
                average_tuning_curve.append(tuning_curve)
        elif type == 'AD':
            if gene_list[index] == 116:
                print('Finished ' + f'{index}')
                tuning_curve = np.mean(All_tuning_curve[index],axis=0)
                average_tuning_curve.append(tuning_curve)
    return average_tuning_curve
def prepare_data(data, group_name):
    dimensions = np.tile(np.arange(0, 100), len(data))
    df = pd.DataFrame({'Location': dimensions, 'Value': data.reshape(-1,1).flatten()})
    df['Group'] = group_name
    return df
# average_tuning_curve1 = average_tune(All_tuning_curve1,'wild type')
# average_tuning_curve2 = average_tune(All_tuning_curve2,'wild type')
# average_tuning_curve3 = average_tune(All_tuning_curve3,'wild type')

# average_tuning_curve1 = average_tune(All_tuning_curve1,'AD')
# average_tuning_curve2 = average_tune(All_tuning_curve2,'AD')
# average_tuning_curve3 = average_tune(All_tuning_curve3,'AD')

# df1 = prepare_data(np.array(average_tuning_curve1), 'fam1')
# df2 = prepare_data(np.array(average_tuning_curve2), 'Nov')
# df3 = prepare_data(np.array(average_tuning_curve3), 'famr2')

# df = pd.concat([df1, df2, df3])
# plt.close()
# # 绘制曲线图
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df, x='Location',
#              y='Value', hue='Group', style='Group', markers=True, dashes=False)
# plt.title('Global tuning curves for three groups No.'+f'{index}')
# plt.xlabel('Location (cm)')
# plt.ylabel('Fraction of cells')
# plt.legend(title='Group')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.show()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
it = 0
for index in range(len(All_tuning_curve1)):
    if gene_list[index] == 119:
        type = 'wild type'
        print('Finished ' + f'{index}')
        df1 = prepare_data(All_tuning_curve1[index], 'fam1')
        df2 = prepare_data(All_tuning_curve2[index], 'Nov')
        df3 = prepare_data(All_tuning_curve3[index], 'famr2')

        df = pd.concat([df1, df2, df3])
        # 绘制曲线图
        sns.lineplot(data=df, x='Location',
                     y='Value', hue='Group', style='Group', markers=True, dashes=False,ax=axes[it])

        axes[it].set_xlabel('Location (cm)', fontsize=13)
        axes[it].set_ylabel('Fraction of cells', fontsize=13)
        axes[it].set_title('Global tuning curves for three groups No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        df = pd.concat([df1, df2, df3])
        # 绘制曲线图
        sns.lineplot(data=df, x='Location',
                     y='Value', hue='Group', style='Group', markers=True, dashes=False,ax=axes[it])

        axes[it].set_xlabel('Location (cm)', fontsize=13)
        axes[it].set_ylabel('Fraction of cells', fontsize=13)
        axes[it].set_title('Global tuning curves for three groups No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        axes[it].legend(title='Group')
        axes[it].legend(title='Group')
        it += 1
    else:
        type = 'AD'
        pass
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT Global tuning curve for 3'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT Global tuning curve for 3'+'.jpg')
    plt.close()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(12, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
it = 0
for index in range(len(All_tuning_curve1)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        df1 = prepare_data(All_tuning_curve1[index], 'fam1')
        df2 = prepare_data(All_tuning_curve2[index], 'Nov')
        df3 = prepare_data(All_tuning_curve3[index], 'famr2')

        df = pd.concat([df1, df2, df3])
        # 绘制曲线图
        sns.lineplot(data=df, x='Location',
                     y='Value', hue='Group', style='Group', markers=True, dashes=False,ax=axes[it])

        axes[it].set_xlabel('Location (cm)', fontsize=13)
        axes[it].set_ylabel('Fraction of cells', fontsize=13)
        axes[it].set_title('Global tuning curves for three groups No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        axes[it].legend(title='Group')
        it += 1
    else:
        type = 'AD'
        pass
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'AD Global tuning curve for 3'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD Global tuning curve for 3'+'.jpg')
    plt.close()

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
def normal(A):
    min_val = np.min(A)
    max_val = np.max(A)
    A = (A - min_val) / (max_val - min_val)

    return A
def sparse (A):
    N = A.shape[0]
    np.fill_diagonal(A, 0)
    #A = normal(A)
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
            random_index = np.random.randint(0, Q.shape[1])  # 随机选择一个列索引
            Q[i, random_index] = 0.001

    return Q

def asymmetry(Q):
    N = np.size(Q,0)
    Detail_Balance_Q = []
    Compone_strength_Q = []


    for i in range(N):
        for j in range(N):
            if (Q[i, j] > 0):
                Compone_strength_Q.append(Q[i, j])
            if (i > j and (Q[i, j] + Q[j, i]) > 0):
                #             print(i,j,Be[i,j])
                Detail_Balance_Q.append(abs(Q[i, j] - Q[j, i]) / (Q[i, j] + Q[j, i]))
    return Detail_Balance_Q
def prepare_data_asy(data,group_name):
    df = pd.DataFrame({'Value': data.reshape(-1,1).flatten()})
    df['Group'] = group_name
    return df
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(12, 36))
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_all_EPSP_young.pkl', 'rb') as file:
        All_dy_list1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_all_EPSP_young.pkl', 'rb') as file:
        All_dy_list2 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_all_EPSP_young.pkl', 'rb') as file:
        All_dy_list3 = pickle.load(file)
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_all_EPSP.pkl', 'rb') as file:
        All_dy_list1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_all_EPSP.pkl', 'rb') as file:
        All_dy_list2 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_all_EPSP.pkl', 'rb') as file:
        All_dy_list3 = pickle.load(file)

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        A1 = All_dy_list1[index]
        Q1 = sparse(A1)
        Detail_Balance_Q1 = np.array(asymmetry(Q1))

        A2 = All_dy_list2[index]
        Q2 = sparse(A2)
        Detail_Balance_Q2 = np.array(asymmetry(Q2))

        A3 = All_dy_list3[index]
        Q3 = sparse(A3)
        Detail_Balance_Q3 = np.array(asymmetry(Q3))

        df1 = prepare_data_asy(Detail_Balance_Q1, 'fam1')
        df2 = prepare_data_asy(Detail_Balance_Q2, 'Nov')
        df3 = prepare_data_asy(Detail_Balance_Q3, 'famr2')
        df = pd.concat([df1, df2, df3])
        # 绘制曲线图

        sns.histplot(data=df,x='Value', hue='Group',kde=True, multiple="stack",alpha=0.5, bins=bins, ax=axes[it])
        # sns.histplot(Detail_Balance_Q1, color='skyblue', kde=True, label='fam1', alpha=0.5, bins=bins,ax=axes[it])
        # sns.histplot(Detail_Balance_Q2, color='red', kde=True, label='Nov', alpha=0.5, bins=bins, ax=axes[it])
        # sns.histplot(Detail_Balance_Q3, color='green', kde=True, label='famr2', alpha=0.5, bins=bins, ax=axes[it])

        axes[it].set_xlabel('Degree of asymmetry', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        it += 1
    else:
        type = 'AD'
        pass

if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'AD asymmetry for 3'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD asymmetry for 3'+'.jpg')
    plt.close()


#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(12, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))

it = 0
bins = np.linspace(0, 1, num=50)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 116:
        A3 = All_dy_list3[index]
        Q3 = sparse(A3)
        Q= np.copy(Q3)
        #Q[np.where(Q >= 1)] = 1
        non_zero_indices = Q != 0
        Q = Q[non_zero_indices]
        df1 = prepare_data_asy(Q, 'fam1r2')
        axes[it].set_xlabel('Weight of EPSP', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
        it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'AD EPSP fam1r2'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD EPSP fam1r2'+'.jpg')
    plt.close()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(12, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))

it = 0
bins = np.linspace(0, 1, num=50)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 116:
        A1 = All_dy_list1[index]
        Q1 = sparse(A1)
        Q = np.copy(Q1)
        #Q[np.where(Q >= 1)] = 1
        non_zero_indices = Q != 0
        Q = Q[non_zero_indices]
        df1 = prepare_data_asy(Q, 'fam1')
        axes[it].set_xlabel('Weight of EPSP', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
        it += 1

if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'AD EPSP fam1'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD EPSP fam1'+'.jpg')
    plt.close()

#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(6, 1, figsize=(12, 36))
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))

it = 0
bins = np.linspace(0, 1, num=50)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 116:
        A2 = All_dy_list2[index]
        Q2 = sparse(A2)
        Q = np.copy(Q2)
        #Q[np.where(Q >= 1)] = 1
        non_zero_indices = Q != 0
        Q = Q[non_zero_indices]
        df1 = prepare_data_asy(Q, 'fam1')
        axes[it].set_xlabel('Weight of EPSP', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
        it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'AD EPSP nov'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD EPSP nov'+'.jpg')
    plt.close()

#%%

if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        type = 'Wild type'
        print('Finished ' + f'{index}')
        A1 = All_dy_list1[index]
        Q1 = sparse(A1)
        Detail_Balance_Q1 = np.array(asymmetry(Q1))

        A2 = All_dy_list2[index]
        Q2 = sparse(A2)
        Detail_Balance_Q2 = np.array(asymmetry(Q2))

        A3 = All_dy_list3[index]
        Q3 = sparse(A3)
        Detail_Balance_Q3 = np.array(asymmetry(Q3))

        df1 = prepare_data_asy(Detail_Balance_Q1, 'fam1')
        df2 = prepare_data_asy(Detail_Balance_Q2, 'Nov')
        df3 = prepare_data_asy(Detail_Balance_Q3, 'famr2')
        df = pd.concat([df1, df2, df3])
        # 绘制曲线图

        sns.histplot(data=df,x='Value', hue='Group',kde=True, multiple="stack",alpha=0.5, bins=bins, ax=axes[it])
        # sns.histplot(Detail_Balance_Q1, color='skyblue', kde=True, label='fam1', alpha=0.5, bins=bins,ax=axes[it])
        # sns.histplot(Detail_Balance_Q2, color='red', kde=True, label='Nov', alpha=0.5, bins=bins, ax=axes[it])
        # sns.histplot(Detail_Balance_Q3, color='green', kde=True, label='famr2', alpha=0.5, bins=bins, ax=axes[it])

        axes[it].set_xlabel('Degree of asymmetry', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        it += 1

    else:
        type = 'AD'
        pass
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT asymmetry for 3'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT asymmetry for 3'+'.jpg')
    plt.close()

#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        A3 = All_dy_list3[index]
        Q3= sparse(A3)
        Q= np.copy(Q3)
        #Q[np.where(Q >= 1)] = 1
        non_zero_indices = Q != 0
        Q = Q[non_zero_indices]
        df1 = prepare_data_asy(Q, 'fam1r2')
        axes[it].set_xlabel('Weight of EPSP', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
        it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT EPSP fam1r2'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT EPSP fam1r2'+'.jpg')
    plt.close()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        A1 = All_dy_list1[index]
        Q1 = sparse(A1)
        Q= np.copy(Q1)
        #Q[np.where(Q >= 1)] = 1
        non_zero_indices = Q != 0
        Q = Q[non_zero_indices]
        df1 = prepare_data_asy(Q, 'fam1')
        axes[it].set_xlabel('Weight of EPSP', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
        it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT EPSP fam1'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT EPSP fam1'+'.jpg')
    plt.close()
#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        A2 = All_dy_list2[index]
        Q2 = sparse(A2)
        Q= np.copy(Q2)
        #Q[np.where(Q >= 1)] = 1
        non_zero_indices = Q != 0
        Q = Q[non_zero_indices]
        df1 = prepare_data_asy(Q, 'nov')
        axes[it].set_xlabel('Weight of EPSP', fontsize=13)
        axes[it].set_title('No.' + f'{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
        it += 1

if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT EPSP nov'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT EPSP nov'+'.jpg')
    plt.close()


#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        A3 = All_dy_list3[index]
        Q3 = normal(A3)
        eigenvalues = np.linalg.eigvals(Q3)
        # Calculate the modulus (absolute values) of the eigenvalues
        # modulus_eigenvalues = np.abs(eigenvalues)
        # sorted_modulus = np.sort(modulus_eigenvalues)[::-1]
        #
        # sns.lineplot(x=np.arange(len(sorted_modulus)), y=sorted_modulus, marker='o', linestyle='-', color='red',
        #              ax=axes[it])
        sns.scatterplot(x=eigenvalues.real, y=eigenvalues.imag, ax=axes[it])
        axes[it].set_xlabel('Index', fontsize=13)
        axes[it].set_ylabel('Modulus', fontsize=13)

        axes[it].set_title('Sorted Modulus of Eigenvalues' + f'__{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        it += 1

if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1r2'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1r2'+'.jpg')
    plt.close()


#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        A1 = All_dy_list1[index]
        Q1 = normal(A1)
        eigenvalues = np.linalg.eigvals(Q1)
        # Calculate the modulus (absolute values) of the eigenvalues
        # modulus_eigenvalues = np.abs(eigenvalues)
        # sorted_modulus = np.sort(modulus_eigenvalues)[::-1]
        # sns.lineplot(x=np.arange(len(sorted_modulus)), y=sorted_modulus, marker='o', linestyle='-', color='red',
        #              ax=axes[it])
        sns.scatterplot(x=eigenvalues.real, y=eigenvalues.imag, ax=axes[it])
        axes[it].set_xlabel('Index', fontsize=13)
        axes[it].set_ylabel('Modulus', fontsize=13)

        axes[it].set_title('Sorted Modulus of Eigenvalues' + f'__{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1'+'.jpg')
    plt.close()

#%%
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))

elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_dy_list1)):
    if gene_list[index] == 119:
        A2 = All_dy_list2[index]
        Q2 = normal(A2)
        eigenvalues = np.linalg.eigvals(Q2)
        # Calculate the modulus (absolute values) of the eigenvalues
        # modulus_eigenvalues = np.abs(eigenvalues)
        # sorted_modulus = np.sort(modulus_eigenvalues)[::-1]
        # sns.lineplot(x=np.arange(len(sorted_modulus)), y=sorted_modulus, marker='o', linestyle='-', color='red',
        #              ax=axes[it])
        sns.scatterplot(x=eigenvalues.real, y=eigenvalues.imag, ax=axes[it])

        axes[it].set_xlabel('Index', fontsize=13)
        axes[it].set_ylabel('Modulus', fontsize=13)

        axes[it].set_title('Sorted Modulus of Eigenvalues' + f'__{index} {type}')
        axes[it].spines['top'].set_visible(False)
        axes[it].spines['right'].set_visible(False)
        it += 1

if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT egenEPSP_F nov'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT egenEPSP_F nov'+'.jpg')
    plt.close()


#%%
from PIL import Image

# 图像文件路径
if Type == 'Young':
    image_files = ['/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1.jpg',
                   '/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT egenEPSP_F nov.jpg',
                   '/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1r2.jpg']
if Type == 'Old':
    image_files = ['/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1.jpg',
                   '/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT egenEPSP_F nov.jpg',
                   '/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'WT egenEPSP_F fam1r2.jpg']
# 打开图像并放入列表
images = [Image.open(image) for image in image_files]

# 计算新图像的总宽度和最大高度
total_width = sum(image.width for image in images)
max_height = max(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (total_width, max_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (x_offset, 0))
    x_offset += image.width

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/combined_egenF_WT.jpg')
if Type == 'Old':
    new_image.save('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/combined_egenF_WT.jpg')


#%%

Type = 'Old'
if Type == 'Young':
    gene_list = gene_list_young
    fig, axes = plt.subplots(11, 1, figsize=(12, 66))
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1/fam1_signal_corr_WT', 'rb') as file:
        All_sc_fam1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_nov/nov_signal_corr_WT', 'rb') as file:
        All_sc_nov = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/fam1r2_signal_corr_WT', 'rb') as file:
        All_sc_fam1r2 = pickle.load(file)
elif Type == 'Old':
    gene_list = gene_list_old
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1/fam1_signal_corr_AD' ,'rb') as file:
        All_sc_fam1 = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_nov/nov_signal_corr_AD', 'rb') as file:
        All_sc_nov = pickle.load(file)
    with open('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/fam1r2_signal_corr_AD', 'rb') as file:
        All_sc_fam1r2 = pickle.load(file)

it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_sc_fam1)):
   #if gene_list[index] == 116:
    Q= np.copy(All_sc_fam1r2[index])
    #Q= sparse(Q)
    Q[np.where(Q >= 1)] = 1
    # non_zero_indices = Q != 0
    # Q = Q[non_zero_indices]
    df1 = prepare_data_asy(Q, 'fam1r2')
    axes[it].set_xlabel('Weight of signal correlation', fontsize=13)
    axes[it].set_title('No.' + f'{index} {type}')
    axes[it].spines['top'].set_visible(False)
    axes[it].spines['right'].set_visible(False)
    sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
    axes[it].set_xlim(left=0)
    it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT SC fam1r2'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD SC fam1r2'+'.jpg')
    plt.close()



it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_sc_fam1)):
   #if gene_list[index] == 116:
    Q= np.copy(All_sc_fam1[index])
    #Q= sparse(Q)
    Q[np.where(Q >= 1)] = 1
    # non_zero_indices = Q != 0
    # Q = Q[non_zero_indices]
    df1 = prepare_data_asy(Q, 'fam1')
    axes[it].set_xlabel('Weight of signal correlation', fontsize=13)
    axes[it].set_title('No.' + f'{index} {type}')
    axes[it].spines['top'].set_visible(False)
    axes[it].spines['right'].set_visible(False)
    sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
    axes[it].set_xlim(left=0)
    it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT SC fam1'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD SC fam1'+'.jpg')
    plt.close()



it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_sc_fam1)):
   #if gene_list[index] == 116:
    Q= np.copy(All_sc_nov[index])
    #Q= sparse(Q)
    Q[np.where(Q >= 1)] = 1
    # non_zero_indices = Q != 0
    # Q = Q[non_zero_indices]
    df1 = prepare_data_asy(Q, 'nov')
    axes[it].set_xlabel('Weight of signal correlation', fontsize=13)
    axes[it].set_title('No.' + f'{index} {type}')
    axes[it].spines['top'].set_visible(False)
    axes[it].spines['right'].set_visible(False)
    sns.histplot(data=df1, x='Value', kde=True, alpha=0.5, bins=bins, ax=axes[it])
    axes[it].set_xlim(left=0)
    it += 1
if Type == 'Young':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age2 result_fam1r2/signal_corr/'+'WT SC nov'+'.jpg')
    plt.close()
elif Type == 'Old':
    plt.savefig('/Users/shengyuancai/Downloads/Imperial paper/Data/age10 result_fam1r2/signal_corr/'+'AD SC nov'+'.jpg')
    plt.close()




