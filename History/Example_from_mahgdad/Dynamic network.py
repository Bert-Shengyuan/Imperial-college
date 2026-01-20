
import math
from joblib import Parallel, delayed
import random
import numpy as np

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
def Splitter(freq, i, tau):
    Spike_train = np.load('/rds/general/user/msaeedia/home/Results/Brain_no-neuropil_V1/Spi_Train/spik_train.npy',
                          allow_pickle=True)

    Result = []
    for k in Nodes[l]:
        Result.append(Strength_computer(Spike_train, i, k, tau))
        np.save('/rds/general/user/msaeedia/home/Results/Brain_no-neuropil_V1/Adj_Nod/Adj_EPSP_shuf_freq_%.1f_N_%d_tau_fixed_%d_row_%d' % (
        freq, N, tau, i), Result)

    print(i)
import pickle
with open('/Users/sonmjack/Downloads/simon_paper/neuron_list.pkl', 'rb') as file:
    neuron_spike = pickle.load(file)
freq = 30.9
tau = 5
dy_list = []

for k in range(len(neuron_spike)):
    Spike_train = neuron_spike[k]
    dy_distance = []
    num_features = neuron_spike[k].shape[0]
    for m in range(num_features):
        for n in range(m + 1, num_features):
            dy_distance.append(Strength_computer(Spike_train, m, n, tau))
    dy_list.append(dy_distance)
# #%%
# freq = 7.5
# Tau = [1, 10]
#
# N = 3000  # number of nourons
#
# test = neuron_spike[int(1)].sort()
# #%%
#
# results = Parallel(n_jobs=256)(delayed(Splitter)(freq, i, tau) for i in Nodes[l] for tau in Tau)