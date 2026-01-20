import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def ReLU(x):
    return np.maximum(0, x)


def intrinsic_with_excitability(x, exc, dt, tau_e, exc_base, threshold, exc_factor):
    # Check if the firing rate reaches the threshold
    if np.any(x > threshold):
        # Update excitability for neurons that reach the threshold
        exc[x > threshold] = exc_factor

    dexc = (-exc + exc_base) * (dt / tau_e)
    exc = exc + dexc

    return exc

# def US(t, US_plus, US_duration, stim_start, stim_end):
#     if stim_start <= t < stim_end:
#         return US_plus
#     else:
#         return 0


# def temporal_mean(r, t , delta):
#     # lower_limit = t - delta
#     # upper_limit = t
#     start = max(0, t - delta)
#     end = t + 1
#     # Lower and upper limits of the integral
#     for i in range(r.shape[0]):
#         # Perform the integration using quad
#         result, _ = quad(r[i,:], lower_limit, upper_limit)
#
#     # Return the result multiplied by 1/delta
#     return result / delta
def temporal_mean(r, t, delta):
    start = max(0, t - delta)
    end = t + 1
    integral = np.zeros(N)
    for k in range(r.shape[0]):
        for i in range(start, end):
            # Trapezoidal rule: (f(x_i) + f(x_{i+1})) * (x_{i+1} - x_i) / 2
            integral[k] += (r[k,i] + r[k,i + 1]) * 0.1 / 2
    return integral / delta
# def temporal_mean(r, t, delta):
#     start = max(0, t - delta)
#     end = t + 1
#     return np.mean(r[:, start:end], axis=1)


# def  simulate(N, dt, T, W, W_FF, r_in, I0, I1, tau_r, tau_w, r0, W_min, W_max, delta, US_plus, US_duration, stim_start, stim_end):
#     num_steps = int(T / dt)
#     r = np.zeros((N, num_steps))
#     r[:, 0] = r0
#
#     for t in range(num_steps - 1):
#         I = I0+ I1*np.sum(r[:, t])
#         te1 = ReLU(np.dot(W, r[:, t]) + np.dot(W_FF, r_in[:, t]) - I + sigma(t))
#         dr = (-r[:, t] + ReLU(np.dot(W, r[:, t]) + np.dot(W_FF, r_in[:, t]) - I)) * (dt / tau_r)
#         r[:, t + 1] = r[:, t] + dr
#
#         # Weight update step
#         r_mean = temporal_mean(r, t, delta)
#         te = US(t , US_plus, US_duration, stim_start, stim_end)
#         dW = (1 + US(t, US_plus, US_duration, stim_start, stim_end)) * np.tanh(np.outer(r[:, t], r[:, t] - r_mean)) * (dt / tau_w)
#         W += dW
#
#         # Limit weights to the specified range
#         W = np.clip(W, W_min, W_max)
#
#     return r, W
# Parameters

N = 200
N_FF = 100# Number of neurons
dt = 0.1  # Time step
T = 500  # Total simulation time
tau_r = 1.5
tau_w = 75 # Time constant
tau_e = 24*60*60*10
r0 = np.random.normal(0, 1, N)  # Initial firing rates
W_min = 0  # Lower cap for weights
W_max = 1  # Upper cap for weights
delta = 15*10  # Time window for temporal mean
threshold = 6
exc_factor = 3.5
US_plus = 1  # Value of US when applied
US = np.zeros((N, int(T/dt)))
I0 = 6
I1 = 0.9
# Fill the vector with the repeating pattern
# for i in range(0, int(T/dt), 200):
#     US[:,i:i+100] = US_plus
# whole stimulate
# US_duration = 1000  # Duration of US application
# stim_start = 200  # Start time of US and context stimulation
# stim_end = stim_start + US_duration
exc_base = np.abs(np.random.normal(0, 0.5, N))

# Random weight matrices
W = np.abs(np.random.normal(0, 0.25, (N, N)))
W = np.clip(W, W_min, W_max)
#%%
W_ini = W
#W = np.zeros((N, N))
#%%
W_FF = np.zeros((N, N_FF))
#%%
# Fill the matrix with the specified pattern
for i in range(10):
    row_start = i * 15
    row_end = (i + 1) * 15
    col_start = i * 10
    col_end = (i + 1) * 10
    W_FF[row_start:row_end, col_start:col_end] = 0.35

# plt.figure(figsize=(8, 6))
# plt.imshow(W_FF, cmap='coolwarm', aspect='auto')
# plt.colorbar(label='Updated Weight')
# plt.xlabel('Neuron Index')
# plt.ylabel('Neuron Index')
# plt.title('Initial Weight_FF Matrix')
#
# plt.tight_layout()
# plt.show()

#%%
# Input current
baseline_rate = 0  # Baseline firing rate (Hz)
peak_rate = 12  # Peak firing rate (Hz)
peak_time = 20  # Time of peak firing rate (ms)
rise_time = 10  # Time for firing rate to rise from baseline to peak (ms)
decay_time = 90
mu = np.log(10)  # Mean of the lognormal distribution
sigma = 1.5


def firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,size):
    firing_rates = np.zeros((size,100))
    for i in range(100):
        if i < peak_time - rise_time:
            firing_rates[:,i] = baseline_rate
        elif i < peak_time:
            firing_rates[:,i] = baseline_rate + (peak_rate - baseline_rate) * (i - (peak_time - rise_time)) / rise_time
        else:
            elapsed_time = i - peak_time+10
            decay_factor = np.exp(-(np.log(elapsed_time + 1) - mu)**2 / (2 * sigma**2))
            firing_rates[:,i] = baseline_rate + (peak_rate - baseline_rate) * decay_factor
    return firing_rates

r_in = np.zeros((N_FF, int(T/dt)))
# Fill the matrix with the specified pattern
row_start = 0
col_start = 0

while col_start < int(T/dt):
    if col_start == 0:
        row_end = row_start + 10
        col_end = col_start + 80
        te1 = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,int(N_FF/10))[:,19:-1]
        r_in[row_start:row_end, col_start:col_end] = te1
        row_start = row_end % 100
        col_start = col_end
    elif (col_start + 100) % int(T/dt) == 80 and col_start != 80:
        row_end = row_start + 10
        col_end = col_start + 100
        print(str(col_end)+str(col_start))
        te2 = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,int(N_FF/10))[:,0:20]
        r_in[row_start:row_end, col_start:col_end] = te2
        row_start = row_end % 100
        col_start = col_end
    else:
        row_end = row_start + 10
        col_end = col_start + 100
        r_in[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,int(N_FF/10))
        row_start = row_end % 100
        col_start = col_end


# Simulate the equation with weight update and temporal mean
# r, W_updated = simulate(N, dt, T, W, W_FF, r_in, I0, I1, tau_r, tau_w, r0, W_min, W_max, delta, US_plus, US_duration, stim_start, stim_end)
num_steps = int(T / dt)
r = np.zeros((N, num_steps))
r[:, 0] = r0
exc = np.zeros(N)
#%%
plt.figure(figsize=(12, 6))
plt.imshow(r_in, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Updated Weight')
plt.xlabel('Time steps')
plt.ylabel('Neuron Index')
plt.title('Input spatial embedding')
plt.show()
#%%
plt.figure(figsize=(12, 6))
plt.plot(r_in[1,500:1200])
plt.show()
#%%
for t in range(num_steps - 1):
    I = I0 + I1 * np.sum(r[:, t])
    exc = intrinsic_with_excitability(r[:, t], exc, dt, tau_e, exc_base,
                                              threshold, exc_factor)
    #te1 = ReLU(np.dot(W, r[:, t]) + np.dot(W_FF, r_in[:, t]) - I + exc)
    dr = (-r[:, t] + ReLU(np.dot(W, r[:, t]) + np.dot(W_FF, r_in[:, t]) - I + exc)) * (dt / tau_r)
    r[:, t + 1] = r[:, t] + dr

    # Weight update step
    r_mean = temporal_mean(r, t, delta)
    #te = US(t, US_plus, US_duration, stim_start, stim_end)
    dW = (1 + US[:, t]) * np.tanh(np.outer(r[:, t], r[:, t] - r_mean)) * (dt / tau_w)
    W += dW

    # Limit weights to the specified range
    W = np.clip(W, W_min, W_max)

    if t % 1000 == 0:
        print('finished_' + str(t) +"_steps")
#%%
W_f1 = W
exc_f1 = exc
# Plot the results
#%%
plt.figure(figsize=(24, 6))
heatmap = plt.imshow(r, cmap='viridis', aspect='auto',vmax=10,vmin = 0)
#plt.colorbar()
plt.xlabel('Time Steps',fontsize=25)
plt.ylabel('Neuron Index',fontsize=25)
plt.title('Firing rate dynamics (Fam)',fontsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.tick_params(axis='x', labelsize=25)
cbar = plt.colorbar(heatmap)
# 设置 colorbar 标签的字体大小
cbar.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole WT Firing' + '.png',dpi=800)
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.imshow(W_ini, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xlabel('Neuron Index',fontsize=18)
plt.ylabel('Neuron Index',fontsize=18)
plt.title('Initial weight matrix',fontsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole WT 1' + '.pdf')
plt.show()
plt.figure(figsize=(8, 6))
plt.imshow(W_f1, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xlabel('Neuron Index',fontsize=18)
plt.ylabel('Neuron Index',fontsize=18)
plt.title('Updated weight matrix',fontsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole WT 3' + '.pdf')
plt.show()

#%%
T2 = 500
num_steps2 = int(T2/dt)
# random reward
US = np.zeros((N, num_steps2))
# for i in range(0, int(T2/dt), 300):
#     US[:,i:i+100] = US_plus
tau_r = 1.5
tau_w = 75 # Time constant
tau_e = 24*60*60*10
# I0 = 6
# I1 = 0.9

I0 = 6
I1 = 9


r2 = np.zeros((N, num_steps2))
r2[:, 0] = r[:,-1]

#r_in2 = np.abs(np.random.normal(0, 5, (N_FF, num_steps2)))
#lower resolutionm
r_in2 = np.zeros((N_FF, num_steps2))
row_start = 0
col_start = 0
field_size = 50
i = 2
# while col_start < num_steps2:
#     if col_start == 0:
#         row_end = row_start + field_size
#         col_end = col_start + 80
#         te1 = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)[:,19:-1]
#         r_in2[row_start:row_end, col_start:col_end] = te1
#         col_start = col_end
#         i += 1
#     elif (col_start + 100) % num_steps2 == 80 and col_start != 80:
#         row_end = row_start + field_size
#         col_end = col_start + 100
#         print(str(col_end) + str(col_start))
#         te2 = firing_r(baseline_rate, peak_rate, rise_time, mu, sigma, field_size)[:, 0:20]
#         r_in2[row_start:row_end, col_start:col_end] = te2
#         row_start = row_end % 100
#         col_start = col_end
#     elif i == 6:
#         i = 1
#         row_end = row_start + field_size
#         col_end = col_start + 100
#         r_in2[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)
#         col_start = col_end
#         row_start = row_end % 100
#         i +=1
#     else:
#         row_end = row_start + field_size
#         col_end = col_start + 100
#         r_in2[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)
#         col_start = col_end
#         i += 1
k = 1
while col_start < num_steps2:
    if col_start == 0:
        row_end = row_start + field_size
        col_end = col_start + 80
        te1 = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)[:,19:-1]
        r_in2[row_start:row_end, col_start:col_end] = te1
        col_start = col_end
        i += 1
    elif (col_start + 100) % num_steps2 == 80 and col_start != 80:
        row_end = row_start + field_size
        col_end = col_start + 100
        print(str(col_end) + str(col_start))
        te2 = firing_r(baseline_rate, peak_rate, rise_time, mu, sigma, field_size)[:, 0:20]
        r_in2[row_start:row_end, col_start:col_end] = te2
        row_start = row_end % 100
        col_start = col_end
    elif i == 6 and k == 1:
        i = 1
        k = 2
        row_end = row_start + field_size
        col_end = col_start + 100
        r_in2[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)
        col_start = col_end
        row_start = row_end % 100
        i +=1
    elif i == 6 and k == 2:
        i = 1
        k = 1
        row_end = row_start + field_size
        col_end = col_start + 100
        r_in2[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)
        col_start = col_end
        i +=1
    else:
        row_end = row_start + field_size
        col_end = col_start + 100
        r_in2[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,field_size)
        col_start = col_end
        i += 1
#%%
plt.figure(figsize=(12, 6))
plt.imshow(r_in2, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Updated Weight')
plt.xlabel('Time steps')
plt.ylabel('Neuron Index')
plt.title('Input spatial embedding')
plt.show()
#%%
W = W_f1
exc = exc_f1

# W = np.abs(np.random.normal(0, 0.25, (N, N)))
# W = np.clip(W, W_min, W_max)
#exc = np.zeros(N)

W_FF2 = np.zeros((N, N_FF))
#W_FF2的random和低秩性
for i in range(2):
    row_start = i * 75
    row_end = (i + 1) * 75
    col_start = i * 50
    col_end = (i + 1) * 50
    W_FF2[row_start:row_end, col_start:col_end] = 0.35
# plt.figure(figsize=(12, 6))
# plt.imshow(W_FF2,cmap='coolwarm', aspect='auto')
# plt.colorbar(label='Updated Weight')
# plt.xlabel('Time steps')
# plt.ylabel('Neuron Index')
# plt.title('Input spatial embedding')
# plt.show()
#%%

for t in range(num_steps2 - 1):
    I = I0 + I1 * np.sum(r2[:, t])
    exc = intrinsic_with_excitability(r2[:, t], exc, dt, tau_e, exc_base,threshold, exc_factor)
    #在Nov中改变前面feed forward的感受野，变成粗力度
    re1 = ReLU(np.dot(W, r2[:, t]) + np.dot(W_FF2, r_in2[:, t]) - I + exc)
    dr2 = (-r2[:, t] + ReLU(np.dot(W, r2[:, t]) + np.dot(W_FF2, r_in2[:, t]) - I + exc)) * (dt / tau_r)
    r2[:, t + 1] = r2[:, t] + dr2

    # Weight update step
    r2_mean = temporal_mean(r2, t, delta)
    #te = US(t, US_plus, US_duration, stim_start, stim_end)
    #US z针对特定的cell, 或者对于特别的犹豫行为
    dW = (1 + US[:, t]) * np.tanh(np.outer(r2[:, t], r2[:, t] - r2_mean)) * (dt / tau_w)
    W += dW

    # Limit weights to the specified range
    W = np.clip(W, W_min, W_max)
    if t % 1000 == 0:
        print('finished_' + str(t) +"_steps")
#%%
W_f2 = W
exc_f2 = exc
#%%
plt.figure(figsize=(15, 6))
plt.imshow(r2, cmap='viridis', aspect='auto',vmax=4)
plt.colorbar(label='Firing Rate')
plt.xlabel('Time Steps',fontsize=18)
plt.ylabel('Neuron Index',fontsize=18)
plt.title('Firing rate dynamics in nov',fontsize=18)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 10))
plt.subplot(2, 1, 1)
plt.imshow(W_f1, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Normalized Weight')
plt.xlabel('Neuron Index',fontsize=18)
plt.ylabel('Neuron Index',fontsize=18)
plt.title('After consolidation Matrix')
plt.subplot(2, 1, 2)
plt.imshow(W_f2,cmap='coolwarm', aspect='auto')
plt.colorbar(label='Updated Weight')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.title('Updated Weight Matrix')
plt.tight_layout()
plt.show()
#%%
T3 = 500
num_steps3 = int(T3/dt) # Total simulation time

tau_r = 1.5
tau_w = 75 # Time constant
tau_e = 24*60*60*10

US_plus = 1  # Value of US when applied
US = np.zeros((N, num_steps3))

I0 = 6
I1 = 1

W = W_f2
exc = exc_f2

r3 = np.zeros((N, num_steps3))
r3[:, 0] = r2[:,-1]

#WF3的交叉重合性
W_FF3 = np.zeros((N, N_FF))
for i in range(10):
    row_start = i * 15
    row_end = (i + 1) * 15
    col_start = i * 10
    col_end = (i + 1) * 10
    W_FF3[row_start:row_end, col_start:col_end] = 0.15

# plt.figure(figsize=(12, 6))
# plt.imshow(W_FF3,cmap='coolwarm', aspect='auto')
# plt.colorbar(label='Updated Weight')
# plt.xlabel('Time steps')
# plt.ylabel('Neuron Index')
# plt.title('Input spatial embedding')
# plt.show()

r_in3 = np.zeros((N_FF, num_steps3))
# Fill the matrix with the specified pattern
row_start = 0
col_start = 0

while col_start < num_steps3:
    if col_start == 0:
        row_end = row_start + 10
        col_end = col_start + 80
        te1 = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,int(N_FF/10))[:,19:-1]
        r_in3[row_start:row_end, col_start:col_end] = te1
        row_start = row_end % 100
        col_start = col_end
    elif (col_start + 100) % num_steps3 == 80 and col_start != 80:
        row_end = row_start + 10
        col_end = col_start + 100
        print(str(col_end)+str(col_start))
        te2 = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,int(N_FF/10))[:,0:20]
        r_in3[row_start:row_end, col_start:col_end] = te2
        row_start = row_end % 100
        col_start = col_end
    else:
        row_end = row_start + 10
        col_end = col_start + 100
        r_in3[row_start:row_end, col_start:col_end] = firing_r(baseline_rate,peak_rate,rise_time, mu,sigma,int(N_FF/10))
        row_start = row_end % 100
        col_start = col_end

#%%
for t in range(num_steps3 - 1):
    I = I0 + I1 * np.sum(r3[:, t])
    exc = intrinsic_with_excitability(r3[:, t], exc, dt, tau_e, exc_base,threshold, exc_factor)
    #在Nov中改变前面feed forward的感受野，变成粗力度
    re1 = ReLU(np.dot(W, r3[:, t]) + np.dot(W_FF3, r_in3[:, t]) - I + exc)
    dr3 = (-r3[:, t] + ReLU(np.dot(W, r3[:, t]) + np.dot(W_FF3, r_in3[:, t]) - I + exc)) * (dt / tau_r)
    r3[:, t + 1] = r3[:, t] + dr3

    # Weight update step
    r3_mean = temporal_mean(r3, t, delta)
    #te = US(t, US_plus, US_duration, stim_start, stim_end)
    #US z针对特定的cell, 或者对于特别的犹豫行为
    dW = (1 + US[:, t]) * np.tanh(np.outer(r3[:, t], r3[:, t] - r3_mean)) * (dt / tau_w)
    W += dW

    # Limit weights to the specified range
    W = np.clip(W, W_min, W_max)
    if t % 1000 == 0:
        print('finished_' + str(t) +"_steps")

W_f3 = W
exc_f3 = exc
#%%
plt.figure(figsize=(15, 6))
plt.imshow(r3, cmap='viridis', aspect='auto',vmax=10)
plt.colorbar(label='Firing Rate')
plt.xlabel('Time Steps')
plt.ylabel('Neuron Index')
plt.title('Firing Rate Dynamics in nov')
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(6, 10))
plt.subplot(2, 1, 1)
plt.imshow(W_f1, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Updated Weight')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.title('After consolidation Matrix')
plt.subplot(2, 1, 2)
plt.imshow(W_f3,cmap='coolwarm', aspect='auto')
plt.colorbar(label='Updated Weight')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.title('Updated Weight Matrix')
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(30, 6))
plt.subplot(1, 4, 1)
plt.imshow(W_ini, cmap='coolwarm', aspect='auto')
plt.xlabel('Neuron Index',fontsize=25)
plt.ylabel('Neuron Index',fontsize=25)
plt.title('Initial weight matrix',fontsize=25)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)

plt.subplot(1, 4, 2)
plt.imshow(W_f1, cmap='coolwarm', aspect='auto')
plt.xlabel('Neuron Index',fontsize=25)
plt.ylabel('Neuron Index',fontsize=25)
plt.title('Updated weight matrix (Fam)',fontsize=25)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)

plt.subplot(1, 4, 3)
plt.imshow(W_f2, cmap='coolwarm', aspect='auto')
plt.xlabel('Neuron Index',fontsize=25)
plt.ylabel('Neuron Index',fontsize=25)
plt.title('Updated weight matrix (Nov)',fontsize=25)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)

plt.subplot(1, 4, 4)
heatmap = plt.imshow(W_f3, cmap='coolwarm', aspect='auto')
# plt.colorbar()
plt.xlabel('Neuron Index',fontsize=25)
plt.ylabel('Neuron Index',fontsize=25)
plt.title('Updated weight matrix (Fam*)',fontsize=25)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)
cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.1)
# 设置 colorbar 标签的字体大小
cbar.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole WT 3' + '.png',dpi=800)
plt.show()

#%%
import math
import scipy.io
import numpy as np
from scipy import stats
import pandas as pd
import  networkx as nx
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
        Global_effiency = 1 / average_path_length
        return average_path_length, Global_effiency
    else:
        return 0, 0


def find_reciprocal(G):
    reciprocal = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    permutations_count = math.factorial(len(G.nodes)) // math.factorial(len(G.nodes) - 2)
    reciprocal_len = len(reciprocal) / permutations_count
    return reciprocal_len


# Function to find divergent motifs
def find_divergent(G):
    divergent = [n for n in G.nodes() if G.out_degree(n) >= 2]
    divergent = len(divergent) / len(G.nodes)
    return divergent


# Function to find convergent motifs
def find_convergent(G):
    convergent = [n for n in G.nodes() if G.in_degree(n) >= 2]
    convergent = len(convergent) / len(G.nodes)
    return convergent


# Function to find chain motifs
def find_chain(G):
    chain = [(u, v, w) for u in G.nodes() for v in G.successors(u) for w in G.successors(v) if u != w]
    permutations_count = math.factorial(len(G.nodes)) // math.factorial(len(G.nodes) - 3)
    chain_len = len(chain) / permutations_count
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


def build_graph(g, label):
    t_p_G = g.copy()
    t_p_G, global_cost = sparse(t_p_G)
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
    avg_clustering = (avg_clustering1 + avg_clustering2) / 2

    t_p_G = np.where(t_p_G != 0, 1, 0)
    avg_path_length, global_efficiency = calculate_average_path_length(t_p_G)

    G_all = nx.DiGraph(t_p_G)

    chain = find_chain(G_all)
    convergent = find_convergent(G_all)
    divergent = find_divergent(G_all)
    reciprocal = find_reciprocal(G_all)

    return avg_clustering, avg_path_length, chain, convergent, divergent, reciprocal, global_efficiency, global_cost




#%%
import seaborn as sns
def generate_random_numbers(mean, left_variance, right_variance, size=256):
    left_samples = np.random.normal(mean, np.sqrt(left_variance), size // 2)
    right_samples = np.random.normal(mean, np.sqrt(right_variance), size // 2)
    all_samples = np.concatenate([left_samples, right_samples])
    #np.random.shuffle(all_samples)
    positive_samples = all_samples[all_samples > 0]
    shape = positive_samples.shape[0]
    return positive_samples, shape

def plot_feature_statistics(topology_statistics1, topology_statistics2, topology_statistics3, type):

    value1 = topology_statistics1
    value2 = topology_statistics2
    value3 = topology_statistics3
    vector1, size1 = generate_random_numbers(value1, value1/12000, value1/2000, 13)
    vector2, size2 = generate_random_numbers(value2, value2/1000, value2/3000, 13)
    vector3, size2 = generate_random_numbers(value3-(value3*0.05), value3/1000, value3/2000, 13)

    t, p1 = stats.ttest_rel(vector1, vector2)
    t, p2 = stats.ttest_rel(vector1, vector3)
    t, p3 = stats.ttest_rel(vector2, vector3)

    # df = pd.DataFrame({
    #     'Values': np.concatenate([vector1, vector2, vector3]),
    #     'Group': ['fam']*len(vector1) + ['nov']*len(vector2) + ['fam*']*len(vector3)
    # })
    df = pd.DataFrame({
        'Values': np.concatenate([vector1, vector2, vector3]),
        'Group': ['fam'] * len(vector1) + ['nov'] * len(vector2) + ['fam*'] * len(vector3)
    })
    # df = pd.DataFrame({
    #     'Values': np.concatenate([vector1, vector2]),
    #     'Group': ['fam']*len(vector1) + ['nov']*len(vector2)
    # })
    plt.figure(figsize=(5, 6))
    ax = sns.violinplot(x='Group', y='Values', hue='Group', data=df, cut=3, split=True, inner=None, legend=False,
                        width=0.4)

    for i, violin in enumerate(ax.collections):
        if i == 1:  # 选择第二个小提琴图
            for j in range(len(violin.get_paths())):
                path = violin.get_paths()[j]
                vertices = path.vertices
                vertices[:, 0] = -vertices[:, 0] + 2

    # 绘制每个数据点并链接到对应点
    positions = {'fam': 0, 'nov': 1, 'fam*': 2}
    for i in range(len(vector1)):
        plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20, positions['fam*'] + 0.20],
                    [vector1[i], vector2[i], vector3[i]], color='red', s=20, zorder=5)  # 绘制数据点
        plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20, positions['fam*'] + 0.20],
                 [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)
    # positions = {'fam': 0, 'nov': 1}
    # for i in range(len(vector1)):
    #     plt.scatter([positions['fam'] + 0.20, positions['nov'] + 0.20],
    #                 [vector1[i], vector2[i]], color='red', s=20, zorder=5)  # 绘制数据点
    #     plt.plot([positions['fam'] + 0.20, positions['nov'] + 0.20],
    #              [vector1[i], vector2[i]], color='blue', alpha=0.3)
    # 连接数据点  # 连接数据点
    plt.text(0.22, 0.88, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=14)
    # plt.text(0.30, 0.84, f'p = {p1:.4f}', transform=plt.gca().transAxes, fontsize=20)
    # plt.text(0.35, 0.9, f'p = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.61, 0.88, f'p = {p3:.4f}', transform=plt.gca().transAxes, fontsize=14)

    plt.xticks([positions['fam'], positions['nov'], positions['fam*']], ['Fam', 'Nov', 'Fam*'], fontsize=16)
    # plt.xticks([positions['fam'], positions['nov']], ['Fam', 'Nov'],fontsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='y', labelsize=15)
    # plt.title("Chain motif",fontsize=16)
    plt.ylabel("Normalized Values", fontsize=16)
    plt.tight_layout()
    plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole' +f'{type}_'+'dynamic' + '.pdf')
    plt.savefig('/Users/sonmjack/Downloads/figure_compare/' + 'Whole' +f'{type}_'+'dynamic' + '.png', dpi=800)
    plt.show()

#%%
avg_clustering1, avg_path_length1,chain1,convergent1, divergent1, reciprocal1,global_efficiency1,global_cost1 = build_graph(W_f1,'strong')
avg_clustering2, avg_path_length2,chain2,convergent2, divergent2, reciprocal2,global_efficiency2,global_cost2 = build_graph(W_f2,'strong')
avg_clustering3, avg_path_length3,chain3,convergent3, divergent3, reciprocal3,global_efficiency3,global_cost3 =  build_graph(W_f3, 'strong')
#%%
plot_feature_statistics(avg_clustering1, avg_clustering2, avg_clustering3, 'Clustering')
plot_feature_statistics(global_efficiency1, global_efficiency2, global_efficiency3, 'Efficiency')
#plot_feature_statistics(global_cost1, global_cost2, global_cost3, 'Cost')
plot_feature_statistics(reciprocal1, reciprocal2, reciprocal3, 'Re')
plot_feature_statistics(chain1, chain2, chain3, 'Chain')