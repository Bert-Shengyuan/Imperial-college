#%% SHuhan
import scipy.io
import numpy as np

from cebra.datasets import hippocampus
from cebra import CEBRA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mat73

# be_data  = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/linearized_position.mat')
# neu_data = mat73.loadmat('/Users/sonmjack/Downloads/data_lab/spikes_presentation.mat')
# #%%
# be = be_data['linearized_position'].T
# cells = neu_data['spikes_presentation'].T


#%%
be_data  = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/0721/linearized_behavior_after_interpolation.mat')
neu_data = np.load('/Users/sonmjack/Downloads/data_lab/0721/neural_data_after_trial_limitation.npy')
be_data_3D = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/0721/behacior_after_interpolation_xyz.mat')
triggers = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/0721/triggers.mat')
#%%


be = be_data['linearized_position']
be_x = be_data_3D['continuous_index'][ :,0]
trigger = triggers['trials'][0,0][0]
#左右4.5 定为决策后的trigger
be_label  = np.zeros((142710,2))

#%%

#%%
for i in range(len(be_label)):
    if i == 142709:
        break
    else:
        if be_x[i] > 4.5:

            be_label[i,0] = 1 #右转
            if be_x[i-1] <= 4.5:
                be_label[i,1]  = i
            elif be_x[i+1] <= 4.5:
                    be_label[i, 1] = i*100 #trail结束
        elif  be_x[i] <= -4.5:

            be_label[i,0] = 2 #左转

            if be_x[i-1] >= -4.5:
                be_label[i,1]  = i
            elif be_x[i+1] >= -4.5:
                    be_label[i, 1] = i*100 #trail结束
        else:
            be_label[i,0] = 0

be_trigger = be_label[be_label[:,1]!=0]
path = '/Users/sonmjack/Downloads/data_lab/shuhan_trigger.npy'
np.save(path,be_trigger)
#%%
list1 = range(0,len(be_trigger),2)
#%% 左右的部分
neu = []
be = []
for i in range(0,len(be_trigger),2):
    test = be_trigger[i,1]
    neu_data1 = neu_data[int(be_trigger[i,1]):int(be_trigger[(i+1),1]/100),:]
    be_x_neu = be_data_3D['continuous_index'][int(be_trigger[i,1]):int(be_trigger[(i+1),1]/100),0]
    neu.append(neu_data1)
    be.append(be_x_neu)

#%% 中间的部分
neu = []
be = []
for i in range(1, len(be_trigger), 2):
    neu_data1 = neu_data[int(be_trigger[i,1]/100):int(be_trigger[(i+1),1]),:]
    be_x_neu = be_data_3D['continuous_index'][int(be_trigger[i,1]/100):int(be_trigger[(i+1),1]),0]
    neu.append(neu_data1)
    be.append(be_x_neu)


#%%
be_n = np.concatenate(be)
neu_n = np.concatenate(neu)
be_s= be_n .copy()
np.random.shuffle(be_s)


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
                           hybrid = True)  # hybrid = True

cebra_hybrid_model.fit(neu_n, be_n)
cebra_hybrid = cebra_hybrid_model.transform(neu_n)

length = np.size(cebra_hybrid, 0)

# %%
# matplotlib notebook
def plot(ax, embedding, label):
    p = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=0.5, c=label, cmap='viridis')
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



fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='3d')
p, ax = plot(ax, cebra_hybrid, be_n)

# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax.set_title('CEBRA-Hybrid')
fig.colorbar(p)
#plt.savefig('/Users/sonmjack/Downloads/data_lab/cybrd_shuhan'+f'_{i}'+f'_Turn_{be_trigger[i,0]}'+'.jpg')
plt.savefig('/Users/sonmjack/Downloads/data_lab/cybrd_shuhan_4.jpg')



#%%
fig = plt.figure(figsize=(20,3 ))
plt.plot(neu_data1[0:4000,0])
plt.show()

#%%
fig = plt.figure(figsize=(10,10))
plt.plot(be_x_neu)
plt.show()
#%%
for i in range(1,len(be_trigger),2):
    test = be_trigger[i,1]
    neu_data1 = neu_data[int(be_trigger[i,1]/100):int(be_trigger[(i+1),1]),:]
    be_x_neu = be_data_3D['continuous_index'][int(be_trigger[i,1]/100):int(be_trigger[(i+1),1]),0]
    # neu_data1 = neu_data[4787:5516,:]
    # be_x_neu = be_data_3D['continuous_index'][4787:5516,0]

    # neu_data1 = neu_data[9100:9506,:]
    # be_x_neu = be_data_3D['continuous_index'][9100:9506,0]
    #
    # neu_data1 = neu_data[9506:10263,:]
    # be_x_neu = be_data_3D['continuous_index'][9506:10263,0]
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

    cebra_hybrid_model.fit(neu_data1, be_x_neu)
    cebra_hybrid = cebra_hybrid_model.transform(neu_data1)

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
    plt.savefig('/Users/sonmjack/Downloads/data_lab/cybrd_shuhan'+f'_middel_{i}'+f'_Turn_{be_trigger[i,0]}'+'.jpg')
# #%%
# from PyEMD import EMD
# import numpy  as np
# import pylab as plt
# # 先做PCA
# pca = PCA (n_components=3)
# #time
# reduced_data = pca.fit_transform(neu_data1.T)
# #space
# #
# # fig = plt.figure(figsize=(20,20), dpi=10)
# #
# # # 创建一个三维散点图
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
#
# # 绘制数据点
# # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=10,c=be_array_pi,cmap ='viridis')
# # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=10)
# # #ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=10)
# # # 设置坐标轴标签
# # ax.set_xlabel('PC1')
# # ax.set_ylabel('PC2')
# # ax.set_zlabel('PC3')
# # Define signal
#
# t1 = np.linspace(0, 757*0.02, 757)
# test = 2854.2/142710
# #Define signal
# cell = np.mean(neu_data1,1)
#
#
# #%%
# # for n in range(211):
# #     t_211[:,n] = t
# IMF = EMD().emd(cell,t1)
# N = IMF.shape[0]+1
#
# fig = plt.figure(figsize=(10,10))
# # Plot results
# plt.subplot(N,1,1)
# #  画图这里也改一下
# plt.plot(t1, cell, 'r')
# plt.xlabel("Time [s]")
#
# for n, imf in enumerate(IMF):
#     plt.subplot(N,1,n+2)
#     plt.plot(t1, imf, 'g')
#
# plt.tight_layout()
# plt.savefig('simple_example')
# plt.show()







