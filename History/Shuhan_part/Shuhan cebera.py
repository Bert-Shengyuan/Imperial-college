#%% SHuhan
#%%
import scipy.io
import numpy as np
from cebra import CEBRA
from sklearn.decomposition import PCA
import mat73

be_data  = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/0721/linearized_behavior_after_interpolation.mat')
neu_data = np.load('/Users/sonmjack/Downloads/data_lab/0721/neural_data_after_trial_limitation.npy')
be_data_3D = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/0721/behacior_after_interpolation_xyz.mat')
triggers = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/0721/triggers.mat')
#%%
be = be_data['linearized_position'].T
be_3D = be_data_3D['continuous_index']#[:,0]
neu_data = neu_data#[12066:13066]
#这里还得处理
trigger = triggers['trials'][0,0][0]

#%%
from cebra.datasets import hippocampus

from cebra import CEBRA

max_iterations = 1000#default is 5000.
output_dimension = 32 #here, we set as a variable for hypothesis testing below.

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
                        hybrid = True)# hybrid = True

cebra_hybrid_model.fit(neu_data, be_3D)
cebra_hybrid = cebra_hybrid_model.transform(neu_data)
#%%
length = np.size(cebra_hybrid,0)
#%%
def plot_hippocampus(ax, embedding, label, gray = False, idx_order = (0,1,2)):


    p = ax.scatter(embedding [:, 0], embedding [:, 1], embedding [:, 2],s=5,c=be_3D,cmap ='viridis')
    #ax.plot(embedding[:, 0], embedding[:, 1], embedding [:, 2],c='gray', linewidth=0.5)

    ax.grid(False)
    ax.set_box_aspect([np.ptp(i) for i in embedding.T])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    return p,ax

#%%
#matplotlib notebook
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,30))

# ax1 = plt.subplot(141, projection='3d')
# ax2 = plt.subplot(142, projection='3d')
# ax3 = plt.subplot(143, projection='3d')
ax4 = plt.subplot(111, projection='3d')

# ax1=plot_hippocampus(ax1, cebra_posdir3, hippocampus_pos.continuous_index)
# ax2=plot_hippocampus(ax2, cebra_posdir_shuffled3, hippocampus_pos.continuous_index)
# ax3=plot_hippocampus(ax3, cebra_time3, hippocampus_pos.continuous_index)
p,ax4=plot_hippocampus(ax4, cebra_hybrid, be)

# ax1.set_title('CEBRA-Behavior')
# ax2.set_title('CEBRA-Shuffled')
# ax3.set_title('CEBRA-Time')
ax4.set_title('CEBRA-Hybrid')
fig.colorbar(p)
plt.savefig('/Users/sonmjack/Downloads/data_lab/cybrd_shuhan.jpg')