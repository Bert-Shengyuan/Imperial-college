#%%
import matplotlib.pyplot as plt
import numpy as np


#%%  角度
# be_right_x = []
# be_right_y = []
# neuron_right = []
#
# color_tensor = np.zeros([neuron_right.shape[0],1000,2])
#
# top_n_indices = np.argpartition(neuron_right, -1000, axis=1)[:, -1000:]
#
# for i in range(top_n_indices.shape[0]):
#     for j in range(top_n_indices.shape[1]):
#         color_tensor[i, j, 0] = be_right_x[top_n_indices[i,j]]
#         color_tensor[i, j, 1] = be_right_y[top_n_indices[i, j]]
#
# # 找到每一行最大的6个值对应的列索引
# color_right = np.mean(color_tensor, axis=1)

#%%
# be_left_x = []
# be_left_y = []
# neuron_left = []
#
# color_tensor = np.zeros([neuron_left.shape[0], 1000, 2])
#
# top_n_indices = np.argpartition(neuron_left, -1000, axis=1)[:, -1000:]
#
# for i in range(top_n_indices.shape[0]):
#     for j in range(top_n_indices.shape[1]):
#         color_tensor[i, j, 0] = be_left_x[top_n_indices[i, j]]
#         color_tensor[i, j, 1] = be_left_y[top_n_indices[i, j]]
#
# # 找到每一行最大的6个值对应的列索引
# color_left = np.mean(color_tensor, axis=1)
#%%
import math
def calculate_angle(x, y):
    # 计算两点之间的水平和垂直距离
    dx = x - 0
    dy = y - 0
    # 使用反正切函数计算角度（以弧度为单位）
    angle_rad = math.atan2(dy, dx)
    # 将弧度转换为度数
    angle_deg = math.degrees(angle_rad)
    return angle_deg

angle_r = []
angle_l = []

be_right_x = np.load('/Users/sonmjack/Downloads/Shuhan_new/be_right_x.npy').reshape(-1)
be_right_y = np.load('/Users/sonmjack/Downloads/Shuhan_new/be_right_y.npy').reshape(-1)
for i in range(len(be_right_x)):
    # x1 = be_left_x[i]
    # y1 = be_left_y[i]
    x2 = be_right_x[i]
    y2 = be_right_y[i]
    # angle_l.append(calculate_angle(x1, y1))
    angle_r.append(calculate_angle(x2, y2))
angle_r = np.array(angle_r).reshape(-1)
# angle_l = np.array(angle_l).reshape(1,-1)
#%%  整体发放颜色
neuron_right = np.load('/Users/sonmjack/Downloads/Shuhan_new/neuron_right.npy')
import matplotlib.colors as mcolors
cmap = plt.get_cmap('viridis')
#%%
for j in range(np.size(neuron_right,0)):# neuron_spike是neuron_right or neuron_left
    neuron = neuron_right[j, :]
    theta = np.deg2rad(angle_r)#angle_data是angle_r或angle_l
    norm = mcolors.Normalize(vmin=min(neuron), vmax=max(neuron))
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    # 将角度数据映射到环上，绘制211个point，每个point的颜色不同
    for i in range(np.size(neuron, 0)):  # 将角度转换为弧度
        radii = 1  # 小块的半径可以根据需要设置
        color = cmap(norm(neuron[i]))  # 根据发放值获取颜色
        ax.plot(theta[i], radii, marker='o', markersize=5, color=color)
    ax.set_rticks([np.pi / 2])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.savefig('/Users/sonmjack/Downloads/Shuhan_new/graph/' + 'phi-right-neuron' + f'{j}.jpg')
    plt.close()






#%%
#%%
import math
#%%
import matplotlib.pyplot as plt
import numpy as np
def calculate_angle(x, y):
    # 计算两点之间的水平和垂直距离
    dx = x - 0
    dy = y - 0
    # 使用反正切函数计算角度（以弧度为单位）
    angle_rad = math.atan2(dy, dx)
    # 将弧度转换为度数
    angle_deg = math.degrees(angle_rad)
    return angle_deg

angle_r = []
angle_l = []

be_right_x = np.load('/Users/sonmjack/Downloads/Shuhan_new/be_right_x.npy').reshape(-1)
be_right_y = np.load('/Users/sonmjack/Downloads/Shuhan_new/be_right_y.npy').reshape(-1)
be_left_x = np.load('/Users/sonmjack/Downloads/Shuhan_new/be_left_x.npy').reshape(-1)
be_left_y = np.load('/Users/sonmjack/Downloads/Shuhan_new/be_left_y.npy').reshape(-1)

for i in range(len(be_right_x)):
    x2 = be_right_x[i]
    y2 = be_right_y[i]
    angle_r.append(calculate_angle(x2, y2))
for i in range(len(be_left_x)):
    x1 = be_left_x[i]
    y1 = be_left_y[i]
    angle_l.append(calculate_angle(x1, y1))

angle_l = np.array(angle_l).reshape(-1)
angle_r = np.array(angle_r).reshape(-1)

angle_all = np.concatenate((angle_r,angle_l))
#%%  整体发放颜色
neuron_right = np.load('/Users/sonmjack/Downloads/Shuhan_new/neuron_right.npy')
neuron_left = np.load('/Users/sonmjack/Downloads/Shuhan_new/neuron_left.npy')
neuron_all = np.concatenate((neuron_right,neuron_left),axis=1)
import matplotlib.colors as mcolors
cmap = plt.get_cmap('viridis')
#%%
for j in range(np.size(neuron_right,0)):# neuron_spike是neuron_right or neuron_left
    neuron = neuron_all[j, :]
    theta = np.deg2rad(angle_all)#angle_data是angle_r或angle_l
    norm = mcolors.Normalize(vmin=min(neuron)+0.1*(max(neuron)-min(neuron)), vmax=max(neuron))
    # 创建极坐标图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    # 将角度数据映射到环上，绘制211个point，每个point的颜色不同
    for i in range(np.size(neuron, 0)):  # 将角度转换为弧度
        radii = 1  # 小块的半径可以根据需要设置
        color = cmap(norm(neuron[i]))  # 根据发放值获取颜色
        ax.plot(theta[i], radii, marker='o', markersize=5, color=color)
    ax.set_rticks([np.pi / 2])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.savefig('/Users/sonmjack/Downloads/Shuhan_new/graph/' + 'phi-right-neuron' + f'{j}.jpg')
    plt.close()