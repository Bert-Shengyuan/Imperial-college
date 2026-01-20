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
    x2 = be_right_x[i  ]
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