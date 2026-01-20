#%%
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
test = normalized_mutual_info_score([2,0,1,1,1,0,1,1], [0,1,1,1,0,1,1,2])

from sklearn.feature_selection import mutual_info_classif
#test = mutual_info_regression([1,0,0,1], [1,1, 0,0])
#改变顺序会改变边际分布
#test = mutual_info_score([2,0,1,1,1,0,1,1], [0,1,1,1,0,1,1,2])
#%%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(0)
X = np.random.rand(1000, 3)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y, edgecolor="black", s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
plt.show()

y1 =y.T
#%%
import numpy as np

# Create 10 random 172x172 matrices
num_matrices = 10
matrix_shape = (172, 172)

matrices = [np.random.random(matrix_shape) for _ in range(num_matrices)]

# Convert the matrices into a three-dimensional tensor
tensor = np.array(matrices)

# Calculate the average tensor along the specified axis (axis=0 for the first dimension)
average_tensor = np.mean(tensor, axis=0)

# Print the result
print("Average Tensor:")
print(average_tensor)

#%%  dataloader

import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import mat73
import math

mat_data  = mat73.loadmat('/Users/sonmjack/Downloads/data_lab/data_fam1_timeseries.mat')
be_data  = scipy.io.loadmat('/Users/sonmjack/Downloads/data_lab/data_fam1_trackdata.mat')

cell_array = mat_data['spikes'][0:181]
num_rows = len(cell_array)
num_column = len(cell_array[0][0])
neurons = np.zeros((num_rows, num_column))
for z in range(num_rows):
    neurons[z, :] = (cell_array[z][0] * 10).astype(int)
#%%
delay = 3/32
sample = 1/32
#neurons = np.array([[0,0,1,0,0,1],[1,0,0,1,0,0]])
num_features = neurons.shape[0]
distance = np.zeros((num_features, num_features))
for m in range(num_features):
    for n in range(num_features):
        f_list = []
        for i in range(neurons[m, :].shape[0]):
            if neurons[m, i] != 0:
                for j in range(i, -1, -1):
                    if neurons[n, j] !=0:
                        t = sample*(i-j)
                        f_list.append(math.exp(-t/delay))
                        break
        if f_list == []:
            f = 0
        else:
            f_list = np.array(f_list)
            #f_list = f_list/max(f_list)
            N_m = np.count_nonzero(neurons[m, :])
            N_n = np.count_nonzero(neurons[n, :])
            f = np.sum(f_list)/max(N_n,N_m)
        distance[n, m] = f

#%%
distance2 = np.zeros((num_features, num_features))
for m in range(num_features):
    for n in range(num_features):
        f_list = []
        for i in range(neurons[m, :].shape[0]):
            if neurons[m, i] != 0:
                for j in range(i, -1, -1):
                    if neurons[n, j] !=0:
                        t = sample*(i-j)
                        f_list.append(1-(math.exp(-t/delay)))
                        break
        if f_list == []:
            f = 0
        else:
            f_list = np.array(f_list)
            #f_list = f_list/max(abs(f_list))
            N_m = np.count_nonzero(neurons[m, :])
            N_n = np.count_nonzero(neurons[n, :])
            f = np.sum(f_list)/max(N_n,N_m)
        distance2[n, m] = f
#%%
import matplotlib.pyplot as plt
np.fill_diagonal(distance, 0)
plt.imshow(distance, cmap='viridis', interpolation='none')

# 添加颜色条
plt.colorbar()

# 显示图形
plt.show()

np.fill_diagonal(distance2, 0)
plt.imshow(distance2, cmap='viridis', interpolation='none')

# 添加颜色条
plt.colorbar()

# 显示图形
plt.show()

#%%

plt.figure(figsize=(20,5))#2,110
plt.plot(neurons[0,:],label = 'neuron1')
plt.plot(neurons[37,:],label = 'neuron37')
plt.legend()
plt.show()


plt.figure(figsize=(20,5))#2,110
plt.plot(neurons[2,:],label = 'neuron2')
plt.plot(neurons[110,:],label = 'neuron110')
plt.legend()
plt.show()
#%%
from sklearn.metrics import mutual_info_score
import numpy as np

# 两个离散变量的观测值
variable1 = np.array([1, 2, 1, 2, 1, 2])
variable2 = np.array([0, 1, 1, 0, 1, 0])

# 构建列联表
contingency_table = np.histogram2d(variable1, variable2, bins=(2, 2))[0]

# 计算互信息
mi = mutual_info_score(None, None, contingency=contingency_table)

print("Mutual Information:", mi)

#%%
import numpy as np

from scipy.signal import find_peaks

# 生成两个示例序列
sequence1 = np.sin(np.linspace(0, 20, 100))
sequence2 = np.cos(np.linspace(0, 20, 100))

# 计算两个序列的 DTW 距离
distance_matrix = cdist(sequence1.reshape(-1, 1), sequence2.reshape(-1, 1), 'euclidean')
accumulated_cost_matrix = np.zeros_like(distance_matrix)


for i in range(1, len(sequence1)):
    for j in range(1, len(sequence2)):
        accumulated_cost_matrix[i, j] = distance_matrix[i, j] + min(
            accumulated_cost_matrix[i-1, j],
            accumulated_cost_matrix[i, j-1],
            accumulated_cost_matrix[i-1, j-1]
        )


# 获取规范化的 DTW 距离
dtw_distance = accumulated_cost_matrix[-1, -1] / sum(accumulated_cost_matrix.shape)

# 打印结果
print(f"DTW距离: {dtw_distance}")

# 如果需要，可以找到 DTW 对齐路径
path = []
i, j = len(sequence1) - 1, len(sequence2) - 1

while i > 0 or j > 0:
    path.append((i, j))
    if i == 0:
        j -= 1
    elif j == 0:
        i -= 1
    else:
        min_idx = np.argmin([accumulated_cost_matrix[i-1, j], accumulated_cost_matrix[i, j-1], accumulated_cost_matrix[i-1, j-1]])
        if min_idx == 0:
            i -= 1
        elif min_idx == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

# 反转路径
path = path[::-1]

# 打印对齐路径
print(f"对齐路径: {path}")

#%%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(0)
X = np.random.rand(1000, 1)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 0]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)


plt.figure(figsize=(30, 5))
for i in range(1):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y, edgecolor="black", s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
plt.show()

#%%
import numpy as np

# 生成10个1x100的行向量
row_vectors_list = [np.random.rand(1, 100) for _ in range(10)]

# 使用numpy.concatenate拼接行向量
concatenated_matrix = np.concatenate(row_vectors_list, axis=0)

# 复制拼接后的矩阵，以免影响原始数据
shuffled_matrix = concatenated_matrix.copy()

# 随机洗牌
np.random.shuffle(shuffled_matrix)

# 打印洗牌后的矩阵的形状
print(shuffled_matrix.shape)

#%%
import matplotlib.pyplot as plt
import numpy as np

# 数据
frequencies = np.array([1350, 1320, 1295, 1270, 1240, 1215])
frequency_errors = np.array([10, 10, 10, 10, 5, 5])
inverse_diameters = np.array([0.44, 0.43, 0.42, 0.41, 0.40, 0.39])
inverse_diameter_errors = np.array([0.02, 0.01, 0.02, 0.03, 0.01, 0.01])

# 线性拟合
fit_params, cov_matrix = np.polyfit(frequencies, inverse_diameters, 1, w=1/frequency_errors, cov=True)
fit_line = np.poly1d(fit_params)

# 计算拟合直线的值
x_fit = np.linspace(frequencies.min() - 20, frequencies.max() + 20, 100)
y_fit = fit_line(x_fit)

# 计算直线系数的误差
fit_errors = np.sqrt(np.diag(cov_matrix))

# 绘图
plt.figure(figsize=(10, 6))
plt.errorbar(frequencies, inverse_diameters, xerr=frequency_errors, yerr=inverse_diameter_errors,
             fmt='o', capsize=5, label='Data Points')
plt.plot(x_fit, y_fit, label=f'Fit Line\nSlope = {fit_params[0]:.4f} ± {fit_errors[0]:.4f}\nIntercept = {fit_params[1]:.4f} ± {fit_errors[1]:.4f}')
plt.xlabel('Fundamental Frequencies of Strings (Hz)')
plt.ylabel('1/Diameter (mm⁻¹)')
plt.title('Linear Fit of Fundamental Frequencies vs. 1/Diameter')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('ia.jpg')