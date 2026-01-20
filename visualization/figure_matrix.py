#%%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
import math
from sklearn.metrics import normalized_mutual_info_score
import scipy.io
import numpy as np

import mat73
import pickle

#%%
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/gene_list_age10.pkl', 'rb') as file:
    gene_list_10 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_fam_age10.pkl', 'rb') as file:
    dy_list_fam1 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_Nov_age10.pkl', 'rb') as file:
    dy_list_nov = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_famr2_age10.pkl', 'rb') as file:
    dy_list_famr2 = pickle.load(file)

#%%
from sklearn.metrics.pairwise import cosine_similarity
score_list_wt_famnov = []
score_list_wt_famfamr2= []
score_list_wt_novfamr2= []

score_list_ad_famnov = []
score_list_ad_famfamr2= []
score_list_ad_novfamr2= []


for i in range(len(dy_list_fam1)):
    dy_fam1 = dy_list_fam1[i]
    dy_nov = dy_list_nov[i]
    dy_nov[np.isnan(dy_nov)] = 0
    dy_famr2 = dy_list_famr2[i]

    score_fam1nov = cosine_similarity(dy_fam1.flatten().reshape(1, -1), dy_nov.flatten().reshape(1, -1))[0,0]
    score_fam1famr2 = cosine_similarity(dy_fam1.flatten().reshape(1, -1), dy_famr2.flatten().reshape(1, -1))[0,0]
    score_novfamr2 = cosine_similarity(dy_nov.flatten().reshape(1, -1), dy_famr2.flatten().reshape(1, -1))[0, 0]

    if gene_list_10[i] == 119:
        type = 'wt'
        score_list_wt_famnov.append(score_fam1nov)
        score_list_wt_famfamr2.append(score_fam1famr2)
        score_list_wt_novfamr2.append(score_novfamr2)
    else:
        type = 'ad'
        score_list_ad_famnov.append(score_fam1nov)
        score_list_ad_famfamr2.append(score_fam1famr2)
        score_list_ad_novfamr2.append(score_novfamr2)
#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(score_list_wt_famnov, score_list_wt_novfamr2)
t, p2 = stats.ttest_ind(score_list_wt_famnov, score_list_wt_famfamr2)
t, p3 = stats.ttest_ind(score_list_wt_novfamr2, score_list_wt_famfamr2)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建三个示例的1x100的NumPy向量
vector1 = np.array(score_list_wt_famnov)
vector2 = np.array(score_list_wt_novfamr2)
vector3 = np.array(score_list_wt_famfamr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df)
plt.title("Cosine similarity of connection matrix in three environments")
plt.ylabel("Values of cosine similarity")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.xticks([0, 1, 2], ['famnov', 'novfamr2', 'famfamr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.text(0.15, 0.6, f'p_value = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.35, 0.7, f'p_value = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
# plt.text(0.55, 0.6, f'p_value = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xlabel("wild type age > 6")
plt.show()
#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(score_list_ad_famnov, score_list_ad_novfamr2)
t, p2 = stats.ttest_ind(score_list_ad_famnov, score_list_ad_famfamr2)
t, p3 = stats.ttest_ind(score_list_ad_novfamr2, score_list_ad_famfamr2)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建三个示例的1x100的NumPy向量
vector1 = np.array(score_list_ad_famnov)
vector2 = np.array(score_list_ad_novfamr2)
vector3 = np.array(score_list_ad_famfamr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("Cosine similarity of connection matrix in three environments")
plt.ylabel("Values of cosine similarity")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.xticks([0, 1, 2], ['famnov', 'novfamr2', 'famfamr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.text(0.15, 0.6, f'p_value = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.35, 0.7, f'p_value = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.55, 0.6, f'p_value = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xlabel("5xFAD age > 6")
plt.show()

#%%
with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/gene_list_age2.pkl', 'rb') as file:
    gene_list_2 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_fam_age2.pkl', 'rb') as file:
    dy_list_fam1 = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_Nov_age2.pkl', 'rb') as file:
    dy_list_nov = pickle.load(file)

with open('/Users/shengyuancai/Downloads/Imperial paper/Data/Raw data/dynamic_list_famr2_age2.pkl', 'rb') as file:
    dy_list_famr2 = pickle.load(file)

#%%
from sklearn.metrics.pairwise import cosine_similarity
score_list_wt_famnov = []
score_list_wt_famfamr2= []
score_list_wt_novfamr2= []

score_list_ad_famnov = []
score_list_ad_famfamr2= []
score_list_ad_novfamr2= []


for i in range(len(dy_list_fam1)):
    dy_fam1 = dy_list_fam1[i]
    dy_nov = dy_list_nov[i]
    dy_nov[np.isnan(dy_nov)] = 0
    dy_famr2 = dy_list_famr2[i]

    score_fam1nov = cosine_similarity(dy_fam1.flatten().reshape(1, -1), dy_nov.flatten().reshape(1, -1))[0,0]
    score_fam1famr2 = cosine_similarity(dy_fam1.flatten().reshape(1, -1), dy_famr2.flatten().reshape(1, -1))[0,0]
    score_novfamr2 = cosine_similarity(dy_nov.flatten().reshape(1, -1), dy_famr2.flatten().reshape(1, -1))[0, 0]

    if gene_list_2[i] == 119:
        type = 'wt'
        score_list_wt_famnov.append(score_fam1nov)
        score_list_wt_famfamr2.append(score_fam1famr2)
        score_list_wt_novfamr2.append(score_novfamr2)
    else:
        type = 'ad'
        score_list_ad_famnov.append(score_fam1nov)
        score_list_ad_famfamr2.append(score_fam1famr2)
        score_list_ad_novfamr2.append(score_novfamr2)

#%%
import scipy.stats as stats

t, p1 = stats.ttest_ind(score_list_ad_famnov, score_list_ad_novfamr2)
t, p2 = stats.ttest_ind(score_list_ad_famnov, score_list_ad_famfamr2)
t, p3 = stats.ttest_ind(score_list_ad_novfamr2, score_list_ad_famfamr2)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建三个示例的1x100的NumPy向量
vector1 = np.array(score_list_ad_famnov)
vector2 = np.array(score_list_ad_novfamr2)
vector3 = np.array(score_list_ad_famfamr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("Cosine similarity of connection matrix in three environments")
plt.ylabel("Values of cosine similarity")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.xticks([0, 1, 2], ['famnov', 'novfamr2', 'famfamr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.text(0.15, 0.3, f'p_value = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.35, 0.1, f'p_value = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.55, 0.3, f'p_value = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xlabel("5xFAD age < 6")
plt.show()
#%%

t, p1 = stats.ttest_ind(score_list_wt_famnov, score_list_wt_novfamr2)
t, p2 = stats.ttest_ind(score_list_wt_famnov, score_list_wt_famfamr2)
t, p3 = stats.ttest_ind(score_list_wt_novfamr2, score_list_wt_famfamr2)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建三个示例的1x100的NumPy向量
vector1 = np.array(score_list_wt_famnov)
vector2 = np.array(score_list_wt_novfamr2)
vector3 = np.array(score_list_wt_famfamr2)

# 创建一个包含所有数据的DataFrame
df = pd.DataFrame({'Vector 1': vector1, 'Vector 2': vector2, 'Vector 3': vector3})

# 绘制箱线图
#sns.set(style="whitegrid")
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=df,width=0.5, whis=1.5)
plt.title("Cosine similarity of connection matrix in three environments")
plt.ylabel("Values of cosine similarity")

# 绘制每个数据点并链接到对应点
for i in range(len(df)):
    plt.scatter([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='red', s=20,zorder=5)  # 绘制数据点
    plt.plot([0, 1, 2], [vector1[i], vector2[i], vector3[i]], color='blue', alpha=0.3)  # 连接数据点

plt.xticks([0, 1, 2], ['famnov', 'novfamr2', 'famfamr2'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.text(0.15, 0.7, f'p_value = {p1:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.35, 0.8, f'p_value = {p2:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.55, 0.7, f'p_value = {p3:.4f}', transform=plt.gca().transAxes, fontsize=12)
plt.xlabel("Wild type age < 6")
plt.show()
#%%
# from sklearn.metrics.pairwise import cosine_similarity
#
# score_list_wt = []
# score_list_ad = []
#
# for i in range(len(dy_list_fam1)):
#     dy_fam1 = dy_list_fam1[i]
#     dy_nov = dy_list_nov[i]
#     dy_nov[np.isnan(dy_nov)] = 0
#     dy_famr2 = dy_list_famr2[i]
#
#     score_fam1nov = (dy_fam1 == dy_nov).sum() / dy_nov.size
#     score_fam1famr2 = (dy_fam1 == dy_famr2).sum() / dy_nov.size
#     sco1_list.append(score_fam1nov)
#     sco2_list.append(score_fam1famr2)
#
#     sco1 = np.corrcoef(dy_fam1.flatten(), dy_nov.flatten())[0, 1]
#     sco2 = np.corrcoef(dy_fam1.flatten(), dy_famr2.flatten())[0, 1]
#     sco11_list.append(sco1)
#     sco22_list.append(sco2)
#
#     sco11 = cosine_similarity(dy_fam1.flatten().reshape(1, -1), dy_nov.flatten().reshape(1, -1))[0, 0]
#     sco22 = cosine_similarity(dy_fam1.flatten().reshape(1, -1), dy_famr2.flatten().reshape(1, -1))[0, 0]
#     sco111_list.append(sco11)
#     sco222_list.append(sco22)