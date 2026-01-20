import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_excel('/Users/sonmjack/Downloads/Cosyne/Dataset_10845.xlsx')

display(df)

#%%
sample1=[]
data1 =[]
sample2=[]
sample_all = []
data2 =[]
label = []
data_all = []
ID =[]
for i in df.iloc[2:32, 1]:
    if df.iloc[i + 1, 2] <= 350:
        sample1.append(df.iloc[i + 1, 2])
        data1.append(df.iloc[i + 1, 3])
        label.append('under')
        ID.append('patient')
    else:
        sample2.append(df.iloc[i + 1, 2])
        data2.append(df.iloc[i + 1, 3])
        label.append('over')
        ID.append('patient')
    data_all.append(df.iloc[i + 1, 3])
    sample_all.append(df.iloc[i + 1, 2])
#%%

df_o = pd.DataFrame()
df_o['MI(Magnesium intake)'] = df.iloc[2:, 2].reset_index(drop=True)
df_o['MI(Magnesium intake)'].astype('float64')
df_o['Bone_density'] = df.iloc[2:, 3].reset_index(drop=True)
df_o['Bone_density'].astype('float64')
df_o['patient'] = pd.Series(ID)
df_o['class'] = pd.Series(label)
plt.figure(figsize=(10, 10))
sns.violinplot(data=df_o, x="patient", y="Bone_density", hue="class", split=True, inner="quart")
plt.show()
#%%
plt.figure(figsize=(10, 10))
sns.lmplot(data=df_o, x="MI(Magnesium intake)", y="Bone_density", hue="class")
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
data = {
    "class": ["First", "First", "First", "Second", "Second", "Third", "Third", "Third"],
    "age": [38, 35, 42, 54, 27, 19, 22, 25],
    "alive": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

# 使用小提琴图比较不同组别之间的数据分布
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="class", y="age", hue="alive", split=True, inner="quart", palette="muted")

# 添加标题和标签
plt.title("Comparison of Age Distribution by Class and Survival")
plt.xlabel("Class")
plt.ylabel("Age")

# 显示图例
plt.legend(title="Survival", loc="upper right")

# 显示图形
plt.show()

#%%
df_under = pd.DataFrame()
df_under['MI(Magnesium intake) under 350'] = pd.Series(sample1)
df_under['Bone density (g/cm^2) under 350'] = pd.Series(data1)
df_above = pd.DataFrame()
df_above['MI(Magnesium intake) above 350'] = pd.Series(sample2)
df_above['Bone density (g/cm^2) above 350'] = pd.Series(data2)


#%%
plt.figure()
plt.subplot(121)
plt.hist(sample1,10)
plt.subplot(122)
plt.hist(sample2,10)
plt.show()

#%%
plt.figure()

#第一张图
plt.subplot(211)
D1 = df_under['Bone density (g/cm^2) under 350']
# 绘制直方图
sns.histplot(D1, bins=15, alpha=0.6,kde=True)
# 拟合正态分布曲线
mu, std = stats.norm.fit(D1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fit results:\n$\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(mu, std))
# 添加标签和图例
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
# 显示方差值
plt.text(0.7, 0.9, f'std = {D1.std():.4f}', transform=plt.gca().transAxes, fontsize=12, color='r')
plt.text(0.7, 0.8, f'mean = {D1.mean():.4f}', transform=plt.gca().transAxes, fontsize=12, color='r')

#第二张图
plt.subplot(212)
D2 = df_above['Bone density (g/cm^2) above 350']
sns.histplot(D2, bins=15, alpha=1,kde=True)
# 拟合正态分布曲线
mu, std = stats.norm.fit(D2)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fit results:\n$\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(mu, std))
# 添加标签和图例
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
# 显示方差值
plt.text(0.7, 0.9, f'std = {D2.std():.4f}', transform=plt.gca().transAxes, fontsize=12, color='r')
plt.text(0.7, 0.8, f'mean = {D2.mean():.4f}', transform=plt.gca().transAxes, fontsize=12, color='r')
plt.show()

#%%
plt.figure(figsize=(10, 10))
D1 = df_under['Bone density (g/cm^2) under 350']
# 绘制直方图
sns.histplot(D1, bins=15, alpha=0.6,kde=True,label="Bone density under 350",color='skyblue')

D2 = df_above['Bone density (g/cm^2) above 350']
ax=sns.histplot(D2, bins=15, alpha=1,kde=True,label="Bone density above 350",color='coral')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()
#%%
fig = plt.figure(figsize=(10, 10))
ax = sns.kdeplot(data=df_under['Bone density (g/cm^2) under 350'],bw_adjust=0.5,label="Bone density under 350",shade=True)
sns.kdeplot(data=df_above['Bone density (g/cm^2) above 350'], bw_adjust=0.5,label="Bone density above 350",shade=True)

D1 = df_under['Bone density (g/cm^2) under 350']
# 绘制直方图
sns.histplot(D1, bins=15, alpha=0.6,kde=True,label="Bone density under 350")

D2 = df_above['Bone density (g/cm^2) above 350']
sns.histplot(D2, bins=15, alpha=1,kde=True,label="Bone density above 350")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()
print(stats.ttest_ind(D1,D2))
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 创建一个示例DataFrame

# 从DataFrame中提取要分析的列
column_data = df_under['MI(Magnesium intake) under 350']

# 绘制直方图
plt.hist(column_data, bins=10,  alpha=0.6, color='b')

# 拟合正态分布曲线
mu, std = stats.norm.fit(column_data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Fit results:\n$\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(mu, std))

# 添加标签和图例
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

# 显示方差值
plt.text(0.7, 0.9, f'Variance = {std:.2f}', transform=plt.gca().transAxes, fontsize=12, color='r')

# 显示图形
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
data = {
    "bill_length_mm": sample_all,
    "bill_depth_mm": data_all,
    "species": label
}

df = pd.DataFrame(data)

# 使用 lmplot 绘制线性回归拟合线和散点图
plt.figure(figsize=(8, 6))
sns.lmplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", height=6)

# 添加标题
plt.title("Relationship between Bill Length and Bill Depth by Species")

# 显示图形
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
data = {
    "bill_length_mm": sample_all,
    "bill_depth_mm": data_all,
    "species": label
}

df = pd.DataFrame(data)

# 使用 regplot 绘制散点图，并标出回归线的参数
plt.figure(figsize=(8, 6))

# 添加标题
plt.title("Relationship between Bill Length and Bill Depth by Species")

# 标出回归线的参数
g = sns.lmplot(data=df, y="bill_length_mm", x="bill_depth_mm", hue="species")


slope1, intercept1 = g.ax.lines[0].get_data()
slope2, intercept2 = g.ax.lines[1].get_data()
x = g.ax.get_xlim()
y = g.ax.get_ylim()

g.ax.text(x[0] , y[1] - 200, f'y = {round(slope1[1], 2)}x + {round(intercept1[1], 2)}', fontsize=12)
g.ax.text(x[0]+0.1 , y[1] - 50, f'y = {round(slope2[1], 2)}x + {round(intercept2[1], 2)}', fontsize=12)



# 显示图形
plt.show()

#%%
sample11=[]
data11 =[]
sample12=[]
data12 =[]
sample13=[]
data13 =[]
label_3 = []
ID =[]

for i in df.iloc[2:32, 1]:
    if df.iloc[i + 1, 2] <= 335:
        sample11.append(df.iloc[i + 1, 2])
        data11.append(df.iloc[i + 1, 3])
        label_3.append('MI(Magnesium intake) under 330')
        ID.append('patient')
    elif df.iloc[i + 1, 2] <= 400:
        sample12.append(df.iloc[i + 1, 2])
        data12.append(df.iloc[i + 1, 3])
        label_3.append('MI(Magnesium intake) between 330 and 400')
        ID.append('patient')
    else:
        sample13.append(df.iloc[i + 1, 2])
        data13.append(df.iloc[i + 1, 3])
        label_3.append('MI(Magnesium intake) over 400')
        ID.append('patient')

data_3 = {
    "MI(Magnesium intake)": sample_all,
    "Bone density (g/cm^2)": data_all,
    "class": label_3
}

df_o3= pd.DataFrame(data_3)
df_o3['ID'] = pd.Series(ID)