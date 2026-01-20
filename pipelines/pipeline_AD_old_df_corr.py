#%%
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
from sklearn.manifold import MDS
import math
import scipy.io
import numpy as np
import pickle
import pandas as pd
#%%
be_data = scipy.io.loadmat('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_trackdata.mat')
import h5py
mat_trigger = np.load('/Users/sonmjack/Downloads/simon_paper/shengyuan_trigger_fam1.npy')
type_array = h5py.File('/Users/sonmjack/Downloads/simon_paper/data_fam1novfam1_timeseries.mat')

gene = type_array['genotype'][:, :].T
# mat_label = np.zeros((gene.shape[0],4))
# be_phi_sum = be_data['nov_phi']
be_phi_sum_fam1 = be_data['fam1_phi']
be_phi_sum_nov = be_data['nov_phi']
be_phi_sum_fam1r2 = be_data['fam1r2_phi']

#%%
be_phi_list_young_fam1 = []
be_phi_list_young_fam1r2 = []
be_phi_list_young_nov = []

gene_list_young = []
for i in range(10,46,2):#0, len(mat_trigger), 2

    if i == len(mat_trigger):
        break
    if i == 18:
        pass
    else:
        be_phi_list_young_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
        be_phi_list_young_nov.append(be_phi_sum_nov[int(i / 2), 0])
        be_phi_list_young_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

        gene_list_young.append(mat_trigger[i, 1])

be_phi_list_old_fam1 = []
be_phi_list_old_nov = []
be_phi_list_old_fam1r2 = []

gene_list_old = []
for i in range(0,10,2):#0, len(mat_trigger), 2
        be_phi_list_old_fam1.append(be_phi_sum_fam1[int(i / 2), 0])
        be_phi_list_old_nov.append(be_phi_sum_nov[int(i / 2), 0])
        be_phi_list_old_fam1r2.append(be_phi_sum_fam1r2[int(i / 2), 0])

        gene_list_old.append(mat_trigger[i, 1])

del be_data, be_phi_sum_fam1,be_phi_sum_nov,be_phi_sum_fam1r2

#%%

import pandas as pd
Type = 'Old'
if Type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        All_df_f1 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_df_f.pkl', 'rb') as file:
        All_df_f2 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
        All_df_f3 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve1 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve2 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve3 = pickle.load(file)
    gene_list = gene_list_young
    #fig, axes = plt.subplots(6, 1, figsize=(8*1, 36))
elif Type == 'Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_df_f.pkl', 'rb') as file:
        All_df_f1 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_nov/nov_S_df_f.pkl', 'rb') as file:
        All_df_f2 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_S_df_f.pkl', 'rb') as file:
        All_df_f3 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1/fam1_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve1 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_nov/nov_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve2 = pickle.load(file)
    with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_S_tuning curve.pkl', 'rb') as file:
        All_tuning_curve3 = pickle.load(file)
    gene_list = gene_list_old
    #fig, axes = plt.subplots(4, 3, figsize=(8*3, 24))
#%%
def prepare_data_corr(data,group_name):
    df = pd.DataFrame(data, columns=[group_name])
    return df

from scipy.stats import linregress


def plot_with_diagonal_and_fit_line(df, x, y, title):
    # 绘制jointplot
    g = sns.jointplot(data=df, x=x, y=y, kind='scatter', height=5)
    plt.suptitle(title, y=0.95)

    # 获取轴对象
    ax = g.ax_joint

    # 添加45度对角线
    lims = np.array([ax.get_xlim(), ax.get_ylim()])  # 获取当前轴的限制以确定对角线的长度
    max_lim = max(lims[0][1], lims[1][1])  # 选择x和y轴上限中的最大值来确保对角线覆盖整个图形
    ax.plot([0, max_lim], [0, max_lim], ':k', linewidth=1)  # 绘制45度对角线

    # 计算并绘制相关系数直线
    slope, intercept, r_value, _, _ = linregress(df[x], df[y])
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '-r', label=f'Slope: {slope:.2f}\n$r^2$: {r_value ** 2:.2f}')

    # 添加图例
    ax.legend()
#%%
it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_df_f1)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        A1 = All_df_f1[index]
        A2 = All_df_f2[index]
        A3 = All_df_f3[index]

        # A1 = All_tuning_curve1[index]
        # A2 = All_tuning_curve2[index]
        # A3 = All_tuning_curve3[index]

        mean1_list = []
        mean2_list = []
        mean3_list = []

        for i in range(min(A1.shape[0], A2.shape[0], A3.shape[0])):
            mean_df1 = np.mean(A1[i, :])
            mean_df2 = np.mean(A2[i, :])
            mean_df3 = np.mean(A3[i, :])
            mean1_list.append(mean_df1)
            mean2_list.append(mean_df2)
            mean3_list.append(mean_df3)

        df1 = prepare_data_corr(mean1_list, 'Activity of fam1 (df/f)')
        df2 = prepare_data_corr(mean2_list, 'Activity of Nov (df/f)')
        df3 = prepare_data_corr(mean3_list, 'Activity of fam1r2 (df/f)')

        df_merged1 = pd.concat([df1, df2], axis=1)
        df_merged2 = pd.concat([df3, df2], axis=1)
        df_merged3 = pd.concat([df1, df3], axis=1)
        # 绘制曲线图

        plt.figure(figsize=(10, 10))
        plot_with_diagonal_and_fit_line(df_merged2, 'Activity of fam1r2 (df/f)', 'Activity of Nov (df/f)',
                                        'No.' + f'{index} {type}')
        # plt.suptitle('No.' + f'{index} {type}')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if Type == 'Young':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/df corr d23/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        elif Type == 'Old':
            plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/df corr d23/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_with_diagonal_and_fit_line(df_merged1, 'Activity of fam1 (df/f)', 'Activity of Nov (df/f)',
                                        'No.' + f'{index} {type}')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if Type == 'Young':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/df corr d12/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        elif Type == 'Old':
            plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/df corr d12/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_with_diagonal_and_fit_line(df_merged3, 'Activity of fam1 (df/f)', 'Activity of fam1r2 (df/f)',
                                        'No.' + f'{index} {type}')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if Type == 'Young':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/df corr d13/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        elif Type == 'Old':
            plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/df corr d13/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        plt.close()

        it += 1

    else:
        type = 'AD'
        pass

#%%

#%%
from PIL import Image
import os
# 图像文件路径
if Type == 'Young':
    image_folder = '/Users/sonmjack/Downloads/age2 result_fam1r2/df corr d13'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

if Type == 'Old':
    image_folder = '/Users/sonmjack/Downloads/age10 result_fam1r2/df corr d13'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# 打开图像并放入列表
images = [Image.open(os.path.join(image_folder, image)) for image in image_files]

# 计算新图像的总宽度和最大高度
max_width = max(image.width for image in images)
total_height = sum(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (max_width, total_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (0,x_offset))
    x_offset += image.height

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/corr AD df_13.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1r2/signal_corr/corr AD df_13.jpg')
#%%
#%%
from PIL import Image
import os
# 图像文件路径
if Type == 'Young':
    image_folder = '/Users/sonmjack/Downloads/age2 result_fam1r2/df corr d12'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

if Type == 'Old':
    image_folder = '/Users/sonmjack/Downloads/age10 result_fam1r2/df corr d12'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# 打开图像并放入列表
images = [Image.open(os.path.join(image_folder, image)) for image in image_files]

# 计算新图像的总宽度和最大高度
max_width = max(image.width for image in images)
total_height = sum(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (max_width, total_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (0,x_offset))
    x_offset += image.height

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/corr AD df_12.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1r2/signal_corr/corr AD df_12.jpg')
#%%
from PIL import Image
import os
# 图像文件路径
if Type == 'Young':
    image_folder = '/Users/sonmjack/Downloads/age2 result_fam1r2/df corr d23'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

if Type == 'Old':
    image_folder = '/Users/sonmjack/Downloads/age10 result_fam1r2/df corr d23'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# 打开图像并放入列表
images = [Image.open(os.path.join(image_folder, image)) for image in image_files]

# 计算新图像的总宽度和最大高度
max_width = max(image.width for image in images)
total_height = sum(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (max_width, total_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (0,x_offset))
    x_offset += image.height

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/corr AD df_23.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1r2/signal_corr/corr AD df_23.jpg')


#%%


# %%
it = 0
bins = np.linspace(0, 1, num=25)
for index in range(len(All_df_f1)):
    if gene_list[index] == 116:
        type = 'AD'
        print('Finished ' + f'{index}')
        # A1 = All_df_f1[index]
        # A2 = All_df_f2[index]
        # A3 = All_df_f3[index]

        A1 = All_tuning_curve1[index]
        A2 = All_tuning_curve2[index]
        A3 = All_tuning_curve3[index]

        mean1_list = []
        mean2_list = []
        mean3_list = []

        for i in range(min(A1.shape[0], A2.shape[0], A3.shape[0])):
            mean_df1 = np.mean(A1[i, :])
            mean_df2 = np.mean(A2[i, :])
            mean_df3 = np.mean(A3[i, :])
            mean1_list.append(mean_df1)
            mean2_list.append(mean_df2)
            mean3_list.append(mean_df3)

        df1 = prepare_data_corr(mean1_list, 'Activity of fam1 (count)')
        df2 = prepare_data_corr(mean2_list, 'Activity of Nov (count)')
        df3 = prepare_data_corr(mean3_list, 'Activity of fam1r2 (count)')

        df_merged1 = pd.concat([df1, df2], axis=1)
        df_merged2 = pd.concat([df3, df2], axis=1)
        df_merged3 = pd.concat([df1, df3], axis=1)
        # 绘制曲线图

        plt.figure(figsize=(10, 10))
        plot_with_diagonal_and_fit_line(df_merged2, 'Activity of fam1r2 (count)', 'Activity of Nov (count)',
                                        'No.' + f'{index} {type}')
        # plt.suptitle('No.' + f'{index} {type}')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if Type == 'Young':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/df tuning d23/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        elif Type == 'Old':
            plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/df tuning d23/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_with_diagonal_and_fit_line(df_merged1, 'Activity of fam1 (count)', 'Activity of Nov (count)',
                                        'No.' + f'{index} {type}')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if Type == 'Young':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/df tuning d12/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        elif Type == 'Old':
            plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/df tuning d12/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        plt.close()

        plt.figure(figsize=(10, 10))
        plot_with_diagonal_and_fit_line(df_merged3, 'Activity of fam1 (count)', 'Activity of fam1r2 (count)',
                                        'No.' + f'{index} {type}')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if Type == 'Young':
            plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/df tuning d13/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        elif Type == 'Old':
            plt.savefig('/Users/sonmjack/Downloads/age10 result_fam1r2/df tuning d13/' + f'{index}_AD df corr' + '.jpg')
            plt.close()
        plt.close()

        it += 1

    else:
        type = 'AD'
        pass






#%%
from PIL import Image
import os
# 图像文件路径
if Type == 'Young':
    image_folder = '/Users/sonmjack/Downloads/age2 result_fam1r2/df tuning d13'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

if Type == 'Old':
    image_folder = '/Users/sonmjack/Downloads/age10 result_fam1r2/df tuning d13'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# 打开图像并放入列表
images = [Image.open(os.path.join(image_folder, image)) for image in image_files]

# 计算新图像的总宽度和最大高度
max_width = max(image.width for image in images)
total_height = sum(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (max_width, total_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (0,x_offset))
    x_offset += image.height

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/corr AD tuning_13.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1r2/signal_corr/corr AD tuning_13.jpg')
#%%
#%%
from PIL import Image
import os
# 图像文件路径
if Type == 'Young':
    image_folder = '/Users/sonmjack/Downloads/age2 result_fam1r2/df tuning d12'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

if Type == 'Old':
    image_folder = '/Users/sonmjack/Downloads/age10 result_fam1r2/df tuning d12'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# 打开图像并放入列表
images = [Image.open(os.path.join(image_folder, image)) for image in image_files]

# 计算新图像的总宽度和最大高度
max_width = max(image.width for image in images)
total_height = sum(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (max_width, total_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (0,x_offset))
    x_offset += image.height

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/corr AD tuning_12.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1r2/signal_corr/corr AD tuning_12.jpg')
#%%
from PIL import Image
import os
# 图像文件路径
if Type == 'Young':
    image_folder = '/Users/sonmjack/Downloads/age2 result_fam1r2/df tuning d23'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

if Type == 'Old':
    image_folder = '/Users/sonmjack/Downloads/age10 result_fam1r2/df tuning d23'
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# 打开图像并放入列表
images = [Image.open(os.path.join(image_folder, image)) for image in image_files]

# 计算新图像的总宽度和最大高度
max_width = max(image.width for image in images)
total_height = sum(image.height for image in images)

# 创建一个新的图像，背景为白色
new_image = Image.new('RGB', (max_width, total_height), 'white')

# 逐一将每个图像粘贴到新图像中
x_offset = 0
for image in images:
    new_image.paste(image, (0,x_offset))
    x_offset += image.height

# 保存新图像
if Type == 'Young':
    new_image.save('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/corr AD tuning_23.jpg')
if Type == 'Old':
    new_image.save('/Users/sonmjack/Downloads/age10 result_fam1r2/signal_corr/corr AD tuning_23.jpg')