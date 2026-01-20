import pickle
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
from sklearn.manifold import MDS
import math
import scipy.io
import numpy as np
from scipy import stats

#%%
if Type == 'Young':
    with open('/Users/sonmjack/Downloads/age2 result_fam1r2/fam1r2_signal_corr_WT', 'wb') as file:
        pickle.dump(All_signal_corr_WT, file)
    plt.savefig('/Users/sonmjack/Downloads/age2 result_fam1r2/signal_corr/' + 'Whole signal corr' + '.jpg')
    plt.close()
elif Type == 'Old':
    with open('/Users/sonmjack/Downloads/age10 result_fam1r2/fam1r2_signal_corr_WT', 'wb') as file:
        pickle.dump(All_signal_corr_WT, file)