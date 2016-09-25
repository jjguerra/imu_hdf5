import numpy as np
import pandas as pd
from sklearn import preprocessing


def obtain_statistical_descriptors():

    n_dataset = ''

    # code starts here

    # obtain statistical descriptors
    mean_min_max = n_dataset.describe().loc[['mean', 'min', 'max']]
    # calculate statistical descriptors
    mean = mean_min_max.loc[['mean']].values
    mn = mean_min_max.loc[['min']].values
    mx = mean_min_max.loc[['max']].values
    # magnitude
    rms = np.sqrt(np.mean(np.square(n_dataset), axis=0))
    rms = rms.as_matrix().reshape(1, n_dataset.shape[1])
    # convert variance to matrix and re-shape it so it has the same shape as other statistical descriptors
    var = n_dataset.var().as_matrix().reshape(1, n_dataset.shape[1])
    tmp_df = pd.DataFrame(np.c_[mean, mn, mx, var, rms])


def normalizing_data():

    statistical_descriptors = ''

    # code starts here
    x = statistical_descriptors.values  # returns a numpy array
    min_max_scale = preprocessing.MinMaxScaler()
    x_scaled = min_max_scale.fit_transform(x)
    n_statistical_descriptors = pd.DataFrame(x_scaled)