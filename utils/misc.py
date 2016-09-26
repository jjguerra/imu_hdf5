from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from utils.output import printout
import threading

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

def load_data(data_dir):

    # data file are store within the project dataset folder
    dataset_files = os.listdir(data_dir)

    # sensordata variable
    sensordata_array = np.empty(shape=(0, 0))

    # dataset user information
    dataset_user_information = list()

    # loop through every file
    for python_file in dataset_files:
        python_file_path = os.path.join(data_dir, python_file)
        print 'reading file: {0}'.format(python_file)
        # getting file information
        user, activity = file_information(python_file)
        start_index = np.shape(sensordata_array)[0]

        # read data from file
        raw_data = np.load(python_file_path)
        # adding the info to the sensordata list
        if start_index == 0:
            sensordata_array = raw_data.copy()
        else:
            sensordata_array = np.append(arr=sensordata_array, values=raw_data, axis=0)
        # get the max length of the added dataset
        end_index = np.shape(sensordata_array)[0]

        user_prop = UserInfo(user=user, activity=activity, start_index=start_index, end_index=end_index)

        dataset_user_information.append(user_prop)

        print '\tuser={0} activity={1}, start/end index={2}'.format(user, activity, (start_index, end_index))

        printout(message='\tdata stored in dataframe\n', verbose=True)

    return sensordata_array, dataset_user_information


def hmm_preprocessing_data(dataset):
    # sliding window properties
    window_size = 60
    step = 1
    chunks = sliding_window(dataset, window_size, step)

    statistical_descriptors = pd.DataFrame()
    label_series = pd.Series()

    for segmented_data in chunks:
        # obtain labels
        labels = segmented_data.ix[:, len(segmented_data.columns) - 1]
        # get the most common label
        label = pd.Series(data=[labels.value_counts().idxmax()])
        # separate the labels from the dataset
        n_dataset = segmented_data.drop(segmented_data.columns[len(segmented_data.columns) - 1], axis=1)

        # calculate statistical descriptors
        mean = n_dataset.mean().values
        mn = n_dataset.min().values
        mx = n_dataset.max().values
        var = n_dataset.var().values
        # magnitude
        rms = np.sqrt(np.mean(np.square(n_dataset))).values
        tmp_df = pd.DataFrame(data=[np.r_[mean, mn, mx, var, rms]])
        # add to the dataframe
        statistical_descriptors = statistical_descriptors.append(tmp_df)
        label_series = label_series.append(label)

    # standardization : transfer data to have zero mean and unit variance
    n_statistical_descriptors = (
                                    statistical_descriptors - statistical_descriptors.mean()) / statistical_descriptors.std()

    return n_statistical_descriptors, label_series


class IMUThread(threading.Thread):

    def __init__(self, name, dataset):

        threading.Thread.__init__(self)
        self.name = name
        self.dataset = dataset
        self.dataset_normalized = ''
        self.labels = ''
        self.thread_done = False

    def run(self):
        printout(message='Pre-processing {0}. Data:{1}'.format(self.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                 verbose=True)
        self.dataset_normalized, self.labels = hmm_preprocessing_data(dataset=self.dataset)
        printout(message='Finished pre-processing {0}. Data:{1}'.format(self.name,
                                                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                 verbose=True)

        self.thread_done = True

    def check_thread_finished(self):
        self.thread_done

def load_data(data_dir):

    # data file are store within the project dataset folder
    dataset_files = os.listdir(data_dir)

    # sensordata variable
    sensordata_dataframe = pd.DataFrame()

    # dataset user information
    dataset_user_information = list()

    # loop through every file
    for python_file in dataset_files:
        python_file_path = os.path.join(data_dir, python_file)
        msg = 'reading file: {0}'.format(python_file)
        printout(message=msg, verbose=True)
        # getting file information
        user, activity = file_information(python_file)
        start_index = sensordata_dataframe.shape[0]

        # read data from file
        raw_data = np.load(python_file_path)
        # convert the array to dataframe
        df_data = pd.DataFrame(raw_data)
        # append does not happen in place so its stored back in data_dataframe
        sensordata_dataframe = sensordata_dataframe.append(df_data)
        end_index = sensordata_dataframe.shape[0]

        user_prop = UserInfo(user=user, activity=activity, start_index=start_index, end_index=end_index)

        dataset_user_information.append(user_prop)

        print '\tuser={0} activity={1}, start/end index={2}'.format(user, activity, (start_index, end_index))

        printout(message='\tdata stored in dataframe\n', verbose=True)

    sensordata_dataframe.index = range(0, sensordata_dataframe.shape[0])
    sensordata_dataframe.columns = range(0, sensordata_dataframe.shape[1])

    return sensordata_dataframe, dataset_user_information


def preprocessing_data(dataset):
    # sliding window properties
    window_size = 60
    step = 1
    chunks = sliding_window(dataset, window_size, step)

    label_list = list()

    # mean, variance and label lists
    mean_list = list()
    variance_list = list()
    min_list = list()
    max_list = list()
    # root mean square
    rms_list = list()

    for segmented_data in chunks:
        # obtain labels
        labels = segmented_data.ix[:, len(segmented_data.columns) - 1]
        # get the most common label
        label_list.append(labels.value_counts().idxmax())
        # separate the labels from the dataset
        n_dataset = segmented_data.drop(segmented_data.columns[len(segmented_data.columns) - 1], axis=1).values

        # calculate statistical descriptors
        mean = np.mean(a=n_dataset, axis=0)
        var = np.var(a=n_dataset, axis=0)
        mn = np.min(a=n_dataset, axis=0)
        mx = np.max(a=n_dataset, axis=0)
        rms = np.sqrt(np.mean(np.square(n_dataset), axis=0))

        mean_list.append(mean)
        variance_list.append(var)
        min_list.append(mn)
        max_list.append(mx)
        rms_list.append(rms)

    # list converted to numpy arrays for future processing
    mean_points = np.array(mean_list)
    var_points = np.array(variance_list)
    min_points = np.array(min_list)
    max_points = np.array(max_list)
    rms_points = np.array(rms_list)

    statistical_descriptors = np.c_[mean_points, min_points, max_points, var_points, rms_points]
    # standardization : transfer data to have zero mean and unit variance
    sd_mean = np.mean(a=statistical_descriptors, axis=0)
    sd_std = np.std(a=statistical_descriptors, axis=0)
    n_statistical_descriptors = (statistical_descriptors - sd_mean) / sd_std

    return n_statistical_descriptors, label_list