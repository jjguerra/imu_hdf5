import numpy as np
import os
import pandas as pd
import re
from utils.output import printout


class UserInfo:

    def __init__(self, user='', activity='', start_index='', end_index=''):
        self.user = user
        self.activity = activity
        self.start_index = start_index
        self.end_index = end_index


def sliding_window(sequence, window_size, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable."""

    # Verify the inputs
    if not isinstance(type(window_size), type(0)) and isinstance(type(step), type(0)):
        raise Exception("**ERROR** type(window_size) and type(step) must be int.")
    if step > window_size:
        raise Exception("**ERROR** step must not be larger than window_size.")
    if window_size > len(sequence):
        raise Exception("**ERROR** window_size must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    number_of_chunks = ((len(sequence) - window_size) / step) + 1

    # Do the work
    for i in range(0, number_of_chunks * step, step):
        yield sequence[i: i + window_size]


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


def file_information(python_file):

    if 'paretic' in python_file:
        expression_pattern = \
            r'(^[A-Z]+[0-9]+)_[nonparetic|paretic]+_[active|nonactive]+_([a-z]+_high|[a-z]+_low|[a-z]+).*\.mat$'
    else:
        expression_pattern = r'(^[A-Z]+[0-9]+)_pilot_OT_[l|r]_([a-z]+_high|[a-z]+_low|[a-z]+).*\.npy$'

    # string_information = extracted useful information of the string
    # i.e.
    #    string = 'N537_pilot_OT_l_book_high.mvnx.mat'
    #    string_information.group(0) = 'N537_pilot_OT_l_book_high.mvnx.mat'
    #    user: string_information.group(1) = 'N537'
    #    activity: string_information.group(2) = 'book_high'
    string_information = re.match(pattern=expression_pattern, string=python_file)
    return string_information.group(1), string_information.group(2)


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
        print 'reading file: {0}'.format(python_file)
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


def preprocessing_logistic_regression(predictions, labels):

    dataset = pd.DataFrame(data=predictions)
    dataset['labels'] = labels
    window_size = 150
    step = 30
    chunks = sliding_window(sequence=dataset, window_size=window_size, step=step)

    new_dataset = pd.DataFrame()
    new_labels = pd.DataFrame()

    for segmented_data in chunks:
        # obtain labels
        labels = segmented_data['labels']
        # get the most common label
        label = pd.Series(data=[labels.value_counts().idxmax()])
        # separate the labels from the dataset
        n_dataset = segmented_data.drop(segmented_data.columns[len(segmented_data.columns) - 1], axis=1)
        average_of_probabilities = n_dataset.mean()

        # add values and labels to dataframes
        new_dataset = new_dataset.append(average_of_probabilities, ignore_index=True)
        new_labels = new_labels.append(label, ignore_index=True)

    return new_dataset, new_labels