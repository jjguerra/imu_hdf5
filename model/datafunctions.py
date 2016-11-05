import numpy as np
import os
import re
from utils.output import printout
from collections import Counter
import random
import h5py
from utils.matlabfunctions import add_attributes


class UserInfo:

    def __init__(self, user='', activity='', start_index='', end_index=''):
        self.user = user
        self.activity = activity
        self.start_index = start_index
        self.end_index = end_index


def sliding_window(sequence, window_size, step=1):
    """
    Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.
    """

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


def most_common(lst):
    m_frequent = Counter(lst).most_common(2)
    # if out of the two most frequent labels, the first one is
    # larger than the second one i.e. different, return the first one (largest frequency)
    if len(m_frequent) > 1:
        if m_frequent[0][1] != m_frequent[1][1]:
            return m_frequent[0][0]
        # if there are two equally frequent labels, randomly return one
        else:
            rv = random.randint(0, 1)
            return m_frequent[rv][0]
    else:
        return m_frequent[0][0]


def preprocessing_data(dataset_object, window_step_size, logger):

        preprocessed_dataset_object = dataset_object[0]
        postprocessed_dataset_object = dataset_object[1]

        msg = 'original data size: {0}'.format(preprocessed_dataset_object[:].shape)
        logger.getLogger('tab.tab.regular').info(msg)

        # sliding window properties
        window_size = window_step_size[0]
        step = window_step_size[1]
        chunks = sliding_window(preprocessed_dataset_object[:], window_size, step)

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
            labels = segmented_data[:, np.shape(segmented_data)[1] - 1]
            # get the most common label
            label_list.append(most_common(labels))
            # separate the labels from the dataset
            n_dataset = segmented_data[:, :np.shape(segmented_data)[1] - 1]

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

        postprocessed_dataset_object.create_dataset(name=preprocessed_dataset_object.name,
                                                    data=np.c_[n_statistical_descriptors, np.array(label_list)])

        add_attributes(postprocessed_dataset_object[preprocessed_dataset_object.name],
                       str(preprocessed_dataset_object.name)[1:])

        msg = 'resulting data size: {0}'.format(postprocessed_dataset_object[preprocessed_dataset_object.name].shape)
        logger.getLogger('tab.tab.regular').info(msg)


def file_information(python_file):

    if 'paretic' in python_file:
        expression_pattern = \
            r'(^[A-Z]+[0-9]+)_[nonparetic|paretic]+_[active|nonactive]+_([a-z]+_high|[a-z]+_low|[a-z]+).*\.mat.npy$'
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
        sensordata_array = append_array(sensordata_array, raw_data)

        # get the max length of the added dataset
        end_index = np.shape(sensordata_array)[0]

        user_prop = UserInfo(user=user, activity=activity, start_index=start_index, end_index=end_index)

        dataset_user_information.append(user_prop)

        print '\tuser={0} activity={1}, start/end index={2}, size={3}'.format(user, activity, (start_index, end_index),
                                                                              np.shape(raw_data)[0])

        printout(message='\tdata stored in dataframes\n', verbose=True)

    return sensordata_array, dataset_user_information


def preprocessing_logistic_regression(predictions, labels):

    dataset = np.append(predictions, labels, 1)
    window_size = 150
    step = 30
    chunks = sliding_window(sequence=dataset, window_size=window_size, step=step)

    new_dataset = list()
    new_labels = list()

    for segmented_data in chunks:
        # obtain labels
        segmented_labels = list(segmented_data[:, -1])
        # get the most common label
        new_labels.append(most_common(segmented_labels))
        # separate the labels from the dataset
        n_dataset = segmented_data[:, :np.shape(segmented_data)[1] - 1]
        average_of_probabilities = np.mean(n_dataset, axis=0)

        # add values and labels to dataframes
        new_dataset.append(average_of_probabilities)

    return np.array(new_dataset), np.array(new_labels)


def append_array(o_array, array_to_add):

    # check to see if original array in empty
    if o_array.size == 0:
        merged_array = array_to_add
    else:
        # append  new array it to the original
        merged_array = np.append(arr=o_array, values=array_to_add, axis=0)

    return merged_array


# getting the statistical descriptors
def featurize(file_properties, window_step_size, logger):

    # converted file
    h5_file_object = h5py.File(name=file_properties.input_path_filename, mode='r')

    # processed dataset file
    processed_file_object = h5py.File(name=file_properties.output_path_filename, mode='w')

    msg = 'Starting pre-processing dataset {0}'.format(h5_file_object.filename)
    logger.getLogger('tab.regular.time').info(msg)

    last_dataset = len(h5_file_object) - 1
    for dataset_index, dataset_key in enumerate(h5_file_object.iterkeys()):

        global_msg = 'dataset {0} out of {1} user:{2} activity:{3}'.\
            format(dataset_index, last_dataset, h5_file_object[dataset_key].attrs['user'],
                   h5_file_object[dataset_key].attrs['activity'])

        # function used to call the statistical descriptor function

        msg = 'Pre-processing {0}'.format(global_msg)
        logger.getLogger('tab.tab.regular').info(msg)
        preprocessing_data(dataset_object=[h5_file_object[dataset_key], processed_file_object],
                           window_step_size=window_step_size, logger=logger)

    msg = 'Finished pre-processing dataset {0}'.format(file_properties.input_filename)
    logger.getLogger('tab.regular.time').info(msg)

    msg = 'Dataset filename location: {0}'.format(file_properties.output_path_filename)
    logger.getLogger('tab.regular').info(msg)

    processed_file_object.close()


