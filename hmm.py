import numpy as np
import os
import pandas as pd
import re


class IMU:

    def __init__(self, data_dir):
        self.python_dataset_path = data_dir  # dataset python location
        self.user_information = dict()  # key = [user], value = [[activity, [start index activity, end index activity]]]


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


def load_data(dataset_info):

    # location of the dataset
    dataset_dir = dataset_info.python_dataset_path
    # data file are store within the project dataset folder
    dataset_files = os.listdir(dataset_dir)

    # sensordata variable
    sensordata_dataframe = pd.DataFrame()

    # loop through every file
    for python_file in dataset_files:
        python_file_path = os.path.join(dataset_dir, python_file)
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
        end_index_number = sensordata_dataframe.shape[0]

        user_prop = [activity, [start_index, end_index_number]]
        if user in dataset_info.user_information:
            tmp_list = dataset_info.user_information[user]
            tmp_list.append(user_prop)
            dataset_info.user_information[user] = tmp_list
        else:
            dataset_info.user_information[user] = user_prop

        print '\tuser={0} activity={1}, start/end index={2}'.format(user, dataset_info.user_information[user][0],
                                                                    dataset_info.user_information[user][1])

        print 'data stored in dataframe'

    return sensordata_dataframe, dataset_info


def imu_hmm(dataset_directory):

    dataset_info = IMU(data_dir=dataset_directory)
    dataset_dataframe, dataset_info = load_data(dataset_info)





