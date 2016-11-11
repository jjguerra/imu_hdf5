import datetime
import os
import h5py
import numpy as np


class Base:
    def __init__(self, input_path, filename, user, activity, type_activity, dataset):
        self.input_path = input_path
        self.data_dir = ''
        self.filename = filename
        self.test_user = user
        self.test_activity = activity
        self.test_activity_type = type_activity
        self.vertical_h5py = ''
        self.horizontal_h5py = ''
        self.last_index = {'vertical': 0, 'horizontal': 0}
        self.first_file = {'vertical': True, 'horizontal': True}
        self.training_data_name = {'vertical': 'training data vertical', 'horizontal': 'training data horizontal'}
        self.training_label_name = {'vertical': 'training labels vertical', 'horizontal': 'training labels horizontal'}
        self.training_dataset_lengths = {'vertical': list(), 'horizontal': list()}
        self.training_testing_filename = ''
        self.training_file_name_path = ''
        self.training_testing_dataset_object = ''

        # location where all the models will be saved
        root_folder = '/'.join(self.input_path.split('/')[:-1])
        self.data_dir = os.path.join(root_folder, 'data')

        # defining train dataset and labels file
        self.training_testing_filename = 'training_testing_file_' + self.filename + '_' + \
                                         datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.hdf5'

        # temporary file to store training and testing data information
        self.training_file_name_path = os.path.join(input_path, self.training_testing_filename)
        self.training_testing_dataset_object = h5py.File(self.training_file_name_path, 'w')

        n_row, n_col = np.shape(dataset.value[:, :-1])
        self.training_testing_dataset_object.create_dataset(name='testing data', shape=(n_row, n_col))
        self.training_testing_dataset_object['testing data'][:, :] = dataset.value[:, :-1]
        self.training_testing_dataset_object.create_dataset(name='testing labels', shape=(n_row, 1))
        self.training_testing_dataset_object['testing labels'][:, 0] = dataset.value[:, -1]

    def add_dataset(self, dataset, activity_type):

        # shape of the inner dataset
        n_inner_row, n_inner_column = dataset.shape

        # removing label columns
        n_inner_column -= 1

        # get the size of the new dataset
        total_rows = self.last_index[activity_type] + n_inner_row

        if self.first_file[activity_type]:
            self.training_testing_dataset_object.create_dataset(name=self.training_data_name[activity_type],
                                                                shape=(n_inner_row, n_inner_column),
                                                                maxshape=(None, n_inner_column), chunks=True)

            self.training_testing_dataset_object.create_dataset(name=self.training_label_name[activity_type],
                                                                shape=(n_inner_row, 1),
                                                                maxshape=(None, 1), chunks=True)

            self.training_testing_dataset_object[self.training_data_name[activity_type]][:, :] = dataset.value[:, :-1]
            self.training_testing_dataset_object[self.training_label_name[activity_type]][:, 0] = dataset.value[:, -1]

            self.first_file[activity_type] = False

        else:

            # resize the dataset to accommodate the new data
            self.training_testing_dataset_object[self.training_data_name[activity_type]].resize(total_rows, axis=0)
            # add new data
            self.training_testing_dataset_object[self.training_data_name[activity_type]][
                self.last_index[activity_type]:] = dataset.value[:, :-1]

            # resize the dataset to accommodate the new labels
            self.training_testing_dataset_object[self.training_label_name[activity_type]].resize(total_rows, axis=0)
            # add new labels
            self.training_testing_dataset_object[self.training_label_name[activity_type]][
                self.last_index[activity_type]:, 0] = dataset.value[:, -1]

        # increase the dataset size
        self.last_index[activity_type] = total_rows
        # add dataset size to a list of lengths
        self.training_dataset_lengths[activity_type].append(n_inner_row)

    def get_shape(self, activity):

        if activity == 'test':
            return self.training_testing_dataset_object['testing data'].shape, \
                   self.training_testing_dataset_object['testing labels'].shape
        else:
            return self.training_testing_dataset_object[self.training_data_name[activity]].shape, \
                   self.training_testing_dataset_object[self.training_label_name[activity]].shape

    def close_and_delete(self):
        self.training_testing_dataset_object.close()
        os.remove(self.training_file_name_path)

