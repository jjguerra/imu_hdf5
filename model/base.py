import datetime
import os
import h5py
import numpy as np


class Base:
    def __init__(self, input_path, filename, user, activity, dataset):
        self.input_path = input_path
        self.input_filename = filename
        self.test_user = user
        self.test_activity = activity
        self.h5py_file = ''
        self.first_file = True
        self.last_index = 0
        self.training_dataset_lengths = list()
        self.training_testing_filename = ''
        self.training_path_filename = ''
        self.training_testing_dataset_object = ''
        self.saved_model_dir = ''

        # location where all the models will be saved
        root_folder = '/'.join(self.input_path.split('/')[:-1])
        self.saved_model_dir = os.path.join(root_folder, 'data')

        # defining train dataset and labels file
        self.training_testing_filename = 'training_testing_file_' + self.input_filename + '_' + \
                                         datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.hdf5'

        # temporary file to store training and testing data information
        self.training_path_filename = os.path.join(input_path, self.training_testing_filename)
        self.training_testing_dataset_object = h5py.File(self.training_path_filename, 'w')

        n_row, n_col = np.shape(dataset.value[:, :-1])
        self.training_testing_dataset_object.create_dataset(name='testing data', shape=(n_row, n_col))
        self.training_testing_dataset_object['testing data'][:, :] = dataset.value[:, :-1]
        self.training_testing_dataset_object.create_dataset(name='testing labels', shape=(n_row, 1))
        self.training_testing_dataset_object['testing labels'][:, 0] = dataset.value[:, -1]

    def add_dataset(self, dataset):

        # shape of the inner dataset
        n_inner_row, n_inner_column = dataset.shape

        # removing label columns
        n_inner_column -= 1

        # get the size of the new dataset
        total_rows = self.last_index + n_inner_row

        if self.first_file:
            self.training_testing_dataset_object.create_dataset(name='training data',
                                                                shape=(n_inner_row, n_inner_column),
                                                                maxshape=(None, n_inner_column), chunks=True)

            self.training_testing_dataset_object.create_dataset(name='training labels',
                                                                shape=(n_inner_row, 1),
                                                                maxshape=(None, 1), chunks=True)

            self.training_testing_dataset_object['training data'][:, :] = dataset.value[:, :-1]
            self.training_testing_dataset_object['training labels'][:, 0] = dataset.value[:, -1]

            self.first_file = False

        else:

            # resize the dataset to accommodate the new data
            self.training_testing_dataset_object['training data'].resize(total_rows, axis=0)
            # add new data
            self.training_testing_dataset_object['training data'][
                self.last_index:] = dataset.value[:, :-1]

            # resize the dataset to accommodate the new labels
            self.training_testing_dataset_object['training labels'].resize(total_rows, axis=0)
            # add new labels
            self.training_testing_dataset_object['training labels'][
                self.last_index:, 0] = dataset.value[:, -1]

        # increase the dataset size
        self.last_index = total_rows
        # add dataset size to a list of lengths
        self.training_dataset_lengths.append(n_inner_row)

    def get_shape(self, dataset_type):

        if dataset_type == 'Testing':
            return self.training_testing_dataset_object['testing data'].shape, \
                   self.training_testing_dataset_object['testing labels'].shape
        else:
            return self.training_testing_dataset_object['training data'].shape, \
                   self.training_testing_dataset_object['training labels'].shape

    def close_and_delete(self):
        self.training_testing_dataset_object.close()
        os.remove(self.training_path_filename)

