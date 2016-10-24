from utils.output import printout
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
from model.lstm import lstm_algo
from utils.matlabfunctions import add_attributes
from utils.matlablabels import MatlabLabels
import numpy as np
import h5py
import os
from datetime import datetime


def imu_algorithm(dataset_directory='', algorithm='', quickrun='', program_path='', logger=''):

    label_object = MatlabLabels()
    excluded_labels = label_object.excluded_users

    # list all the files where the sensordata is stored
    # dataset_files = os.listdir(dataset_directory)
    dataset_files = ['processed_sensordata_merged.hdf5']

    # need to put all the files in the same h5py file
    if len(dataset_files) < 1:
        msg = 'Error no files in the directory:{0}'.format(dataset_directory)
        printout(message=msg, verbose=True)

    if len(dataset_files) > 1:

        file_name = 'merged_' + dataset_directory.split('/')[-1] + '.hdf5'
        file_path = os.path.join(dataset_directory, file_name)
        integrated_datasets_file = h5py.File(name=file_path, mode='w')

        printout(message='more than one file in the directory', verbose=True)
        msg = 'merging files into {0}'.format(file_name)
        printout(message=msg, verbose=True)
        for s_file in dataset_files:
            if '.hdf5' in s_file and 'merged' not in s_file:
                dataset_path = os.path.join(dataset_directory, s_file)
                h5_file_object = h5py.File(dataset_path, 'r')

                for key, value in h5_file_object.iteritems():
                    msg = '\tadding file:{0}'.format(key)
                    printout(message=msg, verbose=True)
                    integrated_datasets_file.create_dataset(name=key, data=value)
                    add_attributes(integrated_datasets_file[key], key)

                h5_file_object.close()

        integrated_datasets_file.close()

        printout(message='finished merging files')

    else:
        file_name = dataset_files[0]

    dataset_path = os.path.join(dataset_directory, file_name)
    h5_file_object = h5py.File(dataset_path, 'r')

    # printing a line for style and visibility
    printout(message='', verbose=True)

    for user_index, user_info in enumerate(h5_file_object.iterkeys()):

        # defining train dataset and labels array
        c_filename = 'training_testing_file_' + str(user_info) + '_' + datetime.now().strftime('%Y%m%d%H%M%S')\
                     + '.hdf5'
        training_file_name = os.path.join(dataset_directory, c_filename)
        training_testing_dataset_object = h5py.File(training_file_name, 'w')

        user = h5_file_object[user_info].attrs['user']
        activity = h5_file_object[user_info].attrs['activity']
        n_row, n_col = np.shape(h5_file_object[user_info][:, :-1])
        training_testing_dataset_object.create_dataset(name='testing data', shape=(n_row, n_col))
        training_testing_dataset_object['testing data'][:, :] = h5_file_object[user_info].value[:, :-1]
        training_testing_dataset_object.create_dataset(name='testing labels', shape=(n_row, 1))
        training_testing_dataset_object['testing labels'][:, 0] = h5_file_object[user_info].value[:, -1]

        msg = 'Starting analysing {0}'.format(user_info)
        logger.getLogger('regular.time').info(msg)

        msg = 'Calculating training and testing dataset'
        logger.getLogger('regular.time').info(msg)
        # fetch testing data from the objects

        # length of each dataset
        training_dataset_lengths = list()

        # flag used to create a dataset for the first file
        first_file = True

        # keep track of the total number of rows
        total_inner_row = 0

        total_inner_users = len(h5_file_object) - 1
        # fetch training data from the objects without :
        #   1. the testing data i.e. the data of user_index
        #   2. other dataset with the same user and activity but different repetition
        for u_index, user_info_inner in enumerate(h5_file_object.iterkeys()):

            # get the attributes of the training example
            inner_user = h5_file_object[user_info_inner].attrs['user']
            inner_activity = h5_file_object[user_info_inner].attrs['activity']

            # shape of the current dataset
            n_inner_row, n_inner_column = h5_file_object[user_info_inner].shape

            # removing label columns
            n_inner_column -= 1

            # make sure its not the same user doing the same activity during a different time
            if (user != inner_user) or (activity != inner_activity):

                # get the size of the dataset because it will be passed as an parameter to the hmm
                total_inner_row += h5_file_object[user_info_inner].shape[0]

                if first_file:
                    training_testing_dataset_object.create_dataset(name='training data',
                                                                   shape=(total_inner_row, n_inner_column),
                                                                   maxshape=(None, n_inner_column), chunks=True)

                    training_testing_dataset_object.create_dataset(name='training labels',
                                                                   shape=(total_inner_row, 1),
                                                                   maxshape=(None, 1), chunks=True)

                    training_testing_dataset_object['training data'][:, :] = \
                        h5_file_object[user_info_inner].value[:, :-1]
                    training_testing_dataset_object['training labels'][:, 0] = \
                        h5_file_object[user_info_inner].value[:, -1]
                    first_file = False

                else:
                    # resize the dataset to accommodate the new data
                    training_testing_dataset_object['training data'].resize(total_inner_row, axis=0)
                    training_testing_dataset_object['training data'][index_start_appending:] = \
                        h5_file_object[user_info_inner].value[:, :-1]

                    training_testing_dataset_object['training labels'].resize(total_inner_row, axis=0)
                    training_testing_dataset_object['training labels'][index_start_appending:, 0] = \
                        h5_file_object[user_info_inner].value[:, -1]

                index_start_appending = total_inner_row

                training_dataset_lengths.append(n_inner_row)
                msg = 'Including {0} (user index {1} of {2})'.format(user_info_inner, u_index, total_inner_users)
                logger.getLogger('tab.regular').info(msg)

            else:
                msg = 'Skipping {0} (user index {1} of {2})'.format(user_info_inner, u_index, total_inner_users)
                logger.getLogger('tab.regular').info(msg)

        training_dataset_lengths = np.array(training_dataset_lengths)
        training_data_object = training_testing_dataset_object['training data']
        training_label_object = training_testing_dataset_object['training labels']
        testing_data_object = training_testing_dataset_object['testing data']
        testing_label_object = training_testing_dataset_object['testing labels']

        msg = 'Training data size:{0}'.format(training_data_object.shape)
        logger.getLogger('line.tab.regular').info(msg)
        msg = 'Training labels size:{0}'.format(training_label_object.shape)
        logger.getLogger('tab.regular').info(msg)
        msg = 'Testing data size:{0}'.format(testing_data_object.shape)
        logger.getLogger('tab.regular').info(msg)
        msg = 'Testing data size:{0}'.format(testing_label_object.shape)
        logger.getLogger('tab.regular.line').info(msg)

        # try:
        if algorithm == 'HMM':
            hmm_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                     quickrun=quickrun, testingdataset=testing_data_object, testinglabels=testing_label_object,
                     lengths=training_dataset_lengths,
                     user=user, activity=activity, program_path=program_path, logger=logger)

        elif algorithm == 'Logistic Regression':
            logreg_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                        quickrun=quickrun, testingdataset=testing_data_object,
                        testinglabels=testing_label_object)

        elif algorithm == 'LSTM':
            lstm_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                      testingdataset=testing_data_object, testinglabels=testing_label_object,
                      lengths=training_dataset_lengths, logger=logger)

        else:
            printout(message='Wrong algorithm provided.', verbose=True)

        # closing h5py file
        training_testing_dataset_object.close()

        msg = 'Finished analysing {0}'.format(user_info)
        logger.getLogger('tab.regular.time.line').info(msg)

        # except:
        #     msg = 'Error while analysing {0}'.format(user_info)
        #     logger.getLogger('tab.regular.time.line').error(msg)
        #
        # # removing training dataset h5py file
        os.remove(training_file_name)
