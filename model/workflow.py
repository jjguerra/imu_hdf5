from datafunctions import append_array
from utils.output import printout
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
from utils.matlabfunctions import add_attributes
import numpy as np
import h5py
import os


def imu_algorithm(dataset_directory='', algorithm='', quickrun=''):

    # list all the files where the sensordata is stored
    dataset_files = os.listdir(dataset_directory)

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

        user = h5_file_object[user_info].attrs['user']
        activity = h5_file_object[user_info].attrs['activity']
        testing_dataset_object = h5_file_object[user_info]

        msg = 'analysing user:{0} activity:{1}'.format(user, activity)
        printout(message=msg, verbose=True)

        printout(message='calculating training and testing dataset', verbose=True, time=True)
        # fetch testing data from the objects

        # length of each dataset
        training_dataset_lengths = list()

        # # temporary training numpy array
        # tmp_training_array = np.empty(shape=(0, 0))

        # flag used to create a dataset for the first file
        first_file = True

        # keep track of the total number of rows
        total_inner_row = 0

        # defining train dataset and labels array
        training_file_name = os.path.join(dataset_directory,
                                          'training_file_' + str(user_info) + '.hdf5')
        training_dataset_object = h5py.File(training_file_name, 'w')

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

            # make sure its not the same user doing the same activity during a different time
            if user != inner_user or activity != inner_activity:

                # get the size of the dataset because it will be passed as an parameter to the hmm
                total_inner_row += h5_file_object[user_info_inner].shape[0]

                if first_file:
                    training_dataset_object.create_dataset(name='training dataset',
                                                           shape=(total_inner_row, n_inner_column),
                                                           maxshape=(None, n_inner_column),
                                                           chunks=True)
                    training_dataset_object['training dataset'][:, :] = h5_file_object[user_info_inner].value
                    first_file = False

                else:
                    # resize the dataset to accommodate the new data
                    training_dataset_object['training dataset'].resize(total_inner_row, axis=0)
                    training_dataset_object['training dataset'][index_start_appending:] = \
                        h5_file_object[user_info_inner].value

                index_start_appending = total_inner_row

                # appending dataset to temporary array
                # tmp_training_array = append_array(tmp_training_array, h5_file_object[user_info_inner].value)
                training_dataset_lengths.append(n_inner_row)
                msg = '\tincluding user:{0} activity:{1} length:{2} (user index {3} of {4})'.format(
                    inner_user, inner_activity, n_inner_row, u_index, total_inner_users)
                printout(message=msg, verbose=True)

            else:
                msg = '\tskipping user:{0} activity:{1} length:{2} (user index {3} of {4})'.format(
                    inner_user, inner_activity, n_inner_row, u_index, total_inner_users)
                printout(message=msg, verbose=True)

        training_dataset_lengths = np.array(training_dataset_lengths)
        training_dset_object = training_dataset_object['training dataset']

        # # defining train dataset and labels array
        # training_file_name = os.path.join(dataset_directory, 'training_file_' + str(user_info)[1:] + '.hdf5')
        # training_dataset_object = h5py.File(training_file_name, 'w')
        # training_dataset_object.create_dataset(name='training dataset', data=tmp_training_array)

        printout(message='', verbose=True)
        printout(message='training data size:{0}'.format(training_dset_object.shape),
                 verbose=True)
        printout(message='testing data size:{0}'.format(testing_dataset_object.shape), verbose=True)

        try:
            if algorithm == 'HMM':
                hmm_algo(trainingdataset=training_dset_object, quickrun=quickrun,
                         testingdataset=testing_dataset_object, lengths=training_dataset_lengths)

            elif algorithm == 'Logistic Regression':
                logreg_algo(trainingdataset=training_dset_object, quickrun=quickrun,
                            testingdataset=testing_dataset_object)

            else:
                printout(message='Wrong algorithm provided.', verbose=True)

            # closing h5py file
            training_dataset_object.close()

            msg = 'finished analysing user:{0} activity:{1}'.format(user, activity)
            printout(message=msg, verbose=True, extraspaces=1)

            # removing training dataset h5py file
            os.remove(training_file_name)

        except:
            msg = 'Failed while running algorithm on user:{0} activity:{1}'.format(user, activity)
            printout(message=msg, verbose=True)
            training_dataset_object.close()
            exit(1)
