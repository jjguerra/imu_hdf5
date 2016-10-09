from datafunctions import load_data, append_array
from utils.output import printout
from multiprocessing.dummy import Pool as ThreadPool
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
import numpy as np
import h5py
import os


def imu_algorithm(dataset_directory='', algorithm='', quickrun=''):

    dataset_array, dataset_info = load_data(data_dir=dataset_directory)

    printout(message='Starting pre-processing dataset', verbose=True, time=True)

    # creating argument list for multiprocessing where a message and the dataset of an specific user is passed
    arg_list = list()
    for user_index, user in enumerate(dataset_info):
        msg = 'dataset number={0} User={1}'.format(user_index, user.user)
        arg_list.append([msg, dataset_array[user.start_index:user.end_index]])

    # multiprocessing
    pool = ThreadPool()

    # run hmm_preprocessing their own threads and return the results
    # results = list
    #   results[0] = testing dataset, results[1] = training dataset
    processed_dataset = pool.map(preprocessing, arg_list)
    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    printout(message='Finished pre-processing dataset', verbose=True, time=True, extraspaces=2)

    for user_index, user_info in enumerate(dataset_info):

        msg = 'Analysing user:{0} activity:{1}'.format(user_info.user, user_info.activity)
        printout(message=msg, verbose=True, extraspaces=1)

        printout(message='Calculating training and testing dataset', verbose=True, time=True)
        # fetch testing data from the objects
        testing_dataset = processed_dataset[user_index][0]
        testing_labels = processed_dataset[user_index][1]

        # defining train dataset and labels array
        train_dataset = np.empty(shape=(0, 0))
        train_labels = np.empty(shape=(0, 0))

        # length of each dataset
        training_dataset_lengths = list()

        # fetch training data from the objects without :
        #   1. the testing data i.e. the data of user_index
        #   2. other dataset with the same user and activity but different repetition
        for user_index_inner, user_info_inner in enumerate(dataset_info):
            if user_info_inner.user != user_info.user or user_info_inner.activity != user_info.activity:
                # location of dataset
                dataset = processed_dataset[user_index_inner][0]
                # location of labels
                labels = processed_dataset[user_index_inner][1]
                # add them to the array
                train_dataset = append_array(o_array=train_dataset, array_to_add=dataset)
                train_labels = append_array(o_array=train_labels, array_to_add=labels)
                # get the size of the dataset because it will be passed as an parameter to the hmm
                length = np.shape(dataset)[0]
                training_dataset_lengths.append(length)
                msg = 'User:{0} Activity:{1} Length:{2}'.format(user_info_inner.user, user_info_inner.activity, length)
                printout(message=msg, verbose=True)
            else:
                msg = 'Skipping User:{0} Activity:{1}'.format(user_info_inner.user, user_info_inner.activity)
                printout(message=msg, verbose=True)

        # converting to numpy arrays
        training_dataset = np.array(train_dataset)
        training_labels = np.array(train_labels)
        training_length = np.array(training_dataset_lengths)

        h5name = 'train_dataset_' + user_info.user + '_' + user_info.activity + '.hdf5'
        h5_training_dataset = h5py.File(name=h5name, mode='w')
        h5_training_dataset.create_dataset(name='train', data=train_dataset, shape=np.shape(train_dataset), chunks=True)

        printout(message='', verbose=True)
        printout(message='training data size:{0}'.format(np.shape(training_dataset)), verbose=True)
        printout(message='training label size:{0}'.format(np.shape(training_labels)), verbose=True)
        printout(message='testing data size:{0}'.format(np.shape(testing_dataset)), verbose=True)
        printout(message='testing label size:{0}'.format(np.shape(testing_labels)), verbose=True)

        try:
            if algorithm == 'HMM':
                hmm_algo(trainingdataset=h5_training_dataset, traininglabels=training_labels, quickrun=quickrun,
                         testingdataset=testing_dataset, testinglabels=testing_labels, lengths=training_length)

            elif algorithm == 'Logistic Regression':
                logreg_algo(trainingdataset=h5_training_dataset, traininglabels=training_labels, quickrun=quickrun,
                            testingdataset=testing_dataset, testinglabels=testing_labels, )

            else:
                printout(message='Wrong algorithm provided.', verbose=True)

            # closing h5py file
            h5_training_dataset.close()

            msg = 'Finished analysing user:{0} activity:{1}'.format(user_info.user, user_info.activity)
            printout(message=msg, verbose=True, extraspaces=2)

        except:
            msg = 'Failed while running algorithm on user: {0} activity:{1}'.format(user_info.user, user_info.activity)
            printout(message=msg, verbose=True)
            h5_training_dataset.close()
            exit(1)

        else:
            msg = 'Finished running {0}'.format(algorithm)
            printout(message=msg, verbose=True)

            # removing created file
            os.remove(h5name)



















