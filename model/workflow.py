from datafunctions import load_data, preprocessing_data, append_array
from utils.output import printout
from multiprocessing.dummy import Pool as ThreadPool
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
import numpy as np


def preprocessing(list_args):

    name = list_args[0]
    dataset = list_args[1]

    msg = 'Pre-processing {0}'.format(name)
    printout(message=msg, verbose=True, time=True)
    dataset_normalized, labels = preprocessing_data(dataset=dataset)
    msg = 'Finished pre-processing {0}.'.format(name)
    printout(message=msg, verbose=True, time=True)

    return dataset_normalized, labels


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

        msg = 'Analysing user:{0}'.format(user_info.user)
        printout(message=msg, verbose=True, extraspaces=1)

        printout(message='Calculating training and testing dataset', verbose=True, time=True)
        # fetch testing data from the objects
        test_dataset = processed_dataset[user_index][0]
        test_labels = processed_dataset[user_index][1]

        # defining train dataset and labels array
        train_dataset = np.empty(shape=(0, 0))
        train_labels = np.empty(shape=(0, 0))

        # length of each dataset
        training_datasets_lengths = list()

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
                training_datasets_lengths.append(length)
                msg = 'User:{0} Activity:{1} Length:{2}'.format(user_info_inner.user, user_info_inner.activity, length)
                printout(message=msg, verbose=True)
            else:
                msg = 'Skipping User:{0} Activity:{1}'.format(user_info_inner.user, user_info_inner.activity)
                printout(message=msg, verbose=True)

        # converting to numpy arrays
        train_dataset = np.array(train_dataset)
        train_labels = np.array(train_labels)
        train_lengths = np.array(training_datasets_lengths)

        printout(message='', verbose=True)
        printout(message='training data size:{0}'.format(np.shape(train_dataset)), verbose=True)
        printout(message='training label size:{0}'.format(np.shape(train_labels)), verbose=True)
        printout(message='testing data size:{0}'.format(np.shape(test_dataset)), verbose=True)
        printout(message='testing label size:{0}'.format(np.shape(test_labels)), verbose=True)

        if algorithm == 'HMM':
            hmm_algo(trainingdataset=train_dataset, traininglabels=train_labels, testingdataset=test_dataset,
                     testinglabels=test_labels, quickrun=quickrun, lengths=train_lengths)

        elif algorithm == 'Logistic Regression':
            logreg_algo(trainingdataset=train_dataset, traininglabels=train_labels, testingdataset=test_dataset,
                        testinglabels=test_labels, quickrun=quickrun)

        else:
            printout(message='Wrong algorithm provided.', verbose=True)

        msg = 'Finished analysing user:{0}'.format(user_info.user)
        printout(message=msg, verbose=True, extraspaces=2)

    msg = 'Finished running {0}'.format(algorithm)
    printout(message=msg, verbose=True)



















