from datafunctions import load_data, preprocessing_data
from utils.output import printout
from multiprocessing.dummy import Pool as ThreadPool
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
import numpy as np


def preprocessing(list_args):

    name = list_args[0]
    dataset = list_args[1]

    msg = 'Pre-processing {0}.'.format(name)
    printout(message=msg, verbose=True, time=True)
    dataset_normalized, labels = preprocessing_data(dataset=dataset)
    msg = 'Finished pre-processing {0}.'.format(name)
    printout(message=msg, verbose=True, time=True)

    return dataset_normalized, labels


def imu_algorithm(dataset_directory='', algorithm='', quickrun=''):

    dataset_array, dataset_info = load_data(data_dir=dataset_directory)

    for user_info in dataset_info:

        testing_data = dataset_array[user_info.start_index:user_info.end_index]
        # create a new array without the testing sensor data
        training_data = np.delete(dataset_array, np.s_[user_info.start_index:user_info.end_index], axis=0)

        # list of name of the dataset and its dataset used for multiprocessing
        arg_list = [['testing dataset. User={0}'.format(user_info.user), testing_data],
                    ['training dataset', training_data]]

        # multiprocessing
        pool = ThreadPool()

        # run hmm_preprocessing their own threads and return the results
        # results = list
        #   results[0] = testing dataset, results[1] = training dataset
        results = pool.map(preprocessing, arg_list)
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()

        # fetch training and testing data from the objects
        test_dataset = results[0][0]
        test_labels = results[0][1]

        train_dataset = results[1][0]
        train_labels = results[1][1]

        printout(message='training data size:{0}'.format(np.shape(train_dataset)), verbose=True)
        printout(message='training label size:{0}'.format(np.shape(train_labels)), verbose=True)
        printout(message='testing data size:{0}'.format(np.shape(test_dataset)), verbose=True)
        printout(message='testing label size:{0}'.format(np.shape(test_labels)), verbose=True)

        if algorithm == 'HMM':
            hmm_algo(trainingdataset=train_dataset, traininglabels=train_labels, testingdataset=test_dataset,
                     testinglabels=test_labels, quickrun=quickrun)

        elif algorithm == 'Logistic Regression':
            logreg_algo(trainingdataset=train_dataset, traininglabels=train_labels, testingdataset=test_dataset,
                        testinglabels=test_labels, quickrun=quickrun)

        else:
            printout(message='Wrong algorithm provided.', verbose=True)

        msg = 'Finished analysing user:{0}\n\n'.format(user_info.user)
        printout(message=msg, verbose=True)

    msg = 'Finished running {0}'.format(algorithm)
    printout(message=msg, verbose=True)



















