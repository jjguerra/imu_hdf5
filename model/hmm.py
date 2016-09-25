from datafunctions import load_data, hmm_preprocessing_data, preprocessing_logistic_regression
from utils.output import printout
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np


def hmm_preprocessing(list_args):

    name = list_args[0]
    dataset = list_args[1]

    msg = 'Pre-processing {0}.'.format(name)
    printout(message=msg, verbose=True, time=True)
    dataset_normalized, labels = hmm_preprocessing_data(dataset=dataset)
    msg = 'Finished pre-processing {0}.'.format(name)
    printout(message=msg, verbose=True, time=True)

    return dataset_normalized, labels


def imu_hmm(dataset_directory):

    dataset_dataframe, dataset_info = load_data(data_dir=dataset_directory)

    for user_info in dataset_info:

        testing_data = dataset_dataframe[user_info.start_index:user_info.end_index]
        training_data = dataset_dataframe.copy()

        # remove testing data from the training dataframe
        training_data.drop(training_data.index[user_info.start_index: user_info.end_index], inplace=True)

        arg_list = [['testing dataset. User={0}'.format(user_info.user), testing_data],
                    ['training dataset', training_data]]

        # multiprocessing
        pool = ThreadPool()

        # run hmm_preprocessing their own threads and return the results
        # results = list
        #   results[0] = testing dataset, results[1] = training dataset
        results = pool.map(hmm_preprocessing, arg_list)
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

        printout(message='Training Hidden Markov Model.', time=True, verbose=True)
        hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='full', n_iter=10, verbose=True)
        hmm_model.fit(X=train_dataset)
        printout(message='Finished training Hidden Markov Model.', time=True, verbose=True)

        printout(message='calculating Predictions', verbose=True)
        train_predictions = hmm_model.predict_proba(train_dataset)
        test_predictions = hmm_model.predict_proba(test_dataset)

        printout(message='processing results for logistic regression algorithm', verbose=True)
        logreg_train_data, logreg_train_labels = preprocessing_logistic_regression(predictions=train_predictions,
                                                                                   labels=train_labels)
        logreg_test_data, logreg_test_labels = preprocessing_logistic_regression(predictions=test_predictions,
                                                                                 labels=test_labels)

        # mapping hmm labels to true labels
        logistic_regression_model = LogisticRegression()

        print 'training logistic regression mapper'
        logistic_regression_model.fit(logreg_train_data, logreg_train_labels.values.ravel())
        train_score = logistic_regression_model.score(logreg_train_data, logreg_train_labels)
        test_score = logistic_regression_model.score(logreg_test_data, logreg_test_labels)

        printout(message='Final training data prediction score: {}'.format(train_score), verbose=True)
        printout(message='Final testing data prediction score: {}'.format(test_score), verbose=True)

















