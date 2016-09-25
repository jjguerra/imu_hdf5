from datafunctions import load_data, hmm_preprocessing_data, preprocessing_logistic_regression
from utils.output import printout
import threading
from datetime import datetime
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression


class IMUThread(threading.Thread):

    def __init__(self, name, dataset):

        threading.Thread.__init__(self)
        self.name = name
        self.dataset = dataset
        self.dataset_normalized = ''
        self.labels = ''
        self.thread_done = False

    def run(self):
        printout(message='Pre-processing {0}. Data:{1}'.format(self.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                 verbose=True)
        self.dataset_normalized, self.labels = hmm_preprocessing_data(dataset=self.dataset)
        printout(message='Finished pre-processing {0}. Data:{1}'.format(self.name,
                                                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                 verbose=True)

        self.thread_done = True

    def check_thread_finished(self):
        self.thread_done


def imu_hmm(dataset_directory):

    dataset_dataframe, dataset_info = load_data(data_dir=dataset_directory)

    for user_info in dataset_info:

        testing_data = dataset_dataframe[user_info.start_index:user_info.end_index]
        training_data = dataset_dataframe.copy()

        # remove training data
        training_data.drop(training_data.index[user_info.start_index: user_info.end_index], inplace=True)

        # creating threads
        testing_thread = IMUThread(name='testing data. User={0}'.format(user_info.user), dataset=testing_data)
        training_thread = IMUThread(name='training data', dataset=training_data)

        # staring threads
        testing_thread.start()
        training_thread.start()

        # wait for the threads to finish
        testing_thread.join()
        training_thread.join()

        # fetch training and testing data from the objects
        train_dataset = training_thread.dataset_normalized
        train_labels = training_thread.labels

        test_dataset = testing_thread.dataset_normalized
        test_labels = testing_thread.labels

        printout(message='training data size: {}'.format(train_dataset.shape), verbose=True)
        printout(message='testing data size: {}'.format(test_dataset.shape), verbose=True)

        printout(message='Training Hidden Markov Model ...', verbose=True)
        hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='full', n_iter=10, verbose=True)
        hmm_model.fit(X=train_dataset)

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
        logistic_regression_model.fit(logreg_train_data, logreg_train_labels)
        train_score = logistic_regression_model.score(logreg_train_data, logreg_train_labels)
        test_score = logistic_regression_model.score(logreg_test_data, logreg_test_labels)

        printout(message='Final training data prediction score: {}'.format(train_score), verbose=True)
        printout(message='Final testing data prediction score: {}'.format(test_score), verbose=True)

















