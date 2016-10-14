from datafunctions import preprocessing_logistic_regression
from utils.output import printout
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime

np.random.seed(0)


def results(train_predictions='', traininglabels='', test_predictions='', testinglabels=''):

    printout(message='processing results for logistic regression algorithm', verbose=True)
    logreg_train_data, logreg_train_labels = preprocessing_logistic_regression(predictions=train_predictions,
                                                                               labels=traininglabels)
    logreg_test_data, logreg_test_labels = preprocessing_logistic_regression(predictions=test_predictions,
                                                                             labels=testinglabels)

    # mapping hmm labels to true labels
    logistic_regression_model = LogisticRegression()

    print 'training logistic regression mapper'
    logistic_regression_model.fit(logreg_train_data, logreg_train_labels)
    train_score = logistic_regression_model.score(logreg_train_data, logreg_train_labels)
    test_score = logistic_regression_model.score(logreg_test_data, logreg_test_labels)

    printout(message='final training data prediction score: {0}'.format(train_score), verbose=True)
    printout(message='final testing data prediction score: {0}'.format(test_score), verbose=True)


def hmm_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels='',
             quickrun='', lengths=0, user='', activity=''):

    if quickrun:

        root_folder = 'data'

        files_in_data = ''
        if os.path.exists(root_folder):
            files_in_data = os.listdir(root_folder)

        loaded_model = False

        # check all the files in the folder and look for the model file
        for sfile in files_in_data:
            # check if user, activity and hmm keyword are part of the file
            if (user in sfile) and (activity in sfile) and ('hmm' in sfile) and ('.npy' not in sfile):
                printout(message='\thmm model found', time=True, verbose=True)
                # calculate the whole path
                data_path = os.path.join(root_folder, sfile)
                # load the model
                hmm_model = joblib.load(data_path)
                # turn on flag so the code does not re-train the model
                loaded_model = True
                printout(message='\thmm model loaded', time=True, verbose=True)
                break

        # check if flag is on
        if not loaded_model:
            # train model
            printout(message='starting training Hidden Markov Model.', time=True, verbose=True)
            hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='diag', n_iter=10, verbose=True)
            hmm_model.fit(X=trainingdataset[:], user=user, activity=activity, lengths=lengths)
            printout(message='finished training Hidden Markov Model.', time=True, verbose=True)

            # create a name for a file based on the user, activity and the time
            filename = 'hmm_' + user + '_' + activity + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
            # calculate the whole path
            data_path = os.path.join(root_folder, filename)

            # if data folder does not exists, make it
            if not os.path.exists(root_folder):
                os.mkdir(root_folder)

                # store the model so its not needed to re-train it
            joblib.dump(hmm_model, data_path)

        printout(message='calculating Predictions', verbose=True)
        train_predictions = hmm_model.predict_proba(trainingdataset[:])
        test_predictions = hmm_model.predict_proba(testingdataset[:])

        # using the model, run algorithms
        results(train_predictions=train_predictions, traininglabels=traininglabels[:],
                test_predictions=test_predictions, testinglabels=testinglabels[:])

    else:
        n_iterations = [10, 50, 100, 1000]
        components = [5, 8, 10, 15, 20]
        tolerance = [0.01, 0.001]
        covariance_types = ['full', 'diag', 'spherical', 'tied']
        for ni in n_iterations:
            for nc in components:
                for t in tolerance:
                    for ct in covariance_types:
                        printout(message='Training HMM model', time=True, verbose=True)
                        msg = '\tmodel parameters: \n ' \
                              '\t\tnumber of states:{0}' \
                              '\t\tnumber of iterations:{1}' \
                              '\t\ttolerance:{2}' \
                              '\t\tcovariance type:{3}'.format(nc, ni, t, ct)
                        printout(message=msg, verbose=True)

                        printout(message='starting raining Hidden Markov Model.', time=True, verbose=True)
                        hmm_model = hmm.GaussianHMM(n_components=nc, covariance_type=ct, n_iter=ni, verbose=True,
                                                    tol=t)
                        hmm_model.fit(X=trainingdataset[:, :-1])
                        printout(message='finished training Hidden Markov Model.', time=True, verbose=True)

                        printout(message='calculating Predictions', verbose=True)
                        train_predictions = hmm_model.predict_proba(trainingdataset[:])
                        test_predictions = hmm_model.predict_proba(testingdataset[:])

                        # using the model, run algorithms
                        results(train_predictions=train_predictions, traininglabels=traininglabels[:],
                                test_predictions=test_predictions, testinglabels=testinglabels[:])

