from datafunctions import preprocessing_logistic_regression
from utils.output import printout
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(0)


def results(hmm_model='', trainingdataset='', traininglabels='', testingdataset='', testinglabels=''):

    printout(message='calculating Predictions', verbose=True)
    train_predictions = hmm_model.predict_proba(trainingdataset[:])
    test_predictions = hmm_model.predict_proba(testingdataset[:])

    printout(message='processing results for logistic regression algorithm', verbose=True)
    logreg_train_data, logreg_train_labels = preprocessing_logistic_regression(predictions=train_predictions,
                                                                               labels=traininglabels[:])
    logreg_test_data, logreg_test_labels = preprocessing_logistic_regression(predictions=test_predictions,
                                                                             labels=testinglabels[:])

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
        printout(message='starting training Hidden Markov Model.', time=True, verbose=True)
        hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='diag', n_iter=10, verbose=True)
        hmm_model.fit(X=trainingdataset[:], user=user, activity=activity, lengths=lengths)
        printout(message='finished training Hidden Markov Model.', time=True, verbose=True)

        results(hmm_model=hmm_model, trainingdataset=trainingdataset, traininglabels=traininglabels,
                testingdataset=testingdataset, testinglabels=testinglabels)

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

                        results(hmm_model=hmm_model, trainingdataset=trainingdataset, traininglabels=traininglabels,
                                testingdataset=testingdataset, testinglabels=testinglabels)

