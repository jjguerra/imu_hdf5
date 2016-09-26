from datafunctions import preprocessing_logistic_regression
from utils.output import printout
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression


def hmm_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels=''):

        printout(message='Training Hidden Markov Model.', time=True, verbose=True)
        hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='full', n_iter=10, verbose=True)
        hmm_model.fit(X=trainingdataset)
        printout(message='Finished training Hidden Markov Model.', time=True, verbose=True)

        printout(message='calculating Predictions', verbose=True)
        train_predictions = hmm_model.predict_proba(trainingdataset)
        test_predictions = hmm_model.predict_proba(testingdataset)

        printout(message='processing results for logistic regression algorithm', verbose=True)
        logreg_train_data, logreg_train_labels = preprocessing_logistic_regression(predictions=train_predictions,
                                                                                   labels=traininglabels)
        logreg_test_data, logreg_test_labels = preprocessing_logistic_regression(predictions=test_predictions,
                                                                                 labels=testinglabels)

        # mapping hmm labels to true labels
        logistic_regression_model = LogisticRegression()

        print 'training logistic regression mapper'
        logistic_regression_model.fit(logreg_train_data, logreg_train_labels.values.ravel())
        train_score = logistic_regression_model.score(logreg_train_data, logreg_train_labels)
        test_score = logistic_regression_model.score(logreg_test_data, logreg_test_labels)

        printout(message='Final training data prediction score: {}'.format(train_score), verbose=True)
        printout(message='Final testing data prediction score: {}'.format(test_score), verbose=True)

















