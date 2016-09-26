from utils.output import printout
from sklearn.linear_model import LogisticRegression


def logreg_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels=''):

    printout(message='Training Logistic Regression Model.', time=True, verbose=True)
    logreg_model = LogisticRegression()
    logreg_model = logreg_model.fit(trainingdataset, traininglabels)

    printout(message='Finished training Logistic Regression Model.', time=True, verbose=True)

    printout(message='calculating Predictions', verbose=True)
    training_score = logreg_model.score(trainingdataset, traininglabels)
    testing_score = logreg_model.score(testingdataset, traininglabels)

    printout(message='Final training data prediction score: {}'.format(training_score), verbose=True)
    printout(message='Final testing data prediction score: {}'.format(testing_score), verbose=True)

















