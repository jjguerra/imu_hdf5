from utils.output import printout
from sklearn.linear_model import LogisticRegression


def results(logreg_model='', trainingdataset='', traininglabels='', testingdataset='',
            testinglabels=''):

    printout(message='Finished training Logistic Regression Model.', time=True, verbose=True)

    printout(message='calculating Predictions', verbose=True)
    training_score = logreg_model.score(trainingdataset[:], traininglabels[:])
    testing_score = logreg_model.score(testingdataset[:], testinglabels[:])

    printout(message='Final training data prediction score: {0}'.format(training_score), verbose=True)
    printout(message='Final testing data prediction score: {0}'.format(testing_score), verbose=True)


def logreg_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels='', quickrun=True):

    if quickrun:

        printout(message='Training Logistic Regression Model.', time=True, verbose=True)
        logreg_model = LogisticRegression()
        logreg_model = logreg_model.fit(trainingdataset[:], traininglabels[:])

        results(logreg_model=logreg_model, trainingdataset=trainingdataset, traininglabels=traininglabels,
                testingdataset=testingdataset, testinglabels=testinglabels)

    else:

        for t in [0.0001, 0.00001, 0.000001, 0.0000001]:
            for c in [10, 100]:
                printout(message='Training Logistic Regression Model', time=True, verbose=True)
                msg = '\tmodel parameters: \n ' \
                      '\t\tInverse of regularization strength:{0}' \
                      '\t\ttolerance:{1}' \
                      '\t\tmax iterations:{2}' \
                      '\t\tnumber of jobs:{3}'.format(c, t, 200, -1)
                printout(message=msg, verbose=True)
                logreg_model = LogisticRegression(C=c, tol=t, max_iter=200, n_jobs=-1, verbose=True)
                logreg_model = logreg_model.fit(trainingdataset[:], traininglabels[:])

                results(logreg_model=logreg_model, trainingdataset=trainingdataset, traininglabels=traininglabels,
                        testingdataset=testingdataset, testinglabels=testinglabels)

















