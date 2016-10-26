from utils.output import printout
from sklearn.linear_model import LogisticRegression


def results(logreg_model='', trainingdataset='', traininglabels='', testingdataset='',
            testinglabels='', logger=''):

    logger.getLogger('tab.regular.time').info('starting training Logistic Regression Model.')

    logger.getLogger('tab.regular.time').info('calculating Predictions')
    training_score = logreg_model.score(trainingdataset[:], traininglabels[:].ravel())
    testing_score = logreg_model.score(testingdataset[:], testinglabels[:].ravel())

    logger.getLogger('tab.regular').info('final training data prediction score: {0}'.format(training_score))
    logger.getLogger('tab.regular.line').info('final testing data prediction score: {0}'.format(testing_score))


def logreg_algo(trainingdataset='', traininglabels='', testingdataset='', testinglabels='', logger='', quickrun=True):

    if quickrun:

        logger.getLogger('tab.regular.time').info('starting training Logistic Regression Model.')
        logreg_model = LogisticRegression(n_jobs=-1, verbose=True)
        logreg_model = logreg_model.fit(trainingdataset, traininglabels[:].ravel())

        results(logreg_model=logreg_model, trainingdataset=trainingdataset, traininglabels=traininglabels,
                testingdataset=testingdataset, testinglabels=testinglabels, logger=logger)

    else:

        for t in [0.0001, 0.00001, 0.000001, 0.0000001]:
            for c in [10, 100]:
                logger.getLogger('tab.regular.time').info('starting training Logistic Regression Model.')
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

















