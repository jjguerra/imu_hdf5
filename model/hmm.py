from datafunctions import preprocessing_logistic_regression
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime
from sklearn import metrics
from utils.matlablabels import MatlabLabels
from utils.misc import batch, process_confusion_matrix
from sklearn.metrics import confusion_matrix

np.random.seed(0)


class ResultClass:
    def __init__(self):
        self.target_names = list()
        self.log_train_predictions = ''
        self.log_test_predictions = ''
        self.logreg_train_labels = ''
        self.logreg_test_labels = ''

    def classification(self, train_predictions, traininglabels, test_predictions, testinglabels, logger=''):

        logger.getLogger('tab.regular.time').info('processing results for logistic regression algorithm')
        logreg_train_data, self.logreg_train_labels = preprocessing_logistic_regression(predictions=train_predictions,
                                                                                        labels=traininglabels)
        logreg_test_data, self.logreg_test_labels = preprocessing_logistic_regression(predictions=test_predictions,
                                                                                      labels=testinglabels)
        logger.getLogger('tab.regular.time').info('finished processing results for logistic regression algorithm')

        # mapping hmm labels to true labels
        logistic_regression_model = LogisticRegression(n_jobs=-1)

        logger.getLogger('tab.regular.time').info('starting training logistic regression mapper')
        logistic_regression_model.fit(logreg_train_data, self.logreg_train_labels)
        logger.getLogger('tab.regular.time').info('finished training logistic regression mapper')
        train_score = logistic_regression_model.score(logreg_train_data, self.logreg_train_labels)
        test_score = logistic_regression_model.score(logreg_test_data, self.logreg_test_labels)

        self.log_train_predictions = logistic_regression_model.predict(logreg_train_data)
        self.log_test_predictions = logistic_regression_model.predict(logreg_test_data)

        logger.getLogger('tab.regular').info('final training data prediction score: {0}'.format(train_score))
        logger.getLogger('tab.regular.line').info('final testing data prediction score: {0}'.format(test_score))

        # label class
        label_class = MatlabLabels()
        self.target_names = label_class.compact_list

    def show_results(self, logger):

        logger.getLogger('line.tab.regular').info('training classification report')
        logger.getLogger('tab.regular.line').info(metrics.classification_report(y_true=self.log_train_predictions,
                                                                                y_pred=self.logreg_train_labels,
                                                                                target_names=self.target_names))

        logger.getLogger('line.tab.regular').info('testing classification report')
        classification_report_string = metrics.classification_report(y_true=self.log_test_predictions,
                                                                     y_pred=self.logreg_test_labels,
                                                                     target_names=self.target_names)
        logger.getLogger('tab.regular.line').info(classification_report_string)

        confusion_matrix_results = confusion_matrix(y_true=self.logreg_test_labels, y_pred=self.log_test_predictions)

        # # normalizing results. Normalization can be interesting in case of class imbalance to have a more visual
        # # interpretation of which class is being misclassified.
        # row_sums = confusion_matrix_results.sum(axis=1)[:, np.newaxis]
        # confusion_matrix_results = confusion_matrix_results.astype('float') / row_sums

        # logger.getLogger('line.tab.regular').info('testing confusion matrix results')
        # logger.getLogger('regular.line').info(confusion_matrix_results)

        # true_positive = confusion_matrix_results.diagonal()

        number_classes = np.shape(confusion_matrix_results)[0]

        for row_index in range(0, number_classes):
            msg = 'motion={0}'.format(self.target_names[row_index])
            logger.getLogger('line.tab.regular').info(msg)
            true_positive, false_positive, true_negative, false_negative = \
                process_confusion_matrix(confusion_matrix=confusion_matrix_results, row_index=row_index)

            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            ppv = true_positive / (true_positive + false_positive)
            npv = true_negative / (true_negative + false_negative)

            msg = 'sensitivity={0}'.format(sensitivity)
            logger.getLogger('tab.regular').info(msg)
            msg = 'specificity={0}'.format(specificity)
            logger.getLogger('tab.regular').info(msg)
            msg = 'positive predicted value={0}'.format(ppv)
            logger.getLogger('tab.regular').info(msg)
            msg = 'negative predictive value={0}'.format(npv)
            logger.getLogger('tab.regular.line').info(msg)


def hmm_algo(base_object, batched_setting, logger, algorithm, kmeans, n_states, quickrun=''):

    # initialize the loaded model flag
    loaded_model = False

    if quickrun:

        files_in_data_folder = ''
        # check if the data folder exists and if it does, get all the files
        if os.path.exists(base_object.saved_model_dir):
            files_in_data_folder = os.listdir(base_object.saved_model_dir)

        if 'low' in base_object.test_activity:
            tmp = base_object.test_activity.split('_')
            activity = tmp[0] + '_l'
        elif 'high' in base_object.test_activity:
            tmp = base_object.test_activity.split('_')
            activity = tmp[0] + '_h'
        else:
            activity = base_object.test_activity

        # check all the files in the folder and look for the model file
        for sfile in files_in_data_folder:
            # check if user, activity and hmm keyword are part of the file
            if (base_object.test_user in sfile) and (activity in sfile) and \
                    ('hmm' in sfile) and ('.npy' not in sfile):
                logger.getLogger('line.tab.regular').info('hmm model found')
                logger.getLogger('tab.regular.line').info('using hmm model {0}'.format(sfile))
                # calculate the whole path
                data_path = os.path.join(base_object.saved_model_dir, sfile)
                # load the model
                hmm_model = joblib.load(data_path)
                # turn on flag so the code does not re-train the model
                loaded_model = True
                logger.getLogger('tab.regular.time').info('hmm model loaded')
                break

    # check if flag is on
    if not loaded_model:

        nc = n_states
        cov_type = 'full'
        iterations = 10
        logger.getLogger('tab.regular.time').info('defining Gaussian Hidden Markov Model.')
        logger.getLogger('tab.regular').info('\tmodel parameters')
        msg = '\t\tnumber of states:{0}'.format(nc)
        logger.getLogger('tab.regular').info(msg)
        msg = '\t\tnumber of iterations:{0}'.format(iterations)
        logger.getLogger('tab.regular').info(msg)
        msg = '\t\tcovariance type:{0}'.format(cov_type)
        logger.getLogger('tab.regular').info(msg)
        # defining models
        hmm_model = hmm.GaussianHMM(n_components=nc, covariance_type=cov_type, n_iter=iterations, verbose=True)

        if batched_setting:
            first_run = True
            total_batches, batched_lengths = batch(base_object.training_dataset_lengths, 30)

            last_batch_index = 0
            end = 0
            for index, sliced_length in enumerate(batched_lengths):

                msg = 'starting training Gaussian Hidden Markov Model on batch {1} out of {2}'. \
                    format(index, total_batches)
                logger.getLogger('tab.regular.time').info(msg)

                end += np.sum(sliced_length).astype(np.int32)
                msg = 'size of dataset: {0}'.format(base_object.training_testing_dataset_object['training data'][
                                                    last_batch_index:end].shape)
                logger.getLogger('tab.regular').debug(msg)

                if first_run:
                    hmm_model.fit(
                        X=base_object.training_testing_dataset_object['training data'][last_batch_index:end],
                        lengths=sliced_length, logger=logger,
                        kmeans_opt=kmeans)
                    first_run = False
                else:
                    # by setting init_params='', we will be able to cascaded the training
                    # results from the previous fitting runs
                    hmm_model.init_params = ''
                    hmm_model.fit(
                        X=base_object.training_testing_dataset_object['training data'][last_batch_index:end],
                        lengths=sliced_length, logger=logger, kmeans_opt=kmeans)

                last_batch_index = end

        else:
            hmm_model.fit(X=base_object.training_testing_dataset_object['training data'],logger=logger,
                          kmeans_opt=kmeans, lengths=base_object.training_dataset_lengths)

        logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')

        # create a name for a file based on the user, activity and the time
        filename = 'hmm_' + base_object.test_user + '_' + base_object.test_activity + '_' + \
                   str(datetime.now().strftime('%Y%m%d%H%M%S'))
        # calculate the whole path
        hmm_path_filename = os.path.join(base_object.saved_model_dir, filename)
        logger.getLogger('tab.regular').debug('hmm model stored as {0}'.format(filename))
        logger.getLogger('tab.regular').debug('location {0}'.format(base_object.saved_model_dir))

        # if data folder does not exists, make it
        if not os.path.exists(base_object.saved_model_dir):
            os.mkdir(base_object.saved_model_dir)

            # store the model so its not needed to re-train it
        joblib.dump(hmm_model, hmm_path_filename)

    logger.getLogger('tab.regular.time').info('calculating predictions')
    train_predictions = hmm_model.predict_proba(
        base_object.training_testing_dataset_object['training data'], lengths=base_object.training_dataset_lengths)
    test_predictions = hmm_model.predict_proba(
        base_object.training_testing_dataset_object['testing data'])

    hmm_object = ResultClass()
    # using the model, run algorithms
    hmm_object.classification(train_predictions=train_predictions,
                              traininglabels=base_object.training_testing_dataset_object['training labels'],
                              test_predictions=test_predictions,
                              testinglabels=base_object.training_testing_dataset_object['testing labels'],
                              logger=logger)

    hmm_object.show_results(logger=logger)

    # if quickrun:
    #
    #     root_folder = '/'.join(program_path.split('/')[:-1])
    #     data_dir = os.path.join(root_folder, 'data')
    #
    #     files_in_data_folder = ''
    #     if os.path.exists(data_dir):
    #         files_in_data_folder = os.listdir(data_dir)
    #
    #     loaded_model = False
    #
    #     # check all the files in the folder and look for the model file
    #     for sfile in files_in_data_folder:
    #         # check if user, activity and hmm keyword are part of the file
    #         if (user in sfile) and (activity in sfile) and ('hmm' in sfile) and ('.npy' not in sfile):
    #             logger.getLogger('line.tab.regular').info('hmm model found')
    #             logger.getLogger('tab.regular.line').info('using hmm model {0}'.format(sfile))
    #             # calculate the whole path
    #             data_path = os.path.join(data_dir, sfile)
    #             # load the model
    #             hmm_model = joblib.load(data_path)
    #             # turn on flag so the code does not re-train the model
    #             loaded_model = True
    #             logger.getLogger('tab.regular.time').info('hmm model loaded')
    #             break
    #
    #     # check if flag is on
    #     if not loaded_model:
    #         # train model
    #         if algorithm == 'GHMM':
    #             logger.getLogger('tab.regular.time').info('starting training Gaussian Hidden Markov Model.')
    #             hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='full', n_iter=10, verbose=True)
    #
    #             if batched_setting:
    #                 first_run = True
    #                 total_batches, batched_lengths = batch(lengths, 50)
    #
    #                 last_batch_index = 0
    #                 end = 0
    #                 for index, sliced_length in enumerate(batched_lengths):
    #
    #                     msg = 'starting training Gaussian Hidden Markov Model on batch {0} out of {1}'. \
    #                         format(index, total_batches)
    #                     logger.getLogger('tab.regular.time').info(msg)
    #
    #                     end += np.sum(sliced_length).astype(np.int32)
    #                     msg = 'size of dataset: {0}'.format(trainingdataset[last_batch_index:end].shape)
    #                     logger.getLogger('tab.regular').debug(msg)
    #
    #                     if first_run:
    #                         hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
    #                                       activity=activity, data_dir=data_dir, lengths=sliced_length,
    #                                       quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #                         first_run = False
    #                     else:
    #                         # by setting init_params='', we will be able to cascaded the training
    #                         # results from the previous fitting runs
    #                         hmm_model.init_params = ''
    #                         hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
    #                                       activity=activity, data_dir=data_dir, lengths=sliced_length,
    #                                       quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #
    #                     last_batch_index = end
    #             else:
    #                 msg = 'starting training Gaussian Hidden Markov Model'
    #                 logger.getLogger('tab.regular.time').info(msg)
    #
    #                 hmm_model.fit(X=trainingdataset, user=user,
    #                               activity=activity, data_dir=data_dir, lengths=lengths,
    #                               quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #         else:
    #             logger.getLogger('tab.regular.time').info('starting training GMM Hidden Markov Model.')
    #             hmm_model = hmm.GMMHMM(n_components=8, n_mix=6)
    #
    #         hmm_model.fit(X=trainingdataset, user=user, activity=activity, data_dir=data_dir, lengths=lengths,
    #                       quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #         logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')
    #
    #         # create a name for a file based on the user, activity and the time
    #         filename = 'hmm_' + user + '_' + activity + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
    #         # calculate the whole path
    #         data_path = os.path.join(data_dir, filename)
    #         logger.getLogger('tab.regular').debug('hmm model stored as {0}'.format(filename))
    #         logger.getLogger('tab.regular').debug('location {0}'.format(data_dir))
    #
    #         # if data folder does not exists, make it
    #         if not os.path.exists(root_folder):
    #             os.mkdir(root_folder)
    #
    #             # store the model so its not needed to re-train it
    #         joblib.dump(hmm_model, data_path)
    #
    #     logger.getLogger('tab.regular.time').info('calculating predictions')
    #     train_predictions = hmm_model.predict_proba(trainingdataset, lengths=lengths)
    #     test_predictions = hmm_model.predict_proba(testingdataset[:])
    #
    #     # using the model, run algorithms
    #     results(train_predictions=train_predictions, traininglabels=traininglabels[:],
    #             test_predictions=test_predictions, testinglabels=testinglabels[:], logger=logger)
    #
    # else:
    #     if algorithm == 'GMMHMM':
    #         n_iterations = [10, 50]
    #         components = [8, 10, 15, 20]
    #         mixes = [6, 9, 12]
    #         tolerance = [0.01]
    #         covariance_types = ['spherical', 'diag', 'full', 'tied']
    #         for nc in components:
    #             for nm in mixes:
    #                 for ct in covariance_types:
    #                     for t in tolerance:
    #                         for ni in n_iterations:
    #                             logger.getLogger('tab.regular.time').info('starting training GMM Hidden Markov Model.')
    #                             logger.getLogger('tab.regular').info('\tmodel parameters')
    #                             msg = '\t\tnumber of states:{0}'.format(nc)
    #                             logger.getLogger('tab.regular').info(msg)
    #                             msg = '\t\tnumber of mixtures:{0}'.format(nm)
    #                             logger.getLogger('tab.regular').info(msg)
    #                             msg = '\t\tnumber of iterations:{0}'.format(ni)
    #                             logger.getLogger('tab.regular').info(msg)
    #                             msg = '\t\ttolerance:{0}'.format(t)
    #                             logger.getLogger('tab.regular').info(msg)
    #                             msg = '\t\tcovariance type:{0}'.format(ct)
    #                             logger.getLogger('tab.regular').info(msg)
    #
    #                             try:
    #                                 logger.getLogger('tab.regular.time').info('starting training GMM-HMM.')
    #                                 hmm_model = hmm.GMMHMM(n_components=nc, n_mix=nm, covariance_type=ct, n_iter=ni, tol=t)
    #                                 hmm_model.fit(X=trainingdataset, user=user, activity=activity, data_dir='',
    #                                               lengths=lengths, quickrun=quickrun, logger=logger)
    #                                 logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')
    #
    #                                 logger.getLogger('tab.regular.time').info('calculating predictions')
    #                                 train_predictions = hmm_model.predict_proba(trainingdataset[:])
    #                                 test_predictions = hmm_model.predict_proba(testingdataset[:])
    #
    #                                 # using the model, run algorithms
    #                                 results(train_predictions=train_predictions, traininglabels=traininglabels[:],
    #                                         test_predictions=test_predictions, testinglabels=testinglabels[:],
    #                                         logger=logger)
    #                             except ValueError as error_message:
    #                                 logger.getLogger('tab.regular.time').error(error_message)
    #     else:
    #
    #         root_folder = '/'.join(program_path.split('/')[:-1])
    #         data_dir = os.path.join(root_folder, 'data')
    #
    #         n_iterations = [10]
    #         components = [8]
    #         tolerance = [0.01]
    #         covariance_types = [ 'diag']
    #         for ni in n_iterations:
    #             for ct in covariance_types:
    #                 for nc in components:
    #                     for t in tolerance:
    #                         logger.getLogger('tab.regular.time').info(
    #                             'running Gaussian Hidden Markov Model with the following model parameters:')
    #                         msg = '\t\tnumber of states:{0}'.format(nc)
    #                         logger.getLogger('tab.regular').info(msg)
    #                         msg = '\t\tnumber of iterations:{0}'.format(ni)
    #                         logger.getLogger('tab.regular').info(msg)
    #                         msg = '\t\ttolerance:{0}'.format(t)
    #                         logger.getLogger('tab.regular').info(msg)
    #                         msg = '\t\tcovariance type:{0}'.format(ct)
    #                         logger.getLogger('tab.regular').info(msg)
    #
    #                         try:
    #                             hmm_model = hmm.GaussianHMM(n_components=nc, covariance_type=ct, n_iter=ni,
    #                                                         verbose=True, tol=t)
    #
    #                             if batched_setting:
    #                                 first_run = True
    #                                 total_batches, batched_lengths = batch(lengths, 40)
    #
    #                                 last_batch_index = 0
    #                                 end = 0
    #                                 for index, sliced_length in enumerate(batched_lengths):
    #
    #                                     msg = 'starting training Gaussian Hidden Markov Model on batch {0} out of {1}'. \
    #                                         format(index, total_batches)
    #                                     logger.getLogger('tab.regular.time').info(msg)
    #
    #                                     end += np.sum(sliced_length).astype(np.int32)
    #                                     msg = 'size of dataset: {0}'.format(trainingdataset[last_batch_index:end].shape)
    #                                     logger.getLogger('tab.regular').debug(msg)
    #
    #                                     if first_run:
    #                                         hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
    #                                                       activity=activity, data_dir='', lengths=sliced_length,
    #                                                       quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #                                         first_run = False
    #                                     else:
    #                                         # by setting init_params='', we will be able to cascaded the training
    #                                         # results from the previous fitting runs
    #                                         hmm_model.init_params = ''
    #                                         hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
    #                                                       activity=activity, data_dir='', lengths=sliced_length,
    #                                                       quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #
    #                                     last_batch_index = end
    #                             else:
    #                                 msg = 'starting training Gaussian Hidden Markov Model'
    #                                 logger.getLogger('tab.regular.time').info(msg)
    #
    #                                 hmm_model.fit(X=trainingdataset, user=user,
    #                                               activity=activity, data_dir='', lengths=lengths,
    #                                               quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
    #
    #                             logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')
    #
    #                             # create a name for a file based on the user, activity and the time
    #                             filename = 'hmm_' + user + '_' + activity + '_' + str(nc) + '_' + str(ct) + '_' + \
    #                                        str(t) + '_' + str(ni) + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
    #                             # calculate the whole path
    #                             data_path = os.path.join(data_dir, filename)
    #                             logger.getLogger('tab.regular').info('hmm model stored as {0}'.format(filename))
    #                             logger.getLogger('tab.regular').info('location {0}'.format(data_dir))
    #
    #                             # if data folder does not exists, make it
    #                             if not os.path.exists(root_folder):
    #                                 os.mkdir(root_folder)
    #
    #                                 # store the model so its not needed to re-train it
    #                             joblib.dump(hmm_model, data_path)
    #
    #                             logger.getLogger('tab.regular.time').info('calculating predictions')
    #                             train_predictions = hmm_model.predict_proba(trainingdataset[:], lengths=lengths)
    #                             test_predictions = hmm_model.predict_proba(testingdataset[:])
    #
    #                             # using the model, run algorithms
    #                             results(train_predictions=train_predictions, traininglabels=traininglabels[:],
    #                                     test_predictions=test_predictions, testinglabels=testinglabels[:],
    #                                     logger=logger)
    #
    #                         except ValueError as error_message:
    #                             logger.getLogger('tab.regular.time').error(error_message)


