from datafunctions import preprocessing_logistic_regression
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import classification_report
from utils.matlablabels import MatlabLabels
from utils.misc import batch
import math

np.random.seed(0)


class ResultClass:
    def __init__(self):
        self.orientation = ''
        self.train_score = ''
        self.test_score = ''
        self.target_names = list()
        self.log_train_predictions = ''
        self.log_test_predictions = ''
        self.logreg_train_labels = ''
        self.logreg_test_labels = ''

    def classification(self, train_predictions, traininglabels, test_predictions, testinglabels, vertical_horizontal,
                       logger=''):

        self.orientation = vertical_horizontal

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
        self.train_score = logistic_regression_model.score(logreg_train_data, self.logreg_train_labels)
        self.test_score = logistic_regression_model.score(logreg_test_data, self.logreg_test_labels)

        self.log_train_predictions = logistic_regression_model.predict(logreg_train_data)
        self.log_test_predictions = logistic_regression_model.predict(logreg_test_data)

        logger.getLogger('tab.regular').info('final training data prediction score: {0}'.format(self.train_score))
        logger.getLogger('tab.regular.line').info('final testing data prediction score: {0}'.format(self.test_score))

        # label class
        label_class = MatlabLabels()
        self.target_names = label_class.compact_list


def show_results(hmm_result_list, logger):

    score = 0
    hmm_index = 0
    for highest_index, hmm_model in enumerate(hmm_result_list):
        if hmm_model.test_score > score:
            score = hmm_model.test_score
            hmm_index = highest_index

    msg = 'highest likelihood mode: {0} '.format(hmm_result_list[hmm_index].orientation)
    logger.getLogger('tab.regular.line').info(msg)

    logger.getLogger('line.tab.regular').info('training classification report')
    logger.getLogger('tab.regular.line').info(classification_report(hmm_result_list[hmm_index].log_train_predictions,
                                                                    hmm_result_list[hmm_index].logreg_train_labels,
                                                                    target_names=hmm_result_list[hmm_index].target_names))

    logger.getLogger('line.tab.regular').info('testing classification report')
    logger.getLogger('tab.regular.line').info(classification_report(hmm_result_list[hmm_index].log_test_predictions,
                                                                    hmm_result_list[hmm_index].logreg_test_labels,
                                                                    target_names=hmm_result_list[hmm_index].target_names))


def hmm_algo(base_object, batched_setting, logger, algorithm, kmeans, quickrun=''):

    possible_direction = ['vertical', 'horizontal']
    hmm_models = dict()
    hmm_result = list()

    for vertical_horizontal in possible_direction:

        if quickrun:

            files_in_data_folder = ''
            # check if the data folder exists and if it does, get all the files
            if os.path.exists(base_object.data_dir):
                files_in_data_folder = os.listdir(base_object.data_dir)

            # initialize the loaded model flag
            loaded_model = False

            # check all the files in the folder and look for the model file
            for sfile in files_in_data_folder:
                # check if user, activity and hmm keyword are part of the file
                if (base_object.test_user in sfile) and (base_object.test_activity in sfile) and \
                        ('hmm' in sfile) and ('.npy' not in sfile) and (vertical_horizontal in sfile):
                    logger.getLogger('line.tab.regular').info('hmm model found')
                    logger.getLogger('tab.regular.line').info('using hmm model {0}'.format(sfile))
                    # calculate the whole path
                    data_path = os.path.join(base_object.data_dir, sfile)
                    # load the model
                    hmm_models[vertical_horizontal] = joblib.load(data_path)
                    # turn on flag so the code does not re-train the model
                    loaded_model = True
                    logger.getLogger('tab.regular.time').info('hmm model loaded')
                    break

            # check if flag is on
            if not loaded_model:

                nc = 8
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
                hmm_models[vertical_horizontal] = hmm.GaussianHMM(n_components=nc, covariance_type=cov_type,
                                                                  n_iter=iterations, verbose=True)

                if batched_setting:
                    first_run = True
                    total_batches, batched_lengths = batch(base_object.training_dataset_lengths[vertical_horizontal],
                                                           50)

                    last_batch_index = 0
                    end = 0
                    for index, sliced_length in enumerate(batched_lengths):

                        msg = 'starting training {0} Gaussian Hidden Markov Model on batch {1} out of {2}'. \
                            format(vertical_horizontal, index, total_batches)
                        logger.getLogger('tab.regular.time').info(msg)

                        end += np.sum(sliced_length).astype(np.int32)
                        msg = 'size of dataset: {0}'.format(base_object.training_testing_dataset_object[
                                      base_object.training_data_name[vertical_horizontal]][last_batch_index:end].shape)
                        logger.getLogger('tab.regular').debug(msg)

                        if first_run:
                            hmm_models[vertical_horizontal].fit(
                                X=base_object.training_testing_dataset_object[
                                      base_object.training_data_name[vertical_horizontal]][last_batch_index:end],
                                lengths=sliced_length, logger=logger,
                                kmeans_opt=kmeans)
                            first_run = False
                        else:
                            # by setting init_params='', we will be able to cascaded the training
                            # results from the previous fitting runs
                            hmm_models[vertical_horizontal].init_params = ''
                            hmm_models[vertical_horizontal].fit(
                                X=base_object.training_testing_dataset_object[
                                      base_object.training_data_name[vertical_horizontal]][last_batch_index:end],
                                lengths=sliced_length, logger=logger, kmeans_opt=kmeans)

                        last_batch_index = end

                logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')

                # create a name for a file based on the user, activity and the time
                filename = 'hmm_' + base_object.test_user + '_' + base_object.test_activity + '_' + \
                           vertical_horizontal + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
                # calculate the whole path
                hmm_path_filename = os.path.join(base_object.data_dir, filename)
                logger.getLogger('tab.regular').debug('hmm model stored as {0}'.format(filename))
                logger.getLogger('tab.regular').debug('location {0}'.format(base_object.data_dir))

                # if data folder does not exists, make it
                if not os.path.exists(base_object.data_dir):
                    os.mkdir(base_object.data_dir)

                    # store the model so its not needed to re-train it
                joblib.dump(hmm_models[vertical_horizontal], hmm_path_filename)

            logger.getLogger('tab.regular.time').info('calculating predictions')
            train_predictions = hmm_models[vertical_horizontal].predict_proba(
                base_object.training_testing_dataset_object[base_object.training_data_name[vertical_horizontal]],
                lengths=base_object.training_dataset_lengths[vertical_horizontal])
            test_predictions = hmm_models[vertical_horizontal].predict_proba(
                base_object.training_testing_dataset_object['testing data'])

            hmm_object = ResultClass()
            # using the model, run algorithms
            hmm_object.classification(train_predictions=train_predictions,
                                      traininglabels=base_object.training_testing_dataset_object[
                                          base_object.training_label_name[vertical_horizontal]],
                                      test_predictions=test_predictions,
                                      testinglabels=base_object.training_testing_dataset_object['testing labels'],
                                      logger=logger, vertical_horizontal=vertical_horizontal)

            hmm_result.append(hmm_object)

    msg = 'comparing results'.format(vertical_horizontal)
    logger.getLogger('tab.regular.time').info(msg)
    show_results(hmm_result, logger)

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


