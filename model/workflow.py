from utils.output import printout
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
from model.lstm import lstm_algo
from utils.matlablabels import MatlabLabels
import numpy as np
import h5py
import os
from datetime import datetime
from sklearn.decomposition import PCA, TruncatedSVD


def imu_algorithm(doc, algorithm='', quickrun='', program_path='', logger='', kmeans='',
                  batched_setting=False):
    # type: (object, str, boolean, str, object, str, bool) -> object

    label_object = MatlabLabels()

    h5_file_object = h5py.File(doc.input_path_filename, 'r')

    # printing a line for style and visibility
    printout(message='', verbose=True)

    for user_index, user_info in enumerate(h5_file_object.iterkeys()):

        # run test on control users only
        # if 'pilot' in user_info and \
        #        ('HS00' in user_info or 'N537' in user_info or 'Q130' in user_info or 'Q430' in user_info or 'Q435' in
        #            user_info):

        if 'pilot' in user_info and \
                ('Q439' in user_info or 'Q568' in user_info or 'Q615' in user_info or 'Q616' in user_info or 'Q617' in
                    user_info):
            # defining train dataset and labels array
            c_filename = 'training_testing_file_' + str(user_info) + '_' + datetime.now().strftime('%Y%m%d%H%M%S')\
                         + '.hdf5'
            training_file_name = os.path.join(doc.input_path, c_filename)
            training_testing_dataset_object = h5py.File(training_file_name, 'w')

            user = h5_file_object[user_info].attrs['user']
            activity = h5_file_object[user_info].attrs['activity']
            n_row, n_col = np.shape(h5_file_object[user_info][:, :-1])
            training_testing_dataset_object.create_dataset(name='testing data', shape=(n_row, n_col))
            training_testing_dataset_object['testing data'][:, :] = h5_file_object[user_info].value[:, :-1]
            training_testing_dataset_object.create_dataset(name='testing labels', shape=(n_row, 1))
            training_testing_dataset_object['testing labels'][:, 0] = h5_file_object[user_info].value[:, -1]

            msg = 'Starting analysing {0}'.format(user_info)
            logger.getLogger('regular.time').info(msg)

            msg = 'Calculating training and testing dataset'
            logger.getLogger('regular.time').info(msg)
            # fetch testing data from the objects

            # length of each dataset
            training_dataset_lengths = list()

            # flag used to create a dataset for the first file
            first_file = True

            # keep track of the total number of rows
            total_inner_row = 0

            # adding user's flag
            adding = False

            total_inner_users = len(h5_file_object) - 1
            # fetch training data from the objects without :
            #   1. the testing data i.e. the data of user_index
            #   2. other dataset with the same user and activity but different repetition
            for u_index, user_info_inner in enumerate(h5_file_object.iterkeys()):

                # get the attributes of the training example
                inner_user = h5_file_object[user_info_inner].attrs['user']
                inner_activity = h5_file_object[user_info_inner].attrs['activity']

                # shape of the current dataset
                n_inner_row, n_inner_column = h5_file_object[user_info_inner].shape

                # removing label columns
                n_inner_column -= 1

                if inner_user != user and 'paretic' not in user_info_inner:
                    # get type of activity i.e. horizontal, vertical or freedly
                    type_activity = label_object.check_type_activity(str(activity))
                    inner_type_activity = label_object.check_type_activity(str(inner_activity))

                    # if testing on the feeding activity or 'freedly' activities, always add other training activities
                    # TO DO: I can add training on the same user but different activities to see if that improves the
                    # feeding activity
                    if (type_activity == 'freedly') and (user != inner_user):
                        adding = True

                    # removing all the users with the freedly activities, check if they have the same type of activity
                    # and add other users
                    # TO DO: I can add training on the same user but different activities to see if that improves the
                    # feeding activity
                    if (type_activity != 'freedly') and (type_activity == inner_type_activity):
                        adding = True

                if adding:

                    # get the size of the dataset because it will be passed as an parameter to the hmm
                    total_inner_row += h5_file_object[user_info_inner].shape[0]

                    if first_file:
                        training_testing_dataset_object.create_dataset(name='training data',
                                                                       shape=(total_inner_row, n_inner_column),
                                                                       maxshape=(None, n_inner_column), chunks=True)

                        training_testing_dataset_object.create_dataset(name='training labels',
                                                                       shape=(total_inner_row, 1),
                                                                       maxshape=(None, 1), chunks=True)

                        training_testing_dataset_object['training data'][:, :] = \
                            h5_file_object[user_info_inner].value[:, :-1]
                        training_testing_dataset_object['training labels'][:, 0] = \
                            h5_file_object[user_info_inner].value[:, -1]
                        first_file = False

                    else:
                        # resize the dataset to accommodate the new data
                        training_testing_dataset_object['training data'].resize(total_inner_row, axis=0)
                        training_testing_dataset_object['training data'][index_start_appending:] = \
                            h5_file_object[user_info_inner].value[:, :-1]

                        training_testing_dataset_object['training labels'].resize(total_inner_row, axis=0)
                        training_testing_dataset_object['training labels'][index_start_appending:, 0] = \
                            h5_file_object[user_info_inner].value[:, -1]

                    index_start_appending = total_inner_row

                    training_dataset_lengths.append(n_inner_row)
                    msg = 'Including {0} (user index {1} of {2} length:{3})'.format(user_info_inner, u_index,
                                                                                    total_inner_users, n_inner_row)
                    logger.getLogger('tab.regular').info(msg)

                    # reset adding
                    adding = False

                elif not adding:
                    msg = 'Skipping {0} (user index {1} of {2})'.format(user_info_inner, u_index, total_inner_users)
                    logger.getLogger('tab.regular').info(msg)

                else:
                    msg = 'problem while processing {0} (user index {1} of {2})'.format(user_info_inner, u_index,
                                                                                        total_inner_users)
                    logger.getLogger('tab.regular').error(msg)
                    logger.getLogger('tab.regular').error('user was not added')
                    exit(1)

            training_dataset_lengths = np.array(training_dataset_lengths)
            training_data_object = training_testing_dataset_object['training data']
            training_label_object = training_testing_dataset_object['training labels']
            testing_data_object = training_testing_dataset_object['testing data']
            testing_label_object = training_testing_dataset_object['testing labels']

            msg = 'Training data size:{0}'.format(training_data_object.shape)
            logger.getLogger('line.tab.regular').info(msg)
            msg = 'Training labels size:{0}'.format(training_label_object.shape)
            logger.getLogger('tab.regular').info(msg)
            msg = 'Testing data size:{0}'.format(testing_data_object.shape)
            logger.getLogger('tab.regular').info(msg)
            msg = 'Testing data size:{0}'.format(testing_label_object.shape)
            logger.getLogger('tab.regular.line').info(msg)
            
            components = [350, 400, 450, 500]
            for c in components:

                msg = 'n_components:{0}'.format(c)
                logger.getLogger('tab.regular.line').info(msg)

                logger.getLogger('tab.regular.line').info('pca algo')
                pca = PCA(n_components=c)
    
                n_training_data_object = pca.fit_transform(X=training_data_object[:])
                training_testing_dataset_object.create_dataset(name='new training data', data=n_training_data_object)
                logger.getLogger('tab.regular.line').info('variance explanation')
                logger.getLogger('tab.regular.line').info(pca.explained_variance_)
        
                new_training_data_object = training_testing_dataset_object['new training data']
                msg = 'new Training data size:{0}'.format(new_training_data_object.shape)
                logger.getLogger('line.tab.regular').info(msg)
        
                n_testing_data_object = pca.transform(X=testing_data_object[:])
                training_testing_dataset_object.create_dataset(name='new testing data', data=n_testing_data_object)
                new_testing_data_object = training_testing_dataset_object['new testing data']
    
                msg = 'New training data size:{0}'.format(new_training_data_object.shape)
                logger.getLogger('line.tab.regular').info(msg)
                msg = 'New testing data size:{0}'.format(training_data_object.shape)
                logger.getLogger('line.tab.regular').info(msg)
                try:
                    if algorithm == 'GHMM' or algorithm == 'GMMHMM':
                        hmm_algo(trainingdataset=new_training_data_object, traininglabels=training_label_object,
                                 quickrun=quickrun, testingdataset=new_testing_data_object, testinglabels=testing_label_object,
                                 lengths=training_dataset_lengths, algorithm=algorithm, batched_setting=batched_setting,
                                 user=user, activity=activity, program_path=program_path, logger=logger, kmeans=kmeans)
        
                    elif algorithm == 'Logistic Regression':
                        logreg_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                                    quickrun=quickrun, testingdataset=testing_data_object, logger=logger,
                                    testinglabels=testing_label_object)
        
                    elif algorithm == 'LSTM':
                        lstm_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                                  testingdataset=testing_data_object, testinglabels=testing_label_object,
                                  lengths=training_dataset_lengths, logger=logger)
        
                    else:
                        printout(message='Wrong algorithm provided.', verbose=True)
        
                    # closing h5py file
                    # training_testing_dataset_object.close()
        
                    msg = 'Finished analysing {0}'.format(user_info)
                    logger.getLogger('tab.regular.time.line').info(msg)
        
                except ValueError as error_message:
                    msg = 'Error while analysing {0}'.format(user_info)
                    logger.getLogger('tab.regular.time').error(msg)
                    logger.getLogger('tab.regular.time.line').eror(error_message)
        
                # removing training dataset h5py file
                # os.remove(training_file_name)
        
                # exit(0)

                logger.getLogger('tab.regular.line').info('lda algo')

                lda = TruncatedSVD(n_components=c)
    
                n_training_data_object = lda.fit_transform(X=training_data_object[:])
                training_testing_dataset_object.create_dataset(name='new training data', data=n_training_data_object)
                logger.getLogger('tab.regular.line').info('variance explanation')
                logger.getLogger('tab.regular.line').info(pca.explained_variance_)
        
                new_training_data_object = training_testing_dataset_object['new training data']
                msg = 'new Training data size:{0}'.format(new_training_data_object.shape)
                logger.getLogger('line.tab.regular').info(msg)
        
                n_testing_data_object = lda.transform(X=testing_data_object[:])
                training_testing_dataset_object.create_dataset(name='new testing data', data=n_testing_data_object)
                new_testing_data_object = training_testing_dataset_object['new testing data']
    
                msg = 'New training data size:{0}'.format(new_training_data_object.shape)
                logger.getLogger('line.tab.regular').info(msg)
                msg = 'New testing data size:{0}'.format(training_data_object.shape)
                logger.getLogger('line.tab.regular').info(msg)
                try:
                    if algorithm == 'GHMM' or algorithm == 'GMMHMM':
                        hmm_algo(trainingdataset=new_training_data_object, traininglabels=training_label_object,
                                 quickrun=quickrun, testingdataset=new_testing_data_object, testinglabels=testing_label_object,
                                 lengths=training_dataset_lengths, algorithm=algorithm, batched_setting=batched_setting,
                                 user=user, activity=activity, program_path=program_path, logger=logger, kmeans=kmeans)
        
                    elif algorithm == 'Logistic Regression':
                        logreg_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                                    quickrun=quickrun, testingdataset=testing_data_object, logger=logger,
                                    testinglabels=testing_label_object)
        
                    elif algorithm == 'LSTM':
                        lstm_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                                  testingdataset=testing_data_object, testinglabels=testing_label_object,
                                  lengths=training_dataset_lengths, logger=logger)
        
                    else:
                        printout(message='Wrong algorithm provided.', verbose=True)
        
                    # closing h5py file
                    # training_testing_dataset_object.close()
        
                    msg = 'Finished analysing {0}'.format(user_info)
                    logger.getLogger('tab.regular.time.line').info(msg)
        
                except ValueError as error_message:
                    msg = 'Error while analysing {0}'.format(user_info)
                    logger.getLogger('tab.regular.time').error(msg)
                    logger.getLogger('tab.regular.time.line').eror(error_message)
        
                # removing training dataset h5py file
                # os.remove(training_file_name)
        
                # exit(0)
