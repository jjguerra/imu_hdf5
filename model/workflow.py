from utils.output import printout
from model.hmm import hmm_algo
from model.logistic_regression import logreg_algo
from model.lstm import lstm_algo
from utils.matlablabels import MatlabLabels
import h5py
from model import base


def imu_algorithm(doc, algorithm='', quickrun='', logger='', kmeans='', window_size='', n_states='',
                  batched_setting=False):
    # type: (object, str, boolean, str, object, str, bool) -> object

    label_object = MatlabLabels()

    h5_file_object = h5py.File(doc.input_path_filename, 'r')

    # printing a line for style and visibility
    printout(message='', verbose=True)

    for user_index, user_info in enumerate(h5_file_object.iterkeys()):

        # only consider control users that are not performing feeding activity
        if ('pilot' in user_info) and ('feeding' not in user_info):
        # run test on control users only
        # if 'pilot' in user_info and \
        #        ('HS00' in user_info or 'N537' in user_info or 'Q130' in user_info or 'Q430' in user_info or 'Q435' in
        #            user_info):

        # if 'pilot' in user_info and \
        #         ('Q439' in user_info or 'Q568' in user_info or 'Q615' in user_info or 'Q616' in user_info or 'Q617' in
        #             user_info) and ('feeding' not in user_info):

            # get user, activity and activity type
            user = h5_file_object[user_info].attrs['user']
            activity = h5_file_object[user_info].attrs['activity']

            # initialize user object
            base_object = base.Base(input_path=doc.input_path, filename=user_info, user=user, activity=activity,
                                    dataset=h5_file_object[user_info], window_size=window_size, n_states=n_states)

            msg = 'Starting analysing {0}'.format(user_info)
            logger.getLogger('regular.time').info(msg)

            msg = 'Calculating training and testing dataset'
            logger.getLogger('regular.time').info(msg)

            # flag used to include or exclude user to/from training dataset
            adding = False

            # total number of users being considered
            total_inner_users = len(h5_file_object) - 1

            # fetch training data from the objects without :
            #   1. the testing user dataset
            #   2. other dataset with the same user and activity
            for u_index, user_info_inner in enumerate(h5_file_object.iterkeys()):

                # get the attributes of the training example
                inner_user = h5_file_object[user_info_inner].attrs['user']
                inner_activity = h5_file_object[user_info_inner].attrs['activity']

                # add the activities for the other control users that are not performing the feeding activity
                if (inner_user != user) and ('paretic' not in user_info_inner) and ('feeding' not in user_info_inner)\
                        and (activity == inner_activity):
                    adding = True

                # string to print for logging information
                print_str = '{0} (user index {1} of {2})'.format(user_info_inner, u_index, total_inner_users)

                if adding:

                    # add user data to the vertical or horizontal dataset
                    base_object.add_dataset(dataset=h5_file_object[user_info_inner])

                    msg = 'Including {0}'.format(print_str)
                    logger.getLogger('tab.regular').info(msg)

                    # reset flag
                    adding = False

                elif not adding:
                    msg = 'Excluding {0}'.format(print_str)
                    logger.getLogger('tab.regular').info(msg)

                else:
                    msg = 'Error while processing {0}. User was not added'.format(print_str)
                    logger.getLogger('tab.regular').error(msg)
                    raise ValueError(msg)

            # logging information
            for dataset_type in ['Training', 'Testing']:
                data_size, label_size = base_object.get_shape(dataset_type)
                msg = '{0} data size:{1}'.format(dataset_type, data_size)
                logger.getLogger('line.tab.regular').info(msg)

                msg = '{0} labels size:{0}'.format(dataset_type, label_size)
                logger.getLogger('tab.regular').info(msg)

            logger.getLogger('tab.regular').info('')
            try:
                if algorithm == 'GHMM' or algorithm == 'GMMHMM':
                    hmm_algo(base_object=base_object, algorithm=algorithm, batched_setting=batched_setting,
                             logger=logger, kmeans=kmeans, quickrun=quickrun, n_states=n_states)
 
                # elif algorithm == 'Logistic Regression':
                #     logreg_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                #                 quickrun=quickrun, testingdataset=testing_data_object, logger=logger,
                #                 testinglabels=testing_label_object)
                #
                # elif algorithm == 'LSTM':
                #     lstm_algo(trainingdataset=training_data_object, traininglabels=training_label_object,
                #               testingdataset=testing_data_object, testinglabels=testing_label_object,
                #               lengths=training_dataset_lengths, logger=logger)
 
                else:
                    printout(message='Wrong algorithm provided.', verbose=True)
 
                msg = 'Finished analysing {0}'.format(user_info)
                logger.getLogger('tab.regular.time.line').info(msg)
 
            except ValueError as error_message:
                msg = 'Error while analysing {0}'.format(user_info)
                logger.getLogger('tab.regular.time').error(msg)
                logger.getLogger('tab.regular.time.line').error(error_message)
        
            # closing and deleting h5py file
            base_object.close_and_delete()