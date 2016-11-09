from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from utils.output import printout
import threading

def obtain_statistical_descriptors():

    n_dataset = ''

    # code starts here

    # obtain statistical descriptors
    mean_min_max = n_dataset.describe().loc[['mean', 'min', 'max']]
    # calculate statistical descriptors
    mean = mean_min_max.loc[['mean']].values
    mn = mean_min_max.loc[['min']].values
    mx = mean_min_max.loc[['max']].values
    # magnitude
    rms = np.sqrt(np.mean(np.square(n_dataset), axis=0))
    rms = rms.as_matrix().reshape(1, n_dataset.shape[1])
    # convert variance to matrix and re-shape it so it has the same shape as other statistical descriptors
    var = n_dataset.var().as_matrix().reshape(1, n_dataset.shape[1])
    tmp_df = pd.DataFrame(np.c_[mean, mn, mx, var, rms])


def normalizing_data():

    statistical_descriptors = ''

    # code starts here
    x = statistical_descriptors.values  # returns a numpy array
    min_max_scale = preprocessing.MinMaxScaler()
    x_scaled = min_max_scale.fit_transform(x)
    n_statistical_descriptors = pd.DataFrame(x_scaled)

def load_data(data_dir):

    # data file are store within the project dataset folder
    dataset_files = os.listdir(data_dir)

    # sensordata variable
    sensordata_array = np.empty(shape=(0, 0))

    # dataset user information
    dataset_user_information = list()

    # loop through every file
    for python_file in dataset_files:
        python_file_path = os.path.join(data_dir, python_file)
        print 'reading file: {0}'.format(python_file)
        # getting file information
        user, activity = file_information(python_file)
        start_index = np.shape(sensordata_array)[0]

        # read data from file
        raw_data = np.load(python_file_path)
        # adding the info to the sensordata list
        if start_index == 0:
            sensordata_array = raw_data.copy()
        else:
            sensordata_array = np.append(arr=sensordata_array, values=raw_data, axis=0)
        # get the max length of the added dataset
        end_index = np.shape(sensordata_array)[0]

        user_prop = UserInfo(user=user, activity=activity, start_index=start_index, end_index=end_index)

        dataset_user_information.append(user_prop)

        print '\tuser={0} activity={1}, start/end index={2}'.format(user, activity, (start_index, end_index))

        printout(message='\tdata stored in dataframe\n', verbose=True)

    return sensordata_array, dataset_user_information


def hmm_preprocessing_data(dataset):
    # sliding window properties
    window_size = 60
    step = 1
    chunks = sliding_window(dataset, window_size, step)

    statistical_descriptors = pd.DataFrame()
    label_series = pd.Series()

    for segmented_data in chunks:
        # obtain labels
        labels = segmented_data.ix[:, len(segmented_data.columns) - 1]
        # get the most common label
        label = pd.Series(data=[labels.value_counts().idxmax()])
        # separate the labels from the dataset
        n_dataset = segmented_data.drop(segmented_data.columns[len(segmented_data.columns) - 1], axis=1)

        # calculate statistical descriptors
        mean = n_dataset.mean().values
        mn = n_dataset.min().values
        mx = n_dataset.max().values
        var = n_dataset.var().values
        # magnitude
        rms = np.sqrt(np.mean(np.square(n_dataset))).values
        tmp_df = pd.DataFrame(data=[np.r_[mean, mn, mx, var, rms]])
        # add to the dataframe
        statistical_descriptors = statistical_descriptors.append(tmp_df)
        label_series = label_series.append(label)

    # standardization : transfer data to have zero mean and unit variance
    n_statistical_descriptors = (
                                    statistical_descriptors - statistical_descriptors.mean()) / statistical_descriptors.std()

    return n_statistical_descriptors, label_series


class IMUThread(threading.Thread):

    def __init__(self, name, dataset):

        threading.Thread.__init__(self)
        self.name = name
        self.dataset = dataset
        self.dataset_normalized = ''
        self.labels = ''
        self.thread_done = False

    def run(self):
        printout(message='Pre-processing {0}. Data:{1}'.format(self.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                 verbose=True)
        self.dataset_normalized, self.labels = hmm_preprocessing_data(dataset=self.dataset)
        printout(message='Finished pre-processing {0}. Data:{1}'.format(self.name,
                                                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                 verbose=True)

        self.thread_done = True

    def check_thread_finished(self):
        self.thread_done

def load_data(data_dir):

    # data file are store within the project dataset folder
    dataset_files = os.listdir(data_dir)

    # sensordata variable
    sensordata_dataframe = pd.DataFrame()

    # dataset user information
    dataset_user_information = list()

    # loop through every file
    for python_file in dataset_files:
        python_file_path = os.path.join(data_dir, python_file)
        msg = 'reading file: {0}'.format(python_file)
        printout(message=msg, verbose=True)
        # getting file information
        user, activity = file_information(python_file)
        start_index = sensordata_dataframe.shape[0]

        # read data from file
        raw_data = np.load(python_file_path)
        # convert the array to dataframe
        df_data = pd.DataFrame(raw_data)
        # append does not happen in place so its stored back in data_dataframe
        sensordata_dataframe = sensordata_dataframe.append(df_data)
        end_index = sensordata_dataframe.shape[0]

        user_prop = UserInfo(user=user, activity=activity, start_index=start_index, end_index=end_index)

        dataset_user_information.append(user_prop)

        print '\tuser={0} activity={1}, start/end index={2}'.format(user, activity, (start_index, end_index))

        printout(message='\tdata stored in dataframe\n', verbose=True)

    sensordata_dataframe.index = range(0, sensordata_dataframe.shape[0])
    sensordata_dataframe.columns = range(0, sensordata_dataframe.shape[1])

    return sensordata_dataframe, dataset_user_information


def preprocessing_data(dataset):
    # sliding window properties
    window_size = 60
    step = 1
    chunks = sliding_window(dataset, window_size, step)

    label_list = list()

    # mean, variance and label lists
    mean_list = list()
    variance_list = list()
    min_list = list()
    max_list = list()
    # root mean square
    rms_list = list()

    for segmented_data in chunks:
        # obtain labels
        labels = segmented_data.ix[:, len(segmented_data.columns) - 1]
        # get the most common label
        label_list.append(labels.value_counts().idxmax())
        # separate the labels from the dataset
        n_dataset = segmented_data.drop(segmented_data.columns[len(segmented_data.columns) - 1], axis=1).values

        # calculate statistical descriptors
        mean = np.mean(a=n_dataset, axis=0)
        var = np.var(a=n_dataset, axis=0)
        mn = np.min(a=n_dataset, axis=0)
        mx = np.max(a=n_dataset, axis=0)
        rms = np.sqrt(np.mean(np.square(n_dataset), axis=0))

        mean_list.append(mean)
        variance_list.append(var)
        min_list.append(mn)
        max_list.append(mx)
        rms_list.append(rms)

    # list converted to numpy arrays for future processing
    mean_points = np.array(mean_list)
    var_points = np.array(variance_list)
    min_points = np.array(min_list)
    max_points = np.array(max_list)
    rms_points = np.array(rms_list)

    statistical_descriptors = np.c_[mean_points, min_points, max_points, var_points, rms_points]
    # standardization : transfer data to have zero mean and unit variance
    sd_mean = np.mean(a=statistical_descriptors, axis=0)
    sd_std = np.std(a=statistical_descriptors, axis=0)
    n_statistical_descriptors = (statistical_descriptors - sd_mean) / sd_std

    return n_statistical_descriptors, label_list


def repeated_labels(passed_list):
    seen = set()
    for x in passed_list:
        if x in seen:
            return True
        seen.add(x)
    return False

def extract_data_and_save_to_file(labels_array='', ignored_indices='', dataset='', motion_class='', dataset_path='',
                                  current_file_name=''):

    # variable to store all the segments and vectors values
    data = np.empty((1, 1))

    for vector in motion_class.vectorsUsed:

        v_data = dataset[vector]
        if 'joint' == vector:
            for joints in motion_class.jointUsed:
                sensor_data = v_data[0][0][joints][0][0][2:]
                number_row, number_column = sensor_data.shape
                _, ds_column = data.shape
                # temporary array
                temp_array = sensor_data
                # if ds_column is 1 it is the first iteration and special measures have
                # to be taken into consideration when specifying the size of the array if not
                # check this condition, then the code would break trying to add the data
                if ds_column != 1:
                    # create new array with extra index for new data
                    temp_array = np.zeros((number_row, number_column + ds_column))
                    # merge data
                    temp_array[:, 0:ds_column] = data
                    temp_array[:, ds_column:] = sensor_data
                # add values to the final variable
                data = np.vstack(temp_array)
        else:
            for segments in motion_class.segmentUsed:
                # obtains the values based on the segments and vectors used
                sensor_data = v_data[0][0][segments][0][0][2:]
                number_row, number_column = sensor_data.shape
                _, ds_column = data.shape
                # temporary array
                temp_array = sensor_data
                # if ds_column is 1 it is the first iteration and special measures have
                # to be taken into consideration when specifying the size of the array if not
                # check this condition, then the code would break trying to add the data
                if ds_column != 1:
                    # create new array with extra index for new data
                    temp_array = np.zeros((number_row, number_column + ds_column))
                    # merge data
                    temp_array[:, 0:ds_column] = data
                    temp_array[:, ds_column:] = sensor_data
                # add values to the final variable
                data = np.vstack(temp_array)

    import IPython
    IPython.embed()

    # merge data with their respective labels
    tmp_arr = ''
    try:
        printout(message='\tMerging data and labels arrays', verbose=True)
        tmp_arr = np.c_[data, labels_array]

    except ValueError:
        msg = '\tsize of data: {0}'.format(np.shape(data))
        printout(message=msg, verbose=True)
        msg = '\tsize of labels: {0}'.format(np.shape(labels_array))
        printout(message=msg, verbose=True, extraspaces=2)
        exit(1)

    if len(ignored_indices) != 0:
        printout(message='\tRemoving \'Ignored\' labels', verbose=True)
        data_labels = remove_ignores(tmp_arr, ignored_indices)
    else:
        data_labels = tmp_arr

    # this information will be used to train the hmm since its important to know the start and end of
    # an activity in order for the EM algo to not learn from non-concurrent activities
    n_datapoints = np.shape(data_labels)[0]
    array_length = np.zeros([n_datapoints, 1])
    array_length[0, 0] = n_datapoints
    dataset = np.c_[data_labels, array_length]

    # current user and activity based on the file name
    user, activity, leftright = file_information(current_file_name)

    # list of files already processed
    files_processed = [pfile for pfile in listdir(dataset_path) if isfile(join(dataset_path, pfile))]

    # list of user already processed
    users_processed = [pfile for pfile in files_processed if (user in pfile and activity in pfile)]

    # concatenate users performing the same activities
    for uprocessed in users_processed:
        old_data = np.load(uprocessed)
        tmp_arr = np.r(old_data, dataset)
        dataset = tmp_arr

    new_file_name = user + '_' + leftright + '_' + activity
    current_out_path = os.path.join(dataset_path, new_file_name)

    msg = '\tOutput file directory: {0}'.format(current_out_path)
    printout(message=msg, verbose=True)
    np.save(current_out_path, dataset)

def load_data(data_dir):

    # data file are store within the project dataset folder
    dataset_files = os.listdir(data_dir)

    # sensordata variable
    sensordata_array = np.empty(shape=(0, 0))

    # dataset user information
    dataset_user_information = list()

    # loop through every file
    for python_file in dataset_files:
        python_file_path = os.path.join(data_dir, python_file)
        print 'reading file: {0}'.format(python_file)
        # getting file information
        user, activity = file_information(python_file)
        start_index = np.shape(sensordata_array)[0]

        # read data from file
        raw_data = np.load(python_file_path)

        # adding the info to the sensordata list
        sensordata_array = append_array(sensordata_array, raw_data)

        # get the max length of the added dataset
        end_index = np.shape(sensordata_array)[0]

        user_prop = UserInfo(user=user, activity=activity, start_index=start_index, end_index=end_index)

        dataset_user_information.append(user_prop)

        print '\tuser={0} activity={1}, start/end index={2}, size={3}'.format(user, activity, (start_index, end_index),
                                                                              np.shape(raw_data)[0])

        printout(message='\tdata stored in dataframes\n', verbose=True)

    return sensordata_array, dataset_user_information


from utils.matlablabels import MatlabLabels
import numpy as np
from utils.misc import check_sequence
from hmmlearn import hmm


def sliding_window(sequence1, sequence2, window_size, step=1):
    """
    Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.
    """

    # Verify the inputs
    if not isinstance(type(window_size), type(0)) and isinstance(type(step), type(0)):
        raise Exception("**ERROR** type(window_size) and type(step) must be int.")
    if step > window_size:
        raise Exception("**ERROR** step must not be larger than window_size.")
    if window_size > len(sequence1):
        raise Exception("**ERROR** window_size must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    number_of_chunks = ((len(sequence1) - window_size) / step) + 1

    # Do the work
    for i in range(0, number_of_chunks * step, step):
        yield sequence1[i: i + window_size], sequence2[i: i + window_size]


def hmm_algo(trainingdataset, traininglabels, testingdataset, testinglabels, batched_setting, user, activity,
             program_path, logger, algorithm, kmeans, lengths='', quickrun=True):
    label_class = MatlabLabels()
    # all labels used in the activities
    possible_labels = label_class.compact_list

    hmm_dict = dict()

    # filter all the points in the dataset relevant to 'label'
    # this will be used to train each HMM
    for label_index, label in enumerate(possible_labels):
        keep_indices = np.where(traininglabels[:] == label_index)

        # obtain the lengths of the continuous sequences so the HMM trains on the right information
        current_lengths = check_sequence(keep_indices)

        hmm_model = hmm.GaussianHMM(n_components=8, verbose=True)
        hmm_model.fit(X=trainingdataset[keep_indices], lengths=current_lengths)

        hmm_dict[label] = hmm_model

    window_size = 60
    chunks = sliding_window(trainingdataset, traininglabels, window_size=window_size, step=60)
    for segmented_data, segmented_labels in chunks:

        best_prediction = ''

        previous_accuracy = 0.0
        for key, hmm_model in hmm_dict.iteritems():
            accuracy = hmm_model.score(segmented_data)

            if accuracy > previous_accuracy:
                best_prediction = [key] * len(accuracy)

def hmm_algo(trainingdataset, traininglabels, testingdataset, testinglabels, quickrun, batched_setting, user, activity,
             program_path, logger, algorithm, kmeans, lengths=''):

    if quickrun:

        root_folder = '/'.join(program_path.split('/')[:-1])
        data_dir = os.path.join(root_folder, 'data')

        files_in_data = ''
        if os.path.exists(data_dir):
            files_in_data = os.listdir(data_dir)

        loaded_model = False

        # check all the files in the folder and look for the model file
        for sfile in files_in_data:
            # check if user, activity and hmm keyword are part of the file
            if (user in sfile) and (activity in sfile) and ('hmm' in sfile) and ('.npy' not in sfile):
                logger.getLogger('line.tab.regular').info('hmm model found')
                logger.getLogger('tab.regular.line').info('using hmm model {0}'.format(sfile))
                # calculate the whole path
                data_path = os.path.join(data_dir, sfile)
                # load the model
                hmm_model = joblib.load(data_path)
                # turn on flag so the code does not re-train the model
                loaded_model = True
                logger.getLogger('tab.regular.time').info('hmm model loaded')
                break

        # check if flag is on
        if not loaded_model:
            # train model
            if algorithm == 'GHMM':
                logger.getLogger('tab.regular.time').info('starting training Gaussian Hidden Markov Model.')
                hmm_model = hmm.GaussianHMM(n_components=8, covariance_type='full', n_iter=10, verbose=True)

                if batched_setting:
                    first_run = True
                    total_batches, batched_lengths = batch(lengths, 50)

                    last_batch_index = 0
                    end = 0
                    for index, sliced_length in enumerate(batched_lengths):

                        msg = 'starting training Gaussian Hidden Markov Model on batch {0} out of {1}'. \
                            format(index, total_batches)
                        logger.getLogger('tab.regular.time').info(msg)

                        end += np.sum(sliced_length).astype(np.int32)
                        msg = 'size of dataset: {0}'.format(trainingdataset[last_batch_index:end].shape)
                        logger.getLogger('tab.regular').debug(msg)

                        if first_run:
                            hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
                                          activity=activity, data_dir=data_dir, lengths=sliced_length,
                                          quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
                            first_run = False
                        else:
                            # by setting init_params='', we will be able to cascaded the training
                            # results from the previous fitting runs
                            hmm_model.init_params = ''
                            hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
                                          activity=activity, data_dir=data_dir, lengths=sliced_length,
                                          quickrun=quickrun, logger=logger, kmeans_opt=kmeans)

                        last_batch_index = end
                else:
                    msg = 'starting training Gaussian Hidden Markov Model'
                    logger.getLogger('tab.regular.time').info(msg)

                    hmm_model.fit(X=trainingdataset, user=user,
                                  activity=activity, data_dir=data_dir, lengths=lengths,
                                  quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
            else:
                logger.getLogger('tab.regular.time').info('starting training GMM Hidden Markov Model.')
                hmm_model = hmm.GMMHMM(n_components=8, n_mix=6)

            hmm_model.fit(X=trainingdataset, user=user, activity=activity, data_dir=data_dir, lengths=lengths,
                          quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
            logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')

            # create a name for a file based on the user, activity and the time
            filename = 'hmm_' + user + '_' + activity + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
            # calculate the whole path
            data_path = os.path.join(data_dir, filename)
            logger.getLogger('tab.regular').debug('hmm model stored as {0}'.format(filename))
            logger.getLogger('tab.regular').debug('location {0}'.format(data_dir))

            # if data folder does not exists, make it
            if not os.path.exists(root_folder):
                os.mkdir(root_folder)

                # store the model so its not needed to re-train it
            joblib.dump(hmm_model, data_path)

        logger.getLogger('tab.regular.time').info('calculating predictions')
        train_predictions = hmm_model.predict_proba(trainingdataset, lengths=lengths)
        test_predictions = hmm_model.predict_proba(testingdataset[:])

        # using the model, run algorithms
        results(train_predictions=train_predictions, traininglabels=traininglabels[:],
                test_predictions=test_predictions, testinglabels=testinglabels[:], logger=logger)

    else:
        if algorithm == 'GMMHMM':
            n_iterations = [10, 50]
            components = [8, 10, 15, 20]
            mixes = [6, 9, 12]
            tolerance = [0.01]
            covariance_types = ['spherical', 'diag', 'full', 'tied']
            for nc in components:
                for nm in mixes:
                    for ct in covariance_types:
                        for t in tolerance:
                            for ni in n_iterations:
                                logger.getLogger('tab.regular.time').info('starting training GMM Hidden Markov Model.')
                                logger.getLogger('tab.regular').info('\tmodel parameters')
                                msg = '\t\tnumber of states:{0}'.format(nc)
                                logger.getLogger('tab.regular').info(msg)
                                msg = '\t\tnumber of mixtures:{0}'.format(nm)
                                logger.getLogger('tab.regular').info(msg)
                                msg = '\t\tnumber of iterations:{0}'.format(ni)
                                logger.getLogger('tab.regular').info(msg)
                                msg = '\t\ttolerance:{0}'.format(t)
                                logger.getLogger('tab.regular').info(msg)
                                msg = '\t\tcovariance type:{0}'.format(ct)
                                logger.getLogger('tab.regular').info(msg)

                                try:
                                    logger.getLogger('tab.regular.time').info('starting training GMM-HMM.')
                                    hmm_model = hmm.GMMHMM(n_components=nc, n_mix=nm, covariance_type=ct, n_iter=ni, tol=t)
                                    hmm_model.fit(X=trainingdataset, user=user, activity=activity, data_dir='',
                                                  lengths=lengths, quickrun=quickrun, logger=logger)
                                    logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')

                                    logger.getLogger('tab.regular.time').info('calculating predictions')
                                    train_predictions = hmm_model.predict_proba(trainingdataset[:])
                                    test_predictions = hmm_model.predict_proba(testingdataset[:])

                                    # using the model, run algorithms
                                    results(train_predictions=train_predictions, traininglabels=traininglabels[:],
                                            test_predictions=test_predictions, testinglabels=testinglabels[:],
                                            logger=logger)
                                except ValueError as error_message:
                                    logger.getLogger('tab.regular.time').error(error_message)
        else:

            root_folder = '/'.join(program_path.split('/')[:-1])
            data_dir = os.path.join(root_folder, 'data')

            n_iterations = [10]
            components = [8]
            tolerance = [0.01]
            covariance_types = [ 'diag']
            for ni in n_iterations:
                for ct in covariance_types:
                    for nc in components:
                        for t in tolerance:
                            logger.getLogger('tab.regular.time').info(
                                'running Gaussian Hidden Markov Model with the following model parameters:')
                            msg = '\t\tnumber of states:{0}'.format(nc)
                            logger.getLogger('tab.regular').info(msg)
                            msg = '\t\tnumber of iterations:{0}'.format(ni)
                            logger.getLogger('tab.regular').info(msg)
                            msg = '\t\ttolerance:{0}'.format(t)
                            logger.getLogger('tab.regular').info(msg)
                            msg = '\t\tcovariance type:{0}'.format(ct)
                            logger.getLogger('tab.regular').info(msg)

                            try:
                                hmm_model = hmm.GaussianHMM(n_components=nc, covariance_type=ct, n_iter=ni,
                                                            verbose=True, tol=t)

                                if batched_setting:
                                    first_run = True
                                    total_batches, batched_lengths = batch(lengths, 40)

                                    last_batch_index = 0
                                    end = 0
                                    for index, sliced_length in enumerate(batched_lengths):

                                        msg = 'starting training Gaussian Hidden Markov Model on batch {0} out of {1}'. \
                                            format(index, total_batches)
                                        logger.getLogger('tab.regular.time').info(msg)

                                        end += np.sum(sliced_length).astype(np.int32)
                                        msg = 'size of dataset: {0}'.format(trainingdataset[last_batch_index:end].shape)
                                        logger.getLogger('tab.regular').debug(msg)

                                        if first_run:
                                            hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
                                                          activity=activity, data_dir='', lengths=sliced_length,
                                                          quickrun=quickrun, logger=logger, kmeans_opt=kmeans)
                                            first_run = False
                                        else:
                                            # by setting init_params='', we will be able to cascaded the training
                                            # results from the previous fitting runs
                                            hmm_model.init_params = ''
                                            hmm_model.fit(X=trainingdataset[last_batch_index:end], user=user,
                                                          activity=activity, data_dir='', lengths=sliced_length,
                                                          quickrun=quickrun, logger=logger, kmeans_opt=kmeans)

                                        last_batch_index = end
                                else:
                                    msg = 'starting training Gaussian Hidden Markov Model'
                                    logger.getLogger('tab.regular.time').info(msg)

                                    hmm_model.fit(X=trainingdataset, user=user,
                                                  activity=activity, data_dir='', lengths=lengths,
                                                  quickrun=quickrun, logger=logger, kmeans_opt=kmeans)

                                logger.getLogger('tab.regular.time').info('finished training Hidden Markov Model.')

                                # create a name for a file based on the user, activity and the time
                                filename = 'hmm_' + user + '_' + activity + '_' + str(nc) + '_' + str(ct) + '_' + \
                                           str(t) + '_' + str(ni) + '_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
                                # calculate the whole path
                                data_path = os.path.join(data_dir, filename)
                                logger.getLogger('tab.regular').info('hmm model stored as {0}'.format(filename))
                                logger.getLogger('tab.regular').info('location {0}'.format(data_dir))

                                # if data folder does not exists, make it
                                if not os.path.exists(root_folder):
                                    os.mkdir(root_folder)

                                    # store the model so its not needed to re-train it
                                joblib.dump(hmm_model, data_path)

                                logger.getLogger('tab.regular.time').info('calculating predictions')
                                train_predictions = hmm_model.predict_proba(trainingdataset[:], lengths=lengths)
                                test_predictions = hmm_model.predict_proba(testingdataset[:])

                                # using the model, run algorithms
                                results(train_predictions=train_predictions, traininglabels=traininglabels[:],
                                        test_predictions=test_predictions, testinglabels=testinglabels[:],
                                        logger=logger)

                            except ValueError as error_message:
                                logger.getLogger('tab.regular.time').error(error_message)


















