import os
import sys
import numpy as np
import scipy.io as sio
from imu.utils.matlablabels import MatlabLabels


def error_message_func(line, label, error_message, debugging, logger):

    if label:
        str_error = 'Line={0}  Label={1} Error={2}'.format(line + 1, label, error_message)
    else:
        str_error = 'Line={0}  Error={1}'.format(line + 1, error_message)

    print '\t\t' + str_error
    logger.append('\t' + str_error + '\n')

    if not debugging:
        exit(1)


def data_collection(file_properties, debugging, extract):

    # set of motions and labels
    motion_class = MatlabLabels()

    ignore_index_list = list()  # ignored list indices

    # flags used for headers in the log file
    first_pass_ever = True

    # loop through each activity
    for activity, matlab_file_list in file_properties.matlab_files_path_dict.iteritems():

        # used for logging information
        first_pass_activity = True

        # loop through all the matlab files
        for index_matlab_file, matlab_file in enumerate(matlab_file_list):

            # get name of matlab file
            matlab_file_name = file_properties.matlab_files_names_dict[activity][index_matlab_file]
            print 'On activity: {}'.format(activity)
            print 'Accessing file: {}'.format(matlab_file_name)
            if matlab_file_name not in matlab_file:
                error_msg = 'Fatal Error. missing file={0} for activity={1}'.format(matlab_file_name, activity)
                print error_msg
                file_properties.output_file_object.write('\t' + error_msg + '\n')
                if not debugging:
                    exit(1)

            # load matlab file content
            print '\tMatlab content has been loaded'
            matlab_content = sio.loadmat(matlab_file)

            # keep track of error within specific files
            temp_log_file_content = list()

            if not ('tree' in matlab_content):
                error_msg = 'Fatal error. tree structure does not exists.'
                print '\t' + error_msg
                temp_log_file_content.append('\t' + error_msg + '\n')
                if not debugging:
                    exit(1)
            else:
                print '\ttree structure exists...'

            # total number of steps recorded by tree structure
            total_number_timesteps = \
                matlab_content['tree']['subject'][0][0]['frames'][0][0]['frame'][0][0]['index'][0][-1][0][0]

            # this is temporary because if the specific file has no errors then no information
            # about the file will be written
            temp_log_file_content = list()

            # tree2 contains the sensor data
            tree2 = True
            if not ('tree2' in matlab_content):
                error_msg = 'Fatal error. tree2 structure does not exists.'
                print '\t' + error_msg
                temp_log_file_content.append('\t' + error_msg + '\n')
                tree2 = False
                if not debugging:
                    exit(1)
            else:
                print '\ttree2 structure exists...'

            # markerExtract contains the label data
            marker_extract = True
            # check for marketExtract
            if not ('markerExtract' in matlab_content):
                error_msg = 'Fatal error. MarkerExtract structure does not exists.'
                print '\t' + error_msg
                temp_log_file_content.append('\t' + error_msg + '\n')
                marker_extract = False
                if not debugging:
                    exit(1)
            else:
                print '\tMarkerExtract structure exists...'

            if marker_extract:

                if 'paretic' in matlab_file_name:
                    if '_paretic_' in matlab_file_name:
                        right_left_paretic_nonparetic_hand_expectation = 'P_'
                    elif '_nonparetic_' in matlab_file_name:
                        right_left_paretic_nonparetic_hand_expectation = 'N_'
                    else:
                        print 'Failed to distinguished between paretic or nonparetic patient.'
                        exit(1)

                else:
                    if '_r_' in matlab_file_name:
                        right_left_paretic_nonparetic_hand_expectation = 'R_'
                    elif '_l_' in matlab_file_name:
                        right_left_paretic_nonparetic_hand_expectation = 'L_'
                    else:
                        print 'Failed to distinguished between right or left hand patient.'
                        exit(1)

                # get label data
                print '\tAccessing MarkerExtract data ...'
                # row_data[0][0] = label
                # row_data[1][0][0] = time step
                data_array = matlab_content['markerExtract']

                # use for checking incoming labels
                start_flag = True
                end_flag = False
                wrong_begin_end_label = False
                previous_label = ''
                expected_label = ''

                # used to overcome index roadblocks
                first_pass = True

                # keep track of new labels
                new_label_list = list()

                # values corresponding labels
                last_timestep = -1

                # variable to store labels
                label_list = list()

                print '\tTraversing data array ...'
                # loop through each row in the markerExtract file
                for current_row_number, data in enumerate(data_array):

                    print total_number_timesteps
                    if total_number_timesteps < current_row_number:
                        error_msg = 'mismatch between number of time steps of tree and markerExtract'
                        error_message_func(line=current_row_number, error_message=error_msg, debugging=debugging,
                                           logger=temp_log_file_content, label='')

                    # check for label
                    try:
                        # read label and convert it upper case
                        current_label = str(data[0][0]).upper()

                    # if empty row_value
                    except ValueError:
                        error_message_func(line=current_row_number, error_message='missing label', debugging=debugging,
                                           logger=temp_log_file_content, label='')
                    except IndexError:
                        error_msg = 'Failed to get label. Probably empty cell []'
                        error_message_func(line=current_row_number, error_message=error_msg, debugging=debugging,
                                           logger=temp_log_file_content, label='')
                        # switch flags
                        start_flag = not start_flag
                        end_flag = not end_flag
                        continue

                    # check for time step
                    try:
                        # read the time step
                        current_timestep = data[1][0][0].astype(int)

                    # if empty row_value
                    except ValueError:
                        error_message_func(line=current_row_number, error_message='missing time step', label='',
                                           debugging=debugging, logger=temp_log_file_content)

                        # switch flags
                        start_flag = not start_flag
                        end_flag = not end_flag
                        continue

                    print '\t\tdata cell information: row={0} label={1} timestep={2}'.format(current_row_number + 1,
                                                                                             current_label,
                                                                                             current_timestep)

                    # remove space in the label
                    if ' ' in current_label:
                        error_message_func(line=current_row_number, error_message='extra space', debugging=debugging,
                                           logger=temp_log_file_content, label=current_label)

                        # removed space in label
                        current_label = str(current_label).replace(" ", "")

                    if not (right_left_paretic_nonparetic_hand_expectation in current_label) and \
                            not ('IGNORE_B' in current_label) and not ('IGNORE_E' in current_label):
                        error_msg = 'Expecting label to start with \'' + \
                                    right_left_paretic_nonparetic_hand_expectation + '\''
                        error_message_func(line=current_row_number, label=current_label, error_message=error_msg,
                                           debugging=debugging, logger=temp_log_file_content)
                        tmp_label = list(current_label)
                        tmp_label[0:2] = right_left_paretic_nonparetic_hand_expectation
                        current_label = ("".join(tmp_label)).upper()

                    if right_left_paretic_nonparetic_hand_expectation in current_label:
                        tmp_label = list(current_label)
                        # removes 'R, L, N or P' in the label in order to find the label in the label class
                        current_label = "".join(tmp_label[1:])

                    # check timestep are increasing
                    if last_timestep > current_timestep and not first_pass:
                        error_msg = 'timestep=' + str(current_timestep + 3) + ' Expected timestep > ' + \
                                    str(last_timestep + 3)
                        error_message_func(line=current_row_number, label=current_label,
                                           error_message=error_msg, debugging=debugging,
                                           logger=temp_log_file_content)

                    # check if wrong '_B' or '_E' label was previously encountered
                    # if wrong label was seeing, then forget expected label and start again
                    if wrong_begin_end_label:
                        if '_B' in current_label:
                            # switch flags
                            start_flag = True
                            end_flag = False
                        elif '_E' in current_label:
                            # switch flags
                            start_flag = False
                            end_flag = True
                        else:
                            error_msg = 'Wrong label suffix'
                            error_message_func(line=current_row_number, label=current_label, error_message=error_msg,
                                               debugging=debugging, logger=temp_log_file_content)

                        wrong_begin_end_label = False

                    # start of activity
                    if start_flag:
                        # make sure the 'Begging' label exists
                        if not ('_B' in current_label):
                            if previous_label:
                                error_msg = 'Expecting label ending in \'_B\' since the last label was \'' + \
                                            previous_label + '\''
                            else:
                                error_msg = 'Expecting label ending in \'_B\''
                            error_message_func(line=current_row_number, label=current_label, error_message=error_msg,
                                               debugging=debugging, logger=temp_log_file_content)
                            wrong_begin_end_label = True

                        else:

                            # reduce the starting position by 3 (model's specifications)
                            current_timestep -= 3

                            # For a '_B' label, the timestep or value of the label has to increase by one from the
                            # previous timestep/value's label. For the error message, we increase it by 2 because
                            # of the requirements of the matlab file (project specific)
                            if last_timestep != current_timestep and not first_pass:
                                error_msg = 'timestep=' + str(current_timestep + 3) + ' Expected timestep >= ' + \
                                            str(last_timestep + 3)
                                error_message_func(line=current_row_number, label=current_label,
                                                   error_message=error_msg, debugging=debugging,
                                                   logger=temp_log_file_content)

                            # error check
                            # create a new label in order to compare it to the next label
                            temporary_variable = list(current_label)
                            temporary_variable[-1] = 'E'
                            # store label for future comparison
                            expected_label = ("".join(temporary_variable)).upper()

                    # '_E' label
                    elif end_flag:

                        # reduce the starting position by 2 (model's specifications) and in order to account
                        # for python not including the last index
                        current_timestep -= 2

                        # make sure the 'Ending' label exists
                        # compare current label to the expected (obtained from changing the previous/start label
                        if not ('_E' in current_label) or current_label != expected_label:
                            if previous_label:
                                error_msg = 'Expecting label ending in \'_E\' since the last label was \'' + \
                                            previous_label + '\''
                            else:
                                error_msg = 'Expecting label ending in \'_E\''
                            error_message_func(line=current_row_number, label=current_label, error_message=error_msg,
                                               debugging=debugging, logger=temp_log_file_content)

                            wrong_begin_end_label = True

                        # need to get the label index based on the '_B' label in the motion_class.label list
                        tmp_var = list(current_label)
                        tmp_var[-1] = 'B'
                        new_label = ("".join(tmp_var)).upper()
                        # check if the motion exists
                        if not (new_label in motion_class.labels):
                            error_msg = 'Unknown label'
                            error_message_func(line=current_row_number, label=current_label,
                                               error_message=error_msg, debugging=debugging,
                                               logger=temp_log_file_content)

                            # keep record of new labels
                            new_label_list.append(current_label)

                            # print 'Finishing program. Unknown label.'
                            # always finish the program
                            # exit(1)

                        if not debugging:
                            current_class_label_index = motion_class.labels.index(current_label)

                            # provide the same labels to multiple time steps for hmm algorithm
                            label_range = current_timestep - last_timestep
                            label_list.extend([current_class_label_index] * label_range)

                    else:
                        error_msg = 'Error while changing start and end checks'
                        error_message_func(line=current_row_number, error_message=error_msg,
                                           debugging=debugging, label='', logger=temp_log_file_content)

                    # store label for future comparison
                    previous_label = current_label

                    # switch flags
                    start_flag = not start_flag
                    end_flag = not end_flag
                    first_pass = False

                    # check for multiple motions per label
                    label_used = [current_label for pMotions in motion_class.possible_motions
                                  if (pMotions in current_label)]
                    if len(label_used) > 1:
                        error_msg = 'Error: More than one motion in the same label'
                        error_message_func(line=current_row_number, label=current_label,
                                           error_message=error_msg, debugging=debugging,
                                           logger=temp_log_file_content)

                    # check for ignored labels/timesteps
                    if 'IGNORE_E' in current_label:
                        ignore_index_list.append([last_timestep, current_timestep])

                    last_timestep = current_timestep

                if not debugging and extract and tree2:

                    label_array = np.array(np.array(label_list))
                    # fetching sensors' data
                    extract_data_and_save_to_file(label_array, ignore_index_list, matlab_content['tree2'], motion_class,
                                                  file_properties, activity, index_matlab_file)

            if len(temp_log_file_content) != 0:

                if first_pass_ever:
                    file_properties.output_file_object.write('List of errors: \n\n')
                    first_pass_ever = False

                if first_pass_activity:
                    # print activity with error
                    file_properties.output_file_object.write('activity: {0}\n\n'.format(activity))
                    first_pass_activity = False

                # print the filename
                file_properties.output_file_object.write(matlab_file_name)
                file_properties.output_file_object.write('\n')
                # loop through the errors and print them
                for log_line in temp_log_file_content:
                    file_properties.output_file_object.write(log_line)

                file_properties.output_file_object.write('\n')

                if len(new_label_list) != 0:
                    file_properties.output_file_object.write('\tSet of new labels: \n\n')
                    for new_labels in new_label_list:
                        file_properties.output_file_object.write('\t' + new_labels + '\n')

                    file_properties.output_file_object.write('\n')

    file_properties.output_file_object.write('Done checking matlab files\n')
    print 'Done checking matlab files.'
    print ''


def remove_ignores(tmp_arr, ignored_index_list):

    # store indices to remove
    change = list()
    for iRange in ignored_index_list:
        change.extend(range(iRange[0], iRange[1]))

    # list without the ignored indices
    final_list = list()
    for rowIndex, rowData in enumerate(tmp_arr):
        if rowIndex not in change:
            final_list.append(rowData)
    return np.array(final_list)


def extract_data_and_save_to_file(labels, ignored_indices, dataset, motion_class, document_information, activities,
                                  index_matlab_file):

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

    # merge data with their respective labels
    tmp_arr = ''
    try:
        print '\tMerging data and labels arrays'
        tmp_arr = np.c_[data, labels]

    except ValueError:
        print 'size of data: {0}'.format(np.shape(data))
        print 'size of labels: {0}'.format(np.shape(labels))
        print '\n'
        exit(1)

    if len(ignored_indices) != 0:
        print '\tRemoving \'Ignore\' labels'
        data_labels = remove_ignores(tmp_arr, ignored_indices)
    else:
        data_labels = tmp_arr

    current_out_path = os.path.join(document_information.dataset_path,
                                    document_information.matlab_files_names_dict[activities][index_matlab_file])

    print '\tOutput file directory: {}'.format(current_out_path)
    np.save(current_out_path, data_labels)


def extract_information(doc, matlab_directory, action, matlab_filter, forward_folder):
    """
    Extracts the relevant information about the directories of the matlab files being considered
    updates two variables:
        doct.matlab_files_names_dict[activity]: key is the activity and the value is the matlab file's name
        doc.matlab_files_path_dict[activity]: key is the activity and the value is the matlab file's path
        :rtype: none
    """
    # current working path
    if matlab_directory == "":
        working_path = os.path.dirname(sys.argv[0])
        # including folder SensorData where all the matlab files are located
        doc.data_path = os.path.join(working_path, 'SensorData')
    else:
        doc.data_path = matlab_directory
        working_path = "/".join(matlab_directory.split('/')[:-1])

    # check data_path
    if not os.path.exists(doc.data_path):
        print "File " + doc.data_path + " does not exist"
        exit(1)

    if action == 'extract':
        if forward_folder == "":
            # location to store csv files
            doc.dataset_path = os.path.join(working_path, 'Dataset')
        else:
            doc.dataset_path = os.path.join(working_path, forward_folder)

        if not os.path.exists(doc.dataset_path):
            os.makedirs(doc.dataset_path)

    doc.log_file = os.path.join(working_path, 'Error_File')

    # all the activities i.e. Shelf_High_Heavycan, Shelf_Low_Heavycan, etc...
    doc.activity_list = next(os.walk(doc.data_path))[1]

    # loop through activities
    for activity in doc.activity_list:
        matlab_path_list = list()
        matlab_files_list = list()
        # obtain the path based on that activity
        activity_path = os.path.join(doc.data_path, activity)

        # loop through each subject of the current activity
        for (subject_path, _, matlab_file_list) in os.walk(activity_path, topdown=False):
            for matlab_file in matlab_file_list:
                # check if matlab files
                if '.mat' in matlab_file:
                    if matlab_filter == "":
                        # full matlab path
                        matlab_path_list.append(os.path.join(subject_path, matlab_file))
                        matlab_files_list.append(matlab_file)
                        doc.count += 1
                    elif matlab_filter in matlab_file:
                        # full matlab path
                        matlab_path_list.append(os.path.join(subject_path, matlab_file))
                        matlab_files_list.append(matlab_file)
                        doc.count += 1

        # add the respective matlab files to their specific activities
        doc.matlab_files_path_dict[activity] = matlab_path_list
        doc.matlab_files_names_dict[activity] = matlab_files_list
