"""
Created on Jun 13, 2016

@author: jjguerra
@version:0.2
"""
import os
from utils.matlabchecker import matlab_labels_data
from utils.matlabmover import move_matlab_files
from utils.featureextract import feature_extraction
import logging
import argparse
from utils.logger import logger_initialization
from utils.output import printout


# imu project path
program_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def window_step_properties():

    window_size_errors = True

    while window_size_errors:
        window_size = raw_input('window size: ')

        # default
        if window_size == '':
            window_size_errors = False
            window_size = 60

        try:
            window_size = int(window_size)
            if window_size == 30 or window_size == 60 or window_size == 120:
                window_size_errors = False
            else:
                printout('Wrong number of window size. Options:30,60 or 120')
        except ValueError as err_message:
            printout(message=err_message)

    step_size_errors = True
    while step_size_errors:
        step_size = raw_input('step size: ').upper()

        if step_size == '':
            step_size_errors = False
            step_size = 1

        try:
            step_size = int(step_size)
            if step_size > 0:
                step_size_errors = False
            # default

            else:
                printout('Wrong number of step size. Select step size greater than 0')
        except ValueError as err_message:
            printout(message=err_message)

    window_step_size = [window_size, step_size]

    return window_step_size


def forwarding_filters():

    right_left_error = True
    while right_left_error:
        leftright_arm = raw_input('right(r) or left(l): ').upper()
        if leftright_arm == 'L' or leftright_arm == 'LEFT':
            leftright_arm = '_l_'
        elif leftright_arm == 'R' or leftright_arm == 'RIGHT':
            leftright_arm = '_r_'

        else:
            msg = 'No side specified. Please specified a side.'
            printout(message=msg, verbose=True)
            continue

        right_left_error = False

    paretic_nonparetic_errors = True
    while paretic_nonparetic_errors:
        specific_side = ''
        paretic_nonparetic_enter = raw_input('paretic(p), non-paretic(n) or neither(\'enter\'): ').upper()

        if paretic_nonparetic_enter == 'P' or paretic_nonparetic_enter == 'PARETIC':
            pareticnonparetic = 'paretic'

        elif paretic_nonparetic_enter == 'N' or paretic_nonparetic_enter == 'NONPARETIC':
            pareticnonparetic = 'nonparetic'

        elif paretic_nonparetic_enter == '':
            pareticnonparetic = ''

        else:
            msg = 'Wrong option selected.'
            printout(message=msg, verbose=True)
            continue

        paretic_nonparetic_errors = False

    active_nonactive_errors = True
    while active_nonactive_errors:
        if pareticnonparetic == 'paretic':
            activenonactive = raw_input('active(a) or non-active(n): ').upper()
            if activenonactive == 'A' or activenonactive == 'ACTIVE':
                specific_side = '_paretic_active_'
            elif activenonactive == 'N' or activenonactive == 'NONACTIVE':
                specific_side = '_paretic_nonactive_'
            else:
                msg = 'Wrong option selected.'
                printout(message=msg, verbose=True)
                continue

        elif pareticnonparetic == 'nonparetic':
            activenonactive = raw_input('active or non-active: ').upper()
            if activenonactive == 'A' or activenonactive == 'ACTIVE':
                specific_side = '_nonparetic_active_'
            elif activenonactive == 'N' or activenonactive == 'NONACTIVE':
                specific_side = '_nonparetic_nonactive_'
            else:
                msg = 'Wrong option selected.'
                printout(message=msg, verbose=True)
                continue

        active_nonactive_errors = False

    return leftright_arm, specific_side


def get_set_dataset_location(matlab_or_dataset, default_folder='', action=''):
    """
    Gets the filename of the datasets or the directory of the matlab files
    :return:
        if dataset: filename
        if matlab: directory
    """

    not_correct_dataset_location = True
    while not_correct_dataset_location:

        if matlab_or_dataset == 'matlab':
            file_directory = raw_input('Matlab directory: ')

            if not os.path.isdir(file_directory) or not os.path.exists(file_directory):
                msg = 'Error. Wrong directory provided. Please, provide correct directory.'
                printout(message=msg, verbose=True)
                continue

            else:
                file_path = file_directory
                not_correct_dataset_location = False

        else:
            if action == 'get':
                filename = raw_input('Input dataset filename: ')
            else:
                filename = raw_input('Output dataset filename: ')

            # get location of program
            if filename == "":
                msg = 'Error. No filename provided. Please, insert filename.'
                printout(message=msg, verbose=True)
                continue
            else:
                if default_folder:

                    folder_filename = os.path.join(default_folder, filename + '.hdf5')
                    file_path = os.path.join(program_path, folder_filename)
                    # right dataset directory was provided
                    not_correct_dataset_location = False

    return file_path


def select_dataset_quickrun(algorithm=''):
    """
    get the dataset directory and the quickrun option
    @:return
        if *HMM: filename, quickrun option and kmeans option
        else: filename, quickrun option
    """
    # get the right dataset location
    file_path = get_set_dataset_location(matlab_or_dataset='dataset', default_folder='processed_dataset', action='get')

    quickrun = ''
    while quickrun == '':

        # quickrun options is to run only the short version of the selected algorithm
        # rather than the algorithm with multiple parameters
        quickrun_selection = raw_input('Quickrun: ').upper()
        print ''

        # get location of program
        if quickrun_selection == "":
            quickrun = True
        elif quickrun_selection == 'TRUE' or quickrun_selection == 'T':
            quickrun = True
        elif quickrun_selection == 'FALSE' or quickrun_selection == 'F':
            quickrun = False
        else:
            msg = 'Error. Wrong option for quickrun selected.'
            printout(message=msg, verbose=True)

    if algorithm == 'GHMM' or algorithm == 'GMMHMM':
        kmeans = raw_input('kmeans (regular or mini): ').upper()

        if kmeans == '':
            kmeans = 'REGULAR'

        return file_path, quickrun, kmeans

    else:
        return file_path, quickrun


# run specific ML model
def ml_algorithm(algorithm=''):

    # get dataset directory
    dataset_location, quickrun, kmeans = select_dataset_quickrun(algorithm)

    logging.getLogger('regular.time.line').info('Running {0} Model'.format(algorithm))

    feature_extraction(h5_directory=dataset_location, algorithm=algorithm, quickrun=quickrun, action='imu',
                       program_path=program_path, logger=logging, kmeans=kmeans)


# go through all the matlab files and make sure there are not data or labels mistakes
def check_matlab():

    # get the right dataset directory to check matlab files
    checking_location = get_set_dataset_location(matlab_or_dataset='matlab')

    # name of the log file
    error_file_name = raw_input('Error log file name: ')

    # get location of program
    if error_file_name == "":
        # get the name in the path
        error_log_name = checking_location.split('\\')[-1]
        file_name = 'Error_File_' + error_log_name + '.txt'
    else:
        file_name = error_file_name + '.txt'

    msg = 'starting checking matlab files'
    logging.getLogger('regular.time').info(msg)

    matlab_labels_data(action='check', matlab_directory=checking_location, error_file_name=file_name)

    msg = 'finished checking matlab files'
    logging.getLogger('regular.time').info(msg)


# convert the matlab files to .npy files so they can be used by the algorithm efficiently
# or obtain statistical descriptors of the time series dataset
def convert_featurize_matlab(action):

    if action == 'extract':
        # get the right dataset location
        dataset_location = get_set_dataset_location(matlab_or_dataset='matlab')
        leftright_arm, specific_side = forwarding_filters()

        # get output folder
        file_path = get_set_dataset_location(matlab_or_dataset='dataset', default_folder='converted_dataset',
                                             action='set')
        msg = 'starting extracting matlab files'
        logging.getLogger('regular.time').info(msg)
        matlab_labels_data(action=action, matlab_directory=dataset_location, program_path=program_path,
                           leftright_arm=leftright_arm, pareticnonparetic=specific_side, folder_name=file_path)
        msg = 'finished extracting matlab files'
        logging.getLogger('regular.time').info(msg)

    else:
        # get the right dataset location
        dataset_location = get_set_dataset_location(matlab_or_dataset='dataset', default_folder='converted_dataset',
                                                    action='get')

        window_step_size = window_step_properties()
        msg = 'starting \'featurization\' of the files'
        logging.getLogger('regular.time').info(msg)
        feature_extraction(h5_directory=dataset_location, action='featurize', window_step_size=window_step_size,
                           logger=logging, program_path=program_path)
        msg = 'finished \'featurization\' of the files'
        logging.getLogger('regular.time').info(msg)


# moves and organizes matlab files based on activity and then on users
def move_matlab():

    # default option is the dropbox directory
    initial_path = get_set_dataset_location(matlab_or_dataset='matlab',
                                            default_folder='/Users/jguerra/Dropbox/SensorJorge')

    forwarding_error = True
    while forwarding_error:

        forwarding_path = raw_input('Matlab forwarding directory: ')
        print forwarding_path

        if forwarding_path == "":
            # get location of program
            final_path = os.path.join(program_path, 'sensordata')
            print final_path

        else:
            if os.path.isdir(forwarding_path):
                final_path = forwarding_path
            elif os.path.isfile(forwarding_path):
                final_path = os.path.join(program_path, forwarding_path)

        try:
            # if does not exists, create it
            if not os.path.exists(final_path):
                os.makedirs(final_path)
        except OSError:
            msg = 'Error. {0} directory cannot be created.'.format(final_path)
            printout(message=msg, verbose=True)
            msg = 'Please chose a different forwarding directory.'
            printout(message=msg, verbose=True)
        else:
            forwarding_error = False

    msg = 'starting moving matlab files from {0} to {1}'.format(initial_path, final_path)
    logging.getLogger('').info(msg)

    move_matlab_files(initial_path, forwarding_path)
    msg = 'finished moving matlab files from {0} to {1}'.format(initial_path, final_path)
    logging.getLogger('').info(msg)


# close
def exit_program():

    msg = 'Program terminated.\n'
    logging.getLogger('').info(msg)
    exit(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'],
                        help="Set the logging level")

    logger_initialization(parser=parser)

    selected_option = -1
    while selected_option != 9:

        # options
        print ''
        print 'Program Menu:\n'
        print '1: Gaussian-HMM program'
        print '2: GMM-HMM program'
        print '3: Check Matlab files'
        print '4: Convert Matlab files to hdf5 format file'
        print '5: Process matlab files with basic features'
        print '6: Move Matlab files from Dropbox to Working Directory'
        print '7: Perform Logistic Regression'
        print '8: Perform LSTM'
        print '9: Exit'
        print ''

        try:
            selected_option = int(raw_input('Select an option: '))

            if selected_option == 1:
                ml_algorithm('GHMM')
            if selected_option == 2:
                ml_algorithm('GMMHMM')
            elif selected_option == 3:
                check_matlab()
            elif selected_option == 4:
                convert_featurize_matlab('extract')
            elif selected_option == 5:
                convert_featurize_matlab('featurize')
            elif selected_option == 6:
                move_matlab()
            elif selected_option == 7:
                ml_algorithm('Logistic Regression')
            elif selected_option == 8:
                ml_algorithm('LSTM')
            elif selected_option == 9:
                exit_program()
            else:
                printout(message='Wrong option selected.', verbose=True)

        except ValueError as error_message:
            logging.getLogger('regular').error(error_message)
