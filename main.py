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


def forwarding_folder(action):
    output_error = True
    while output_error:
        # get the folder where all the processed matlab files will be stored
        dataset_folder_name = raw_input('Output folder\'s name: ')

        if dataset_folder_name == '' and action == 'extract':
            file_path = os.path.join(program_path, 'converted_dataset')
        elif dataset_folder_name == '' and action == 'featurize':
            file_path = os.path.join(program_path, 'processed_dataset')
        else:
            if os.path.isdir(dataset_folder_name):
                file_path = dataset_folder_name
            else:
                file_path = os.path.join(program_path, dataset_folder_name)
        try:
            # if does not exists, create it
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        except OSError:
            msg = 'Error. {0} directory cannot be created.'.format(file_path)
            printout(message=msg, verbose=True)
            msg = 'Chose a different forwarding directory.'
            printout(message=msg, verbose=True)

        else:
            output_error = False

    return file_path


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


# make sure all the directory information is correct
def check_location(matlab_or_dataset, default_folder):

    not_correct_dataset_location = True
    while not_correct_dataset_location:

        dataset_location = raw_input(
            matlab_or_dataset + ' directory/folder (default folder {0}): '.format(default_folder))

        # get location of program
        if dataset_location == "":
            if os.path.isfile(default_folder):
                file_path = os.path.join(program_path, default_folder)
            else:
                # use for the dropbox option
                file_path = default_folder
        else:
            if os.path.isdir(dataset_location):
                file_path = dataset_location
            elif os.path.isfile(dataset_location):
                file_path = os.path.join(program_path, dataset_location)
            else:
                msg = 'Error. input is not a folder or a directory.'
                printout(message=msg, verbose=True)
                msg = 'Please chose a correct folder or directory.'
                printout(message=msg, verbose=True)
                continue

        if not os.path.exists(file_path):
            msg = 'Error. {0} directory {1} does not exists.'.format(matlab_or_dataset, file_path)
            printout(message=msg, verbose=True)
            msg = 'Please chose a different dataset directory.'
            printout(message=msg, verbose=True)
        else:
            # right dataset directory was provided
            not_correct_dataset_location = False

    return file_path


# get the dataset directory and the quickrun option
def select_dataset_quickrun():

    # get the right dataset location
    file_path = check_location(matlab_or_dataset='Dataset', default_folder='sensordata')

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

    return file_path, quickrun


# run specific ML model
def ml_algorithm(algorithm=''):

    # get dataset directory
    dataset_location, quickrun = select_dataset_quickrun()

    logging.getLogger('regular.time.line').info('Running {0} Model'.format(algorithm))

    feature_extraction(h5_directory=dataset_location, algorithm=algorithm, quickrun=quickrun, action='imu',
                       program_path=program_path, logger=logging)


# go through all the matlab files and make sure there are not data or labels mistakes
def check_matlab():

    # get the right dataset directory to check matlab files
    checking_location = check_location(matlab_or_dataset='Matlab', default_folder='sensordata')

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

    # get the right dataset location
    dataset_location = check_location(matlab_or_dataset='Matlab', default_folder='sensordata')
    # get output folder
    file_path = forwarding_folder(action=action)

    if action == 'extract':
        leftright_arm, specific_side = forwarding_filters()
        msg = 'starting extracting matlab files'
        logging.getLogger('regular.time').info(msg)
        matlab_labels_data(action=action, matlab_directory=dataset_location, program_path=program_path,
                           leftright_arm=leftright_arm, pareticnonparetic=specific_side, folder_name=file_path)
        msg = 'finished extracting matlab files'
        logging.getLogger('regular.time').info(msg)

    else:
        msg = 'starting \'featurization\' of the files'
        logging.getLogger('regular.time').info(msg)
        feature_extraction(h5_directory=dataset_location, folder_name=file_path, program_path=program_path,
                           action='featurize')
        msg = 'finished \'featurization\' of the files'
        logging.getLogger('regular.time').info(msg)


# moves and organizes matlab files based on activity and then on users
def move_matlab():

    # default option is the dropbox directory
    initial_path = check_location(matlab_or_dataset='Matlab', default_folder='/Users/jguerra/Dropbox/SensorJorge')

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
    while selected_option != 5:

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
