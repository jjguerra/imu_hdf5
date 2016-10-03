"""
Created on Jun 13, 2016

@author: jjguerra
"""
from utils.output import printout
import os
import sys

from model.workflow import imu_algorithm
from utils.matlabchecker import matlab_labels_data
from utils.matlabmover import move_matlab_files


def select_dataset_quickrun(algorithm=''):

    dataset_location = raw_input('Dataset directory: ')

    temp_path = os.path.dirname(sys.argv[0])

    # get location of program
    if dataset_location == "":
        dataset_location = os.path.join(temp_path, 'Dataset')
    else:
        dataset_location = os.path.join(temp_path, dataset_location)

    quickrun_selection = raw_input('Quickrun: ')

    quickrun = ''
    while quickrun == '':
        # get location of program
        if quickrun_selection == "":
            quickrun = True
        else:
            quickrun = quickrun_selection

    msg = 'Running {0} Models'.format(algorithm)
    printout(message=msg, verbose=True)

    return dataset_location, quickrun


# run model
def ml_algorithm(algorithm=''):

    # get dataset directory
    dataset_location, quickrun = select_dataset_quickrun(algorithm)
    imu_algorithm(dataset_directory=dataset_location, algorithm=algorithm, quickrun=quickrun)


def check_matlab():

    printout(message='Checking matlab files', verbose=True)

    checking_location = raw_input('Matlab file folder: ')
    error_file_name = raw_input('Error file name: ')

    # get location of program
    if error_file_name == "":
        file_name = 'Error_File_' + checking_location + '.txt'
    else:
        file_name = error_file_name + '.txt'

    matlab_labels_data(action='check', matlab_directory=checking_location, error_file_name=file_name)


def process_matlab():

    dataset_location = raw_input('Dataset folder: ')

    directory = False
    if '/' in dataset_location:
        directory = True

    program_path = os.path.dirname(sys.argv[0])

    # get location of program
    if dataset_location == "":
        dataset_location = os.path.join(program_path, 'SensorData')
    else:
        if not directory:
            dataset_location = os.path.join(program_path, dataset_location)

    leftright_arm = raw_input('right(r) or left(l): ').upper()
    if leftright_arm == 'L':
        leftright_arm = '_l_'
    elif leftright_arm == 'R':
        leftright_arm = '_r_'

    else:
        printout(message='No side specified. Please specified a side.', verbose=True)
        return

    specific_side = ''
    paretic_nonparetic_enter = raw_input('paretic(p), non-paretic(n) or neither(\'enter\'): ').upper()

    if paretic_nonparetic_enter == 'P':
        pareticnonparetic = 'paretic'

    elif paretic_nonparetic_enter == 'N':
        pareticnonparetic = 'nonparetic'

    else:
        pareticnonparetic = ''

    if pareticnonparetic == 'paretic':
        activenonactive = raw_input('active or non-active: ').upper()
        if activenonactive == 'A' or activenonactive == 'ACTIVE':
            specific_side = '_paretic_active_'
        elif activenonactive == 'N' or activenonactive == 'NONACTIVE':
            specific_side = '_paretic_nonactive_'

    elif pareticnonparetic == 'nonparetic':
        activenonactive = raw_input('active or non-active: ').upper()
        if activenonactive == 'A' or activenonactive == 'ACTIVE':
            specific_side = '_nonparetic_active_'
        elif activenonactive == 'N' or activenonactive == 'NONACTIVE':
            specific_side = '_nonparetic_nonactive_'

    dataset_folder_name = raw_input('Output folder\'s name: ')

    printout(message='Converting matlab files', verbose=True)
    matlab_labels_data(action='extract', matlab_directory=dataset_location, leftright_arm=leftright_arm,
                       pareticnonparetic=specific_side, folder_name=dataset_folder_name, program_path=program_path)


def move_matlab():

    printout(message='In order to move matlab files', verbose=True)

    initial_path = raw_input('Matlab current directory: ')
    forwarding_path = raw_input('Matlab forwarding directory: ')

    if forwarding_path == "":
        # get location of program
        temp_path = os.path.dirname(sys.argv[0])
        forwarding_path = os.path.join(temp_path, 'SensorData')

    move_matlab_files(initial_path, forwarding_path)


def exit_program():

    printout(message='Program terminated.\n', verbose=True)
    exit(0)


if __name__ == '__main__':

    selected_option = -1
    while selected_option != 5:

        # options
        print 'Program Menu:\n'
        print '1: HMM program'
        print '2: Check Matlab files'
        print '3: Convert Matlab files to HMM format file'
        print '4: Move Matlab files from Dropbox to Working Directory'
        print '5: Perform Logistic Regression'
        print '6: Exit'
        print ''

        try:
            selected_option = int(raw_input('Select an option: '))

            if selected_option == 1:
                ml_algorithm('HMM')
            elif selected_option == 2:
                check_matlab()
            elif selected_option == 3:
                process_matlab()
            elif selected_option == 4:
                move_matlab()
            elif selected_option == 5:
                ml_algorithm('Logistic Regression')
            elif selected_option == 6:
                exit_program()
            else:
                printout(message='Wrong option selected.', verbose=True)

        except ValueError:
            printout(message='Invalid number.', verbose=True)


