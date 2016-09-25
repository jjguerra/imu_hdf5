"""
Created on Jun 13, 2016

@author: jjguerra
"""
from utils.output import printout
import os
import sys

from model.hmm import imu_hmm
from utils.matlabchecker import matlab_labels_data
from utils.matlabmover import move_matlab_files


# define the function blocks
def run_hmm():

    dataset_location = raw_input('Dataset location: ')

    temp_path = os.path.dirname(sys.argv[0])

    # get location of program
    if dataset_location == "":
        dataset_location = os.path.join(temp_path, 'Dataset')
    else:
        dataset_location = os.path.join(temp_path, dataset_location)

    printout(message='dataset directory:{0}'.format(dataset_location), verbose=True)
    printout(message="Running HMM Program", verbose=True)

    imu_hmm(dataset_directory=dataset_location)


def check_matlab():

    printout(message='Checking matlab files', verbose=True)

    checking_location = raw_input('Matlab file location: ')
    matlab_labels_data(action='check', matlab_directory=checking_location, s_property='', folder_name='')


def process_matlab():
    side = raw_input('arm side: ')
    side = side.upper()
    if side == 'L' or side == 'LEFT':
        specific_side = '_l_'
    elif side == 'R' or side == 'RIGHT':
        specific_side = '_r_'
    else:
        printout(message='No side specified. Please specified a side.', verbose=True)
        return

    dataset_folder_name = raw_input('Created folder\' name: ')

    printout(message='Converting matlab files', verbose=True)
    matlab_labels_data(action='extract', matlab_directory='', s_property=specific_side, folder_name=dataset_folder_name)


def move_matlab():

    printout(message='In order to move matlab files', verbose=True)

    initial_path = raw_input('Matlab directory: ')
    forwarding_path = raw_input('Matlab destination: ')

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
        print '5: Exit'

        print ''

        try:
            selected_option = int(raw_input('Select an option: '))

            if selected_option == 1:
                run_hmm()
            elif selected_option == 2:
                check_matlab()
            elif selected_option == 3:
                process_matlab()
            elif selected_option == 4:
                move_matlab()
            elif selected_option == 5:
                exit_program()
            else:
                printout(message='Wrong option selected.', verbose=True)

        except ValueError:
            printout(message='Invalid number.', verbose=True)


