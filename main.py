"""
Created on Jun 13, 2016

@author: jjguerra
"""

import os
import sys

from hmm import imu_hmm
from utils.matlabchecker import matlab_labels_data
from utils.matlabmover import move_matlab_files


# define the function blocks
def run_hmm():

    print "Running HMM Program"
    dataset_location = raw_input('Dataset location: ')

    if dataset_location == "":
        # get location of program
        temp_path = os.path.dirname(sys.argv[0])
        dataset_location = os.path.join(temp_path, 'Dataset')
    print 'dataset directory: '.format(dataset_location)

    imu_hmm(dataset_directory=dataset_location)


def check_matlab():
    print "Checking matlab files"
    checking_location = raw_input('Matlab file location: ')
    matlab_labels_data(action='check', matlab_directory=checking_location, s_property='')


def process_matlab():
    side = raw_input('arm side: ')
    side = side.upper()
    if side == 'L' or side == 'LEFT':
        specific_side = '_l_'
    elif side == 'R' or side == 'RIGHT':
        specific_side = '_r_'
    else:
        print 'No side specified. Please specified a side.'
        return

    print "Converting matlab files"
    matlab_labels_data(action='extract', matlab_directory='', s_property=specific_side)


def move_matlab():
    print "In order to move matlab files"
    initial_path = raw_input('Matlab directory: ')
    forwarding_path = raw_input('Matlab destination: ')

    if forwarding_path == "":
        # get location of program
        temp_path = os.path.dirname(sys.argv[0])
        forwarding_path = os.path.join(temp_path, 'SensorData')

    move_matlab_files(initial_path, forwarding_path)


def exit_program():
    print 'Program terminated.'
    print ''
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
                print 'Wrong option selected.'

        except ValueError:
            print("Invalid number")


