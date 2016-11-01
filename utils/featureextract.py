from utils.logfileproperties import Document
from model.datafunctions import featurize
from model.workflow import imu_algorithm
from output import printout
import os


def extract_h5_information(doc, h5_directory, program_path, action='', window_step_size=''):

    # get the path of the file
    doc.data_path = '/'.join(h5_directory.split('/')[:-1])

    # check data_path
    if not os.path.exists(doc.data_path):
        msg = "File " + doc.data_path + " does not exist"
        printout(message=msg, verbose=True)
        exit(1)

    if action == 'featurize':

        folder_name = 'processed_dataset'
        doc.dataset_path = os.path.join(program_path, folder_name)

        if not os.path.exists(doc.dataset_path):
            os.mkdir(doc.dataset_path)

        doc.dataset_path_name = doc.dataset_path + '/' + folder_name + '_' + str(window_step_size[0]) + '_' + str(
            window_step_size[1]) + '.hdf5'

    else:
        doc.dataset_path_name = h5_directory


def feature_extraction(h5_directory, action, logger, program_path, algorithm='', kmeans='', window_step_size='',
                       quickrun=True, batched_setting=False):

    file_info = Document()

    extract_h5_information(doc=file_info, h5_directory=h5_directory, action=action, program_path=program_path,
                           window_step_size=window_step_size)

    if action == 'featurize':
        featurize(file_properties=file_info, logger=logger, window_step_size=window_step_size)

    elif action == 'imu':
        imu_algorithm(dataset_path_name=file_info.dataset_path_name, algorithm=algorithm, quickrun=quickrun,
                      program_path=program_path, logger=logger, kmeans=kmeans, batched_setting=batched_setting)

