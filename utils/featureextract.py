from utils.logfileproperties import Document
from model.datafunctions import featurize
from model.workflow import imu_algorithm
import os


def extract_h5_information(doc, h5_directory, program_path, action='', window_step_size=''):

    doc.input_path_filename = h5_directory
    doc.input_path = '/'.join(h5_directory.split('/')[:-1])
    temp_filename = h5_directory.split('/')[-1]
    doc.input_filename = temp_filename.replace('.hdf5', '')

    # check data_path
    if not os.path.exists(doc.input_path):
        msg = "File " + doc.input_path + " does not exist"
        raise IOError(msg)

    if action == 'featurize':

        folder_name = 'processed_dataset'
        doc.output_path = os.path.join(program_path, folder_name)

        if not os.path.exists(doc.output_path):
            os.mkdir(doc.output_path)

        doc.output_path_filename = doc.output_path + '/' + doc.input_filename + '_' + str(window_step_size[0]) + '_' \
            + str(window_step_size[1]) + '.hdf5'


def feature_extraction(h5_directory, action, logger, program_path, algorithm='', kmeans='', window_step_size='',
                       quickrun=True, batched_setting=False):

    file_info = Document()

    extract_h5_information(doc=file_info, h5_directory=h5_directory, action=action, program_path=program_path,
                           window_step_size=window_step_size)

    if action == 'featurize':
        featurize(file_properties=file_info, logger=logger, window_step_size=window_step_size)

    elif action == 'imu':
        imu_algorithm(doc=file_info, algorithm=algorithm, quickrun=quickrun, program_path=program_path, logger=logger,
                      kmeans=kmeans, batched_setting=batched_setting)

