from utils.logfileproperties import Document
from model.datafunctions import featurize
from model.workflow import imu_algorithm
from output import printout
import os


def extract_h5_information(doc='', h5_directory='', forward_folder='', script_path='', action=''):

    # current working path
    working_path = script_path
    doc.data_path = h5_directory

    # check data_path
    if not os.path.exists(doc.data_path):
        msg = "File " + doc.data_path + " does not exist"
        printout(message=msg, verbose=True)
        exit(1)

    if action == 'featurize':
        # forwarding directory
        doc.dataset_path = os.path.join(working_path, forward_folder)

        if not os.path.exists(doc.dataset_path):
            os.makedirs(doc.dataset_path)

        file_name = doc.dataset_path.split('/')[-1]
        doc.dataset_path_name = doc.dataset_path + '/' + file_name + '.hdf5'


def feature_extraction(h5_directory='', folder_name='', program_path='', action='',algorithm='', quickrun=''):

    file_info = Document()

    extract_h5_information(doc=file_info, h5_directory=h5_directory, forward_folder=folder_name,
                           script_path=program_path, action=action)

    if action == 'featurize':
        featurize(file_properties=file_info)

    elif action == 'imu':
        imu_algorithm(dataset_directory=file_info.data_path, algorithm=algorithm, quickrun=quickrun)


