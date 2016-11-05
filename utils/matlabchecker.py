from utils.matlabfunctions import extract_mat_information, data_collection
from utils.logfileproperties import Document
from utils.output import printout


def matlab_labels_data(action, leftright_arm, program_path, matlab_directory, logger, pareticnonparetic='',
                       output_path_filename=''):
    """
    either check or extract the user dataset information from the .mat files
    :param action: (str) check or extract
    :param leftright_arm: (str) left or right
    :param pareticnonparetic: (str) paretic or non paretic
    :param program_path: (str) location of the program path
    :param matlab_directory: (str)
    :param logger: (logging object)
    :param output_path_filename: optional (str) path to save the extracted data files
    :return:
    """

    file_info = Document()
    extract_mat_information(doc=file_info, matlab_directory=matlab_directory, action=action, script_path=program_path,
                            output_path_filename=output_path_filename, leftright_arm=leftright_arm,
                            pareticnonparetic=pareticnonparetic)

    if action == 'check':
        file_info.initialize_log_file()
        # running \'check\' on the files
        data_collection(file_properties=file_info, debugging=True, extract=False, logger=logger)

    elif action == 'extract':
        # running \'extract\' on the files
        data_collection(file_properties=file_info, debugging=False, extract=True, logger=logger)

    else:
        msg = 'Wrong action passed. Passed action {0} possible actions: check or extract'.format(action)
        raise ValueError(msg)
