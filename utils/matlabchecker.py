from utils.matlabfunctions import extract_mat_information, data_collection
from utils.logfileproperties import Document
from utils.output import printout


def matlab_labels_data(action='', matlab_directory='', pareticnonparetic='', folder_name='', error_file_name='logfile',
                       program_path='', leftright_arm=''):

    file_info = Document()
    extract_mat_information(doc=file_info, matlab_directory=matlab_directory, action=action, forward_folder=folder_name,
                            error_file_name=error_file_name, leftright_arm=leftright_arm, script_path=program_path,
                            pareticnonparetic=pareticnonparetic)

    if action == 'check':
        file_info.initialize_log_file()
        printout(message='running \'check\' on the files')
        data_collection(file_properties=file_info, debugging=True, extract=False)
        printout(message='Done checking matlab files.', verbose=True, extraspaces=1)

    elif action == 'extract':
        printout(message='running \'extract\' on the files')
        data_collection(file_properties=file_info, debugging=False, extract=True)
        printout(message='Done extracting matlab files.', verbose=True, extraspaces=1)

    else:
        printout(message='Wrong actioned passed.', verbose=True)
