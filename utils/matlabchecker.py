from utils.matlabfunctions import extract_information, data_collection
from utils.logfileproperties import Document
from utils.output import printout


def matlab_labels_data(action='', matlab_directory='', pareticnonparetic='', folder_name='', error_file_name='logfile',
                       program_path='', leftright_arm=''):

    file_info = Document()
    extract_information(doc=file_info, matlab_directory=matlab_directory, action=action, forward_folder=folder_name,
                        error_file_name=error_file_name, leftright_arm=leftright_arm,
                        pareticnonparetic=pareticnonparetic, script_path=program_path)
    file_info.initialize_log_file()

    if action == 'check':
        printout(message='running \'check\' on the files')
        data_collection(file_properties=file_info, debugging=True, extract=False)
        # print 'statistic file: {}'.format(document_descriptions.moved_log_file)
    if action == 'extract':
        printout(message='running \'extract\' on the files')
        data_collection(file_properties=file_info, debugging=False, extract=True)
