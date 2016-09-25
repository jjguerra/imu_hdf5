from utils.matlabfunctions import extract_information, data_collection
from utils.logfileproperties import Document


def matlab_labels_data(action, matlab_directory, s_property, folder_name):

    file_info = Document()
    extract_information(doc=file_info, matlab_directory=matlab_directory, action=action, matlab_filter=s_property,
                        forward_folder=folder_name)
    file_info.initialize_log_file()

    if action == 'check':
        data_collection(file_properties=file_info, debugging=True, extract=False)
        # print 'statistic file: {}'.format(document_descriptions.moved_log_file)
    if action == 'extract':
        data_collection(file_properties=file_info, debugging=False, extract=True)
