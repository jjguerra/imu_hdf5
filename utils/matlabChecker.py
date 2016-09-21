from time import gmtime, strftime
from utils.matlabFunctions import extract_information, data_collection


def initialize_log_file(doc):
    """
    :return: opens and writes to the log file
    """
    doc.output_file_object = open(doc.log_file, "w")
    doc.output_file_object.write('Matlab File Information\n\n\n')

    doc.output_file_object.write('Date:{0}\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    doc.output_file_object.write('Number of Matlab Files:{0} \n\n\n'.format(doc.count))
    for activity, matlab_file_name_list in doc.matlab_files_names_dict.iteritems():
        doc.output_file_object.write('activity: {0}\n\n'.format(activity))
        for matlab_filename in matlab_file_name_list:
            doc.output_file_object.write('\t{0}\n'.format(matlab_filename))
        doc.output_file_object.write('\n')


class Document(object):

    def __init__(self):
        self.data_path = ''  # matlab files location
        self.dataset_path = ''  # converted matlab files location
        self.activity_list = list()
        self.matlab_files_path_dict = dict()
        self.output_file_object = ''
        self.matlab_files_names_dict = dict()
        self.log_file = ''
        self.count = 0


def matlab_labels_data(action, matlab_directory):

    file_info = Document()
    extract_information(doc=file_info, matlab_directory=matlab_directory, action=action)
    initialize_log_file(file_info)
    if action == 'check':
        data_collection(file_properties=file_info, debugging=True, extract=False)
        # print 'statistic file: {}'.format(document_descriptions.moved_log_file)
    if action == 'extract':
        data_collection(file_properties=file_info, debugging=False, extract=True)
