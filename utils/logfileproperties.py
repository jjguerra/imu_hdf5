from time import gmtime, strftime


class Document(object):

    def __init__(self):
        self.data_path = ''  # matlab or hdf5 files location
        self.dataset_path_name = ''  # hdf5 (before processing) or hdf5 (after processing) files final location
        self.dataset_path = ''  # converted matlab files location
        self.activity_list = list()
        self.matlab_files_path_dict = dict()
        self.output_file_object = ''
        self.matlab_files_names_dict = dict()
        self.log_file = ''
        self.count = 0

    def initialize_log_file(self):
        """
        :return: opens and writes to the log file
        """
        self.output_file_object = open(self.log_file, "w")
        self.output_file_object.write('Matlab File Information\n\n\n')

        self.output_file_object.write('Date:{0}\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        self.output_file_object.write('Number of Matlab Files:{0} \n\n\n'.format(self.count))
        for activity, matlab_file_name_list in self.matlab_files_names_dict.iteritems():
            self.output_file_object.write('activity: {0}\n\n'.format(activity))
            for matlab_filename in matlab_file_name_list:
                self.output_file_object.write('\t{0}\n'.format(matlab_filename))

            self.output_file_object.write('\n')
