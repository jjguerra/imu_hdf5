import os
import re


def move_matlab_files(initial_path, forwarding_path):
    """
    move matlab files from a directory to another folder with the
    activity, user, file name structure
    :return: 1 if files where moved correctly, 0 otherwise
    """

    folder_dictionary = {
        'feeding': 'Feeding',
        'radialcan': 'Radial_Lightcan',
        'radialtp': 'Radial_TP',
        'book_high': 'Shelf_High_Book',
        'detergent_high': 'Shelf_High_Detergent',
        'heavycan_high': 'Shelf_High_Heavycan',
        'lightcan_high': 'Shelf_High_Lightcan',
        'tp_high': 'Shelf_High_TP',
        'book_low': 'Shelf_Low_Book',
        'detergent_low': 'Shelf_Low_Detergent',
        'heavycan_low': 'Shelf_Low_Heavycan',
        'lightcan_low': 'Shelf_Low_Lightcan',
        'tp_low': 'Shelf_Low_TP'}

    # check if forwarding path exists and if not, create it
    if not os.path.exists(forwarding_path):
        os.makedirs(forwarding_path)

    # files inside the given folder
    matlab_files = next(os.walk(initial_path))[2]

    for current_matlab_file in matlab_files:
        if '.mat' in current_matlab_file:
            print 'processing file: {}'.format(current_matlab_file)
            old_path = os.path.join(initial_path, current_matlab_file)
            if 'paretic' in current_matlab_file:
                expression_pattern = \
                    r'(^[A-Z]+[0-9]+)_[nonparetic|paretic]+_[active|nonactive]+_([a-z]+_high|[a-z]+_low|[a-z]+).*\.mat$'
            else:
                expression_pattern = r'(^[A-Z]+[0-9]+)_pilot_OT_[l|r]_([a-z]+_high|[a-z]+_low|[a-z]+).*\.mat$'
            # string_information = extracted useful information of the string
            # i.e.
            #    string = 'N537_pilot_OT_l_book_high.mvnx.mat'
            #    string_information.group(0) = 'N537_pilot_OT_l_book_high.mvnx.mat'
            #    user: string_information.group(1) = 'N537'
            #    left|right: string_information.group(2) = 'l'
            #    activity: string_information.group(3) = 'book_high'
            string_information = re.match(pattern=expression_pattern, string=current_matlab_file)
            if string_information.group(1) != "" and string_information.group(2) != "":
                final_activity_path = os.path.join(forwarding_path, folder_dictionary[string_information.group(2)])
                final_user_path = os.path.join(final_activity_path, string_information.group(1))

                if not os.path.exists(final_activity_path):
                    os.makedirs(final_activity_path)
                if not os.path.exists(final_user_path):
                    os.makedirs(final_user_path)

                new_matlab_path = os.path.join(final_user_path, current_matlab_file)

                if os.path.exists(new_matlab_path):
                    os.remove(new_matlab_path)
                os.rename(old_path, new_matlab_path)
            else:
                print 'Fatal error moving ' + current_matlab_file
                exit(1)

    print 'All matlab files were moved.\n\n'
