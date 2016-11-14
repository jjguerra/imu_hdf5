import re
import numpy as np
import xlsxwriter
import math


def file_properties(file_name=''):
    expression_pattern = r'(^[A-Z]+[0-9]+)_pilot_OT_([l|r])_(feeding|radialcan|radialtp|book_low|book_high|' \
                         r'detergent_high|detergent_low|heavycan_high|heavycan_low|lightcan_high|lightcan_low|' \
                         r'tp_high|tp_low)(_(t\d{1})|_(t\d-t\d)|.*).mvnx.mat'

    # string_information = extracted useful information of the string
    # i.e.
    #    string = 'N537_pilot_OT_l_book_high.mvnx.mat'
    #    string_information.group(0) = 'N537_pilot_OT_l_book_high_t1.mvnx.mat'
    #    user: string_information.group(1) = 'N537'
    #    left|right: string_information.group(2) = 'l'
    #    activity: string_information.group(3) = 'book_high'
    #    time: string_information.group(5) = 't1'
    s_information = re.match(pattern=expression_pattern, string=file_name)
    usr = s_information.group(1)
    rightleft = s_information.group(2)
    act = s_information.group(3)
    # if this is empty then there is not time variable in the filename
    flag = s_information.group(4)
    if flag != '':
        if '-' in flag:
            time = s_information.group(6)
        else:
            time = s_information.group(5)
    else:
        time = ''

    return usr, rightleft, act, time


def obtain_data(file_object, spec_activity):

    compact_list = {'REST': list(), 'REA': list(), 'T': list(), 'RET': list(), 'I': list(), 'S': list(),
                    'ST': list(), 'SM': list(), 'M': list(), 'MC': list(), 'THM': list(), 'TC': list()}

    # use to look at the right activity
    activity_flag = False
    # check
    motion_flag = False

    for line in file_object:
        line = line.replace('\n', '').replace('\t', '')
        if activity_flag and motion_flag:
            if 'sensitivity' in line:
                sensitivity = float(line.split('=')[1])
            elif 'specificity' in line:
                specificity = float(line.split('=')[1])
            elif 'positive predicted value' in line:
                ppv = float(line.split('=')[1])
            elif 'negative predictive value' in line:
                npv = float(line.split('=')[1])

                motion_flag = False

                if len(compact_list[motion]) != 0:
                    compact_list[motion][0].append(sensitivity)
                    compact_list[motion][1].append(specificity)
                    compact_list[motion][2].append(ppv)
                    compact_list[motion][3].append(npv)

                else:
                    compact_list[motion] = [[sensitivity], [specificity], [ppv], [npv]]
                    motion_flag = False

        elif activity_flag and ('motion=' in line):
            motion = line.split('=')[1]
            motion_flag = True

            # reset
            sensitivity = 0
            specificity = 0
            ppv = 0
            npv = 0

        elif 'Starting analysing' in line:
            file_name = line.split(' ')[-1]
            _, _, current_user_activity, _ = file_properties(file_name=file_name)
            if spec_activity in current_user_activity:
                activity_flag = True
            else:
                activity_flag = False

    return compact_list


def run(csv_filename='', workbook=''):

    window = csv_filename.split('_')[-2]
    n_states = csv_filename.split('_')[-1]

    worksheet_name = str(window) + '-' + str(n_states)
    worksheet = workbook.add_worksheet(worksheet_name)
    header = 'window={0} Number of states={1}'.format(window, n_states)
    worksheet.write(1, 0, header)

    start_activity_row = 2
    start_activity_col = 1

    activity_type = ['heavycan', 'lightcan', 'book', 'tp', 'detergent', 'radialcan', 'radialtp']

    # keep count of the number of activities within a row
    column_activity = 0

    total_number_row = 0
    for activity_index, activity in enumerate(activity_type):

        # max of 3 activities per row
        if (activity_index != 0) and column_activity < 3:
            start_activity_col += 7
        elif column_activity == 3:
            column_activity = 0
            start_activity_row = total_number_row + 2
            start_activity_col = 1

        log_file = open(name=csv_filename + '.csv', mode='r')
        motions_statistics = obtain_data(log_file, activity)

        # Add a bold format to use to highlight cells.
        bold = workbook.add_format({'bold': True})
        worksheet.write(start_activity_row, start_activity_col, activity, bold)

        next_row = start_activity_row + 1
        next_col = start_activity_col + 1

        for key, value in motions_statistics.iteritems():

            if len(value) != 0:
                worksheet.write(next_row, next_col, key)

                next_row += 1
                next_col += 1

                for stat in ['sensitivity', 'specificity', 'ppv', 'npv']:
                    worksheet.write(next_row, next_col, stat)
                    next_col += 1

                next_col -= 4
                next_row += 1

                value_list = list()
                for index_val, stat in enumerate(value):
                    new_array = np.array(stat)
                    tmp = np.sum(new_array) / np.shape(new_array)[0]
                    value_list.append(tmp)

                for val in value_list:
                    if math.isnan(val):
                        worksheet.write_string(next_row, next_col, 'nan')
                    else:
                        worksheet.write(next_row, next_col, val)
                    next_col += 1

                next_col -= 5
                next_row += 1

        if next_row > total_number_row:
            total_number_row = next_row

        # keep count of the number of activities within a row
        column_activity += 1


if __name__ == '__main__':

    # get the file name and its window and number of states used
    tmp_filename = raw_input('log filename: ')
    list_filename = list()
    if ',' in tmp_filename:
        list_filename = tmp_filename.replace(' ', '').split(',')
    else:
        filename = tmp_filename

    if len(list_filename) != 0:
        # create excel object
        excel_filename = 'imu_logs.xlsx'
        workbook = xlsxwriter.Workbook(excel_filename)

        for s_file in list_filename:
            run(csv_filename=s_file, workbook=workbook)

    else:
        excel_filename = filename + '.xlsx'
        workbook = xlsxwriter.Workbook(excel_filename)
        run(csv_filename=filename, workbook=workbook)

    # closing excel object
    workbook.close()



