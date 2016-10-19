import h5py
import numpy as np
from utils.matlabmover import file_information


if __name__ == '__main__':

    input_filepath = '/Users/jguerra/PycharmProjects/imu_hdf5/dataset/processed_sensordata.hdf5'
    in_h5object = h5py.File(input_filepath, 'r')

    out_filepath = '/Users/jguerra/PycharmProjects/imu_hdf5/dataset/processed_sensordata_merged.hdf5'
    out_h5object = h5py.File(out_filepath, 'w')

    for filename in in_h5object.iterkeys():
        print 'processing: {0}'.format(filename)

        if 'paretic' in filename:
            user, pareticnonparetic, activenonactive, activity, time = file_information(filename)
            n_filename = user + '_' + pareticnonparetic + '_' + activenonactive + '_' + activity + '.mvnx.mat'
        else:
            user, leftright, activity, time = file_information(filename)
            n_filename = user + '_pilot_OT_' + leftright + '_' + activity + '.mvnx.mat'

        # if not different repetitions, then just add the file
        if not time:
            print 'adding new filename: ' + n_filename
            out_h5object.create_dataset(name=filename, data=in_h5object[filename][:], chunks=True)
            if 'paretic' in filename:
                out_h5object[filename].attrs['user'] = user
                out_h5object[filename].attrs['activity'] = activity
                out_h5object[filename].attrs['pareticnonparetic'] = pareticnonparetic
                out_h5object[filename].attrs['activenonactive'] = activenonactive
            else:
                out_h5object[filename].attrs['user'] = user
                out_h5object[filename].attrs['activity'] = activity
                out_h5object[filename].attrs['leftright'] = leftright

        else:
            print 'merging file: {0} to new file:{1}'.format(filename, n_filename)
            n_row, n_col = np.shape(in_h5object[filename][:])

            if n_filename in out_h5object.keys():
                o_row, o_col = np.shape(out_h5object[n_filename][:])
                out_h5object[n_filename].resize(o_row + n_row, axis=0)
                out_h5object[n_filename][o_row:] = in_h5object[filename][:]

            else:
                out_h5object.create_dataset(name=n_filename, data=in_h5object[filename][:], maxshape=(None, n_col),
                                            chunks=True)

            if 'paretic' in n_filename:
                out_h5object[n_filename].attrs['user'] = user
                out_h5object[n_filename].attrs['activity'] = activity
                out_h5object[n_filename].attrs['pareticnonparetic'] = pareticnonparetic
                out_h5object[n_filename].attrs['activenonactive'] = activenonactive
            else:
                out_h5object[n_filename].attrs['user'] = user
                out_h5object[n_filename].attrs['activity'] = activity
                out_h5object[n_filename].attrs['leftright'] = leftright

