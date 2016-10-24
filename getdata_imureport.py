import h5py
import numpy as np

filepath = 'converted_dataset/reportdata.hdf5'
raw_data_object = h5py.File(filepath, 'r')
raw_data = raw_data_object['Q130_pilot_OT_r_radialcan_t1.mvnx.mat']
filepath = 'processed_dataset/processed_dataset.hdf5'
processed_data_object = h5py.File(filepath, 'r')
processed_data = processed_data_object['Q130_pilot_OT_r_radialcan_t1.mvnx.mat']

print 'hello'

np.savetxt("raw_data.csv", raw_data, delimiter=",")
np.savetxt("processed_data.csv", processed_data, delimiter=",")

