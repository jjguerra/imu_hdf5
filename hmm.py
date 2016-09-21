import numpy as np
import os
import re

from utils.logfileproperties import Document


def load_data(data_dir):

    subjectDict = dict()
    # data file are store within the project dataset folder
    activityFolders = os.listdir(dataset)

    for cActivity in activityFolders:
        trainingSetPath = os.path.join(dataset, cActivity)
        fileList = os.listdir(trainingSetPath)
        regex = re.compile('TrainingData_' + sensors + '_' + '((?<=_)[^_]+(?=_))')
        userList = [{matFile: cUser.group(1)} for matFile in fileList for cUser in [regex.search(matFile)] if cUser]
    for lItem in userList:
        cfile, subjectName = lItem.items()[0]
        if subjectName in subjectDict.keys():
            subjectDict[subjectName].files.append(os.path.join(trainingSetPath, cfile))
        else:
            subjectInfo = subjectInformation()
            subjectInfo.files.append(os.path.join(trainingSetPath, cfile))
            subjectDict[subjectName] = subjectInfo

    for cSubject in subjectDict.keys():
        for cFile in subjectDict[cSubject].files:
            raw_data = np.load(cFile)
            # labels are located in the last column
            labelsIndeces = raw_data[:, -1].tolist()
            subjectDict[cSubject].labels.append(labelsIndeces)
            subjectDict[cSubject].sensorData.append(raw_data[:, 0:-1])


def imu_hmm(dataset_directory):

    file_info = Document(dataset_directory)


