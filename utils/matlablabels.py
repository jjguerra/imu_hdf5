

class MatlabLabels(object):

    def __init__(self):

        self.s_patients_dexterity = {'_r_': ['Q440', 'Q445', 'Q480', 'Q482', 'Q492', 'Q540'],
                                     '_l_': ['Q275', 'Q441', 'Q491', 'Q535']}

        self.possible_motions = ['_REST_', '_REA_', '_RET_', '_I_', '_S_', '_ST_', '_SM_', '_M_', '_MC_', '_T_', '_TC_',
                                 '_THM_']

        self.labels = ['IGNORE_B', 'IGNORE_E', '_I*_B', '_I*_E', '_IGNORE_B', '_IGNORE_E', '_I_B', '_I_E', '_MC_B',
                       '_MC_E', '_M_B', '_M_E', '_REA*_1_B', '_REA*_1_E', '_REA*_2_B', '_REA*_2_E', '_REA*_3_B',
                       '_REA*_3_E', '_REA*_4_B', '_REA*_4_E', '_REA*_5_B', '_REA*_5_E', '_REA*_6_B', '_REA*_6_E',
                       '_REA*_7_B', '_REA*_7_E', '_REA*_8_B', '_REA*_8_E', '_REA*_A0_B', '_REA*_A0_E', '_REA*_B',
                       '_REA*_B0_B', '_REA*_B0_E', '_REA*_B1_B', '_REA*_B1_E', '_REA*_C0_B', '_REA*_C0_E', '_REA*_C_B',
                       '_REA*_C_E', '_REA*_E', '_REA_1_B', '_REA_1_E', '_REA_2_B', '_REA_2_E', '_REA_3_B', '_REA_3_E',
                       '_REA_4_B', '_REA_4_E', '_REA_5_B', '_REA_5_E', '_REA_6_B', '_REA_6_E', '_REA_7_B', '_REA_7_E',
                       '_REA_8_B', '_REA_8_E', '_REA_A0_B', '_REA_A0_E', '_REA_A1_B', '_REA_A1_E', '_REA_A2_B',
                       '_REA_A2_E', '_REA_B', '_REA_B0_B', '_REA_B0_E', '_REA_B1_B', '_REA_B1_E', '_REA_B2_B',
                       '_REA_B2_E', '_REA_C0_B', '_REA_C0_E', '_REA_C1_B', '_REA_C1_E', '_REA_C2_B', '_REA_C2_E',
                       '_REA_C_B', '_REA_C_E', '_REA_E', '_REST_B', '_REST_E', '_RET*_1_B', '_RET*_1_E', '_RET*_2_B',
                       '_RET*_2_E', '_RET*_3_B', '_RET*_3_E', '_RET*_4_B', '_RET*_4_E', '_RET*_5_B', '_RET*_5_E',
                       '_RET*_6_B', '_RET*_6_E', '_RET*_7_B', '_RET*_7_E', '_RET*_8_B', '_RET*_8_E', '_RET*_A0_B',
                       '_RET*_A0_E', '_RET*_A1_B', '_RET*_A1_E', '_RET*_A2_B', '_RET*_A2_E', '_RET*_B', '_RET*_B0_B',
                       '_RET*_B0_E', '_RET*_B1_B', '_RET*_B1_E', '_RET*_C0_B', '_RET*_C0_E', '_RET*_C_B', '_RET*_C_E',
                       '_RET*_E', '_RET_1_B', '_RET_1_E', '_RET_2_B', '_RET_2_E', '_RET_3_B', '_RET_3_E', '_RET_4_B',
                       '_RET_4_E', '_RET_5_B', '_RET_5_E', '_RET_6_B', '_RET_6_E', '_RET_7_B', '_RET_7_E', '_RET_8_B',
                       '_RET_8_E', '_RET_A0_B', '_RET_A0_E', '_RET_A1_B', '_RET_A1_E', '_RET_A2A0_B', '_RET_A2A0_E',
                       '_RET_A2_B', '_RET_A2_E', '_RET_B', '_RET_B0_B', '_RET_B0_E', '_RET_B1_B', '_RET_B1_E',
                       '_RET_B2B0_B', '_RET_B2B0_E', '_RET_B2_B', '_RET_B2_E', '_RET_C0_B', '_RET_C0_E', '_RET_C1_B',
                       '_RET_C1_E', '_RET_C2_B', '_RET_C2_E', '_RET_C_B', '_RET_C_E', '_RET_E', '_S*_C_B', '_S*_C_E',
                       '_SM_B', '_SM_E', '_ST_B', '_ST_E', '_S_1_B', '_S_1_E', '_S_2_B', '_S_2_E', '_S_3_B', '_S_3_E',
                       '_S_4_B', '_S_4_E', '_S_5_B', '_S_5_E', '_S_6_B', '_S_6_E', '_S_7_B', '_S_7_E', '_S_8_B',
                       '_S_8_E', '_S_B', '_S_B0_B', '_S_B0_E', '_S_C_B', '_S_C_E', '_S_E', '_T*_1C_B', '_T*_1C_E',
                       '_T*_2C_B', '_T*_2C_E', '_T*_3C_B', '_T*_3C_E', '_T*_4C_B', '_T*_4C_E', '_T*_5C_B', '_T*_5C_E',
                       '_T*_6C_B', '_T*_6C_E', '_T*_7C_B', '_T*_7C_E', '_T*_8C_B', '_T*_8C_E', '_T*_A0A1_B',
                       '_T*_A0A1_E', '_T*_A0A2_B', '_T*_A0A2_E', '_T*_A1A0_B', '_T*_A1A0_E', '_T*_B', '_T*_B0B1_B',
                       '_T*_B0B1_E', '_T*_B2B0_B', '_T*_B2B0_E', '_T*_C1C0_B', '_T*_C1C0_E', '_T*_C1_B', '_T*_C1_E',
                       '_T*_C2C0_B', '_T*_C2C0_E', '_T*_C2_B', '_T*_C2_E', '_T*_C3_B', '_T*_C3_E', '_T*_C4_B',
                       '_T*_C4_E', '_T*_C5_B', '_T*_C5_E', '_T*_C6_B', '_T*_C6_E', '_T*_C7_B', '_T*_C7_E', '_T*_C8_B',
                       '_T*_C8_E', '_T*_E', '_TC_B', '_TC_E', '_THM*_B', '_THM*_E', '_THM_B', '_THM_E', '_T_1C_B',
                       '_T_1C_E', '_T_2C_B', '_T_2C_E', '_T_3C_B', '_T_3C_E', '_T_4C_B', '_T_4C_E', '_T_5C_B',
                       '_T_5C_E', '_T_6C_B', '_T_6C_E', '_T_7C_B', '_T_7C_E', '_T_8C_B', '_T_8C_E', '_T_A0A1_B',
                       '_T_A0A1_E', '_T_A0A2_B', '_T_A0A2_E', '_T_A0B1_B', '_T_A0B1_E', '_T_A1A0_B', '_T_A1A0_E',
                       '_T_A1B0_B', '_T_A1B0_E', '_T_A2A0_B', '_T_A2A0_E', '_T_A2B0_B', '_T_A2B0_E', '_T_B',
                       '_T_B0A1_B', '_T_B0A1_E', '_T_B0A2_B', '_T_B0A2_E', '_T_B0B1_B', '_T_B0B1_E', '_T_B0B2_B',
                       '_T_B0B2_E', '_T_B1B0_B', '_T_B1B0_E', '_T_B1I_B', '_T_B1I_E', '_T_B2B0_B', '_T_B2B0_E',
                       '_T_C0C1_B', '_T_C0C1_E', '_T_C0C2_B', '_T_C0C2_E', '_T_C1C0_B', '_T_C1C0_E', '_T_C1_B',
                       '_T_C1_E', '_T_C2C0_B', '_T_C2C0_E', '_T_C2C1_B', '_T_C2C1_E', '_T_C2_B', '_T_C2_E', '_T_C3_B',
                       '_T_C3_E', '_T_C4_B', '_T_C4_E', '_T_C5_B', '_T_C5_E', '_T_C6_B', '_T_C6_E', '_T_C7_B',
                       '_T_C7_E', '_T_C8_B', '_T_C8_E', '_T_E', '_T_IB0_B', '_T_IB0_E', '_T_T2C0_B', '_T_T2C0_E']

        self.compact_list = ['REST', 'REA', 'T', 'RET', 'I', 'S', 'ST', 'SM', 'M', 'MC', 'THM', 'TC']

        self.jointUsed = ['']

        self.segmentUsed = ['RightHand']

        self.vectorsUsed = ['velocity']

        # self.segmentUsed = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand']
        #
        # self.vectorsUsed = ['orientation', 'position', 'velocity', 'acceleration', 'angularVelocity',
        #                     'angularAcceleration']

        # self.segmentUsed = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
        #                     'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand']

        # self.vectorsUsed = ['orientation', 'position', 'velocity', 'acceleration', 'angularVelocity',
        #                     'angularAcceleration']

        #     vectorsUsed = ['orientation', 'position', 'velocity', 'acceleration', 'angularVelocity',        \
        #                    'angularAcceleration', 'sensorAcceleration', 'sensorAngularVelocity',            \
        #                    'sensorOrientation', 'jointAngle', 'jointAngleXZY']
        #     segmentUsed = ['Pelvis','T8','Head','RightShoulder','RightUpperArm','RightForeArm','RightHand',
        # 'LeftShoulder', \
        #                    'LeftUpperArm','LeftForeArm','LeftHand']
        #     jointUsed = ['jL5S1','jL4L3','jL1T12','jT9T8','jT1C7','jC1Head','jRightC7Shoulder',             \
        #                  'jRightShoulder','jRightElbow','jRightWrist','jLeftC7Shoulder','jLeftShoulder',    \
        #                  'jLeftElbow','jLeftWrist','jRightHip','jRightKnee','jRightAnkle','jRightBallFoot', \
        #                  'jLeftHip','jLeftKnee','jLeftAnkle','jLeftBallFoot']
