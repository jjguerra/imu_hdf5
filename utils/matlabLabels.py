

class MatlabLabels(object):

    def __init__(self):
        self.possible_motions = ['_REST_', '_REA_', '_RET_', '_I_', '_S_', '_ST_', '_SM_', '_M_', '_MC_', '_T_', '_TC_',
                                 '_THM_']

        self.labels = ['R_REST_B', 'R_REST_E', 'R_REA_A0_B', 'R_REA_A0_E', 'R_T_A0A1_B', 'R_T_A0A1_E', 'R_RET_A1_B',
                       'R_RET_A1_E', 'R_REA_A1_B', 'R_REA_A1_E', 'R_T_A1A0_B', 'R_T_A1A0_E', 'R_RET_A0_B', 'R_RET_A0_E',
                       'R_REA_B0_B', 'R_REA_B0_E', 'R_T_B0B1_B', 'R_T_B0B1_E', 'R_RET_B1_B', 'R_RET_B1_E', 'R_REA_B1_B',
                       'R_REA_B1_E', 'R_T_B1I_B', 'R_T_B1I_E', 'R_I_B', 'R_I_E', 'R_T_IB0_B', 'R_T_IB0_E', 'R_RET_B0_B',
                       'R_RET_B0_E', 'R_REA_C0_B', 'R_REA_C0_E', 'R_T_C0C1_B', 'R_T_C0C1_E', 'R_RET_C1_B', 'R_RET_C1_E',
                       'R_REA_C1_B', 'R_REA_C1_E', 'R_T_C1C0_B', 'R_T_C1C0_E', 'R_RET_C0_B', 'R_RET_C0_E', 'R_T_B1B0_B',
                       'R_T_B1B0_E', 'R_REA*_A0_B', 'R_REA*_A0_E', 'R_RET*_A0_B', 'R_RET*_A0_E', 'R_T*_A1A0_B',
                       'R_T*_A1A0_E', 'R_REA_B', 'R_REA_E', 'R_T_B', 'R_T_E', 'R_SM_B', 'R_SM_E', 'R_S_B', 'R_S_E',
                       'R_ST_B', 'R_ST_E', 'R_THM_B', 'R_THM_E', 'R_RET_B', 'R_RET_E', 'R_T*_B', 'R_T*_E', 'R_M_B',
                       'R_M_E', 'R_TC_B', 'R_TC_E', 'R_REA*_B', 'R_REA*_E', 'R_REA_C_B', 'R_REA_C_E', 'R_T_C1_B',
                       'R_T_C1_E', 'R_RET_1_B', 'R_RET_1_E', 'R_REA_1_B', 'R_REA_1_E', 'R_T_1C_B', 'R_T_1C_E',
                       'R_RET_C_B', 'R_RET_C_E', 'R_RET_2_B', 'R_RET_2_E', 'R_REA_2_B', 'R_REA_2_E', 'R_T_2C_B',
                       'R_T_2C_E', 'R_T_C2_B', 'R_T_C2_E', 'R_RET_3_B', 'R_RET_3_E', 'R_REA_3_B', 'R_REA_3_E',
                       'R_T_3C_B', 'R_T_3C_E', 'R_T_C3_B', 'R_T_C3_E', 'R_RET_4_B', 'R_RET_4_E', 'R_REA_4_B',
                       'R_REA_4_E', 'R_T_4C_B', 'R_T_4C_E', 'R_T_C4_B', 'R_T_C4_E', 'R_RET_5_B', 'R_RET_5_E',
                       'R_REA_5_B', 'R_REA_5_E', 'R_T_5C_B', 'R_T_5C_E', 'R_T_C5_B', 'R_T_C5_E', 'R_RET_6_B',
                       'R_RET_6_E', 'R_REA_6_B', 'R_REA_6_E', 'R_T_6C_B', 'R_T_6C_E', 'R_T_C6_B', 'R_T_C6_E',
                       'R_RET_7_B', 'R_RET_7_E', 'R_REA_7_B', 'R_REA_7_E', 'R_T_7C_B', 'R_T_7C_E', 'R_T_C7_B',
                       'R_T_C7_E', 'R_RET_8_B', 'R_RET_8_E', 'R_REA_8_B', 'R_REA_8_E', 'R_T_8C_B', 'R_T_8C_E',
                       'R_T_C8_B', 'R_T_C8_E', 'R_T*_A0A1_B', 'R_T*_A0A1_E', 'R_RET*_B0_B', 'R_RET*_B0_E', 'R_T_B1BO_B',
                       'R_T_B1BO_E', 'R_RET*_B1_B', 'R_RET*_B1_E', 'R_T*_C2C0_B', 'R_T*_C2C0_E', 'R_T*_B2B0_B',
                       'R_T*_B2B0_E', 'R_RET*_C0_B', 'R_RET*_C0_E', 'R_T_C0C2_B', 'R_T_C0C2_E', 'R_RET_C2_B',
                       'R_RET_C2_E', 'R_REA_C2_B', 'R_REA_C2_E', 'R_T_C2C0_B', 'R_T_C2C0_E', 'R_T_B0B2_B', 'R_T_B0B2_E',
                       'R_RET_B2_B', 'R_RET_B2_E', 'R_REA_B2_B', 'R_REA_B2_E', 'R_T_B2B0_B', 'R_T_B2B0_E', 'R_T_A0A2_B',
                       'R_T_A0A2_E', 'R_RET_A2_B', 'R_RET_A2_E', 'R_REA_A2_B', 'R_REA_A2_E', 'R_T_A2A0_B', 'R_T_A2A0_E',
                       'R_T*_A0A2_B', 'R_T*_A0A2_E', 'R_T_T2C0_B', 'R_T_T2C0_E', 'R_REA*_B0_B', 'R_REA*_B0_E',
                       'R_REA*_B1_B', 'R_REA*_B1_E', 'R_S_B0_B', 'R_S_B0_E', 'R_RET_B2B0_B', 'R_RET_B2B0_E',
                       'R_RET_A2A0_B', 'R_RET_A2A0_E', 'R_MC_B', 'R_MC_E', 'R_T_B0A2_B', 'R_T_B0A2_E',
                       'R_T_A2B0_B', 'R_T_A2B0_E', 'R_RET*_A2_B', 'R_RET*_A2_E', 'R_T_B0A1_B', 'R_T_B0A1_E',
                       'R_T_A1B0_B', 'R_T_A1B0_E', 'R_T*_B0B1_B', 'R_T*_B0B1_E', 'R_T*_C1C0_B', 'R_T*_C1C0_E',
                       'R_REA*_C0_B', 'R_REA*_C0_E',
                       'L_REST_B', 'L_REST_E', 'L_REA_A0_B', 'L_REA_A0_E', 'L_T_A0A1_B', 'L_T_A0A1_E', 'L_RET_A1_B',
                       'L_RET_A1_E', 'L_REA_A1_B', 'L_REA_A1_E', 'L_T_A1A0_B', 'L_T_A1A0_E', 'L_RET_A0_B', 'L_RET_A0_E',
                       'L_REA_B0_B', 'L_REA_B0_E', 'L_T_B0B1_B', 'L_T_B0B1_E', 'L_RET_B1_B', 'L_RET_B1_E', 'L_REA_B1_B',
                       'L_REA_B1_E', 'L_T_B1I_B', 'L_T_B1I_E', 'L_I_B', 'L_I_E', 'L_T_IB0_B', 'L_T_IB0_E', 'L_RET_B0_B',
                       'L_RET_B0_E', 'L_REA_C0_B', 'L_REA_C0_E', 'L_T_C0C1_B', 'L_T_C0C1_E', 'L_RET_C1_B', 'L_RET_C1_E',
                       'L_REA_C1_B', 'L_REA_C1_E', 'L_T_C1C0_B', 'L_T_C1C0_E', 'L_RET_C0_B', 'L_RET_C0_E', 'L_T_B1B0_B',
                       'L_T_B1B0_E', 'L_REA*_A0_B', 'L_REA*_A0_E', 'L_RET*_A0_B', 'L_RET*_A0_E', 'L_T*_A1A0_B',
                       'L_T*_A1A0_E', 'L_REA_B', 'L_REA_E', 'L_T_B', 'L_T_E', 'L_SM_B', 'L_SM_E', 'L_S_B', 'L_S_E',
                       'L_ST_B', 'L_ST_E', 'L_THM_B', 'L_THM_E', 'L_RET_B', 'L_RET_E', 'L_T*_B', 'L_T*_E', 'L_M_B',
                       'L_M_E', 'L_TC_B', 'L_TC_E', 'L_REA*_B', 'L_REA*_E', 'L_REA_C_B', 'L_REA_C_E', 'L_T_C1_B',
                       'L_T_C1_E', 'L_RET_1_B', 'L_RET_1_E', 'L_REA_1_B', 'L_REA_1_E', 'L_T_1C_B', 'L_T_1C_E',
                       'L_RET_C_B', 'L_RET_C_E', 'L_RET_2_B', 'L_RET_2_E', 'L_REA_2_B', 'L_REA_2_E', 'L_T_2C_B',
                       'L_T_2C_E', 'L_T_C2_B', 'L_T_C2_E', 'L_RET_3_B', 'L_RET_3_E', 'L_REA_3_B', 'L_REA_3_E',
                       'L_T_3C_B', 'L_T_3C_E', 'L_T_C3_B', 'L_T_C3_E', 'L_RET_4_B', 'L_RET_4_E', 'L_REA_4_B',
                       'L_REA_4_E', 'L_T_4C_B', 'L_T_4C_E', 'L_T_C4_B', 'L_T_C4_E', 'L_RET_5_B', 'L_RET_5_E',
                       'L_REA_5_B', 'L_REA_5_E', 'L_T_5C_B', 'L_T_5C_E', 'L_T_C5_B', 'L_T_C5_E', 'L_RET_6_B',
                       'L_RET_6_E', 'L_REA_6_B', 'L_REA_6_E', 'L_T_6C_B', 'L_T_6C_E', 'L_T_C6_B', 'L_T_C6_E',
                       'L_RET_7_B', 'L_RET_7_E', 'L_REA_7_B', 'L_REA_7_E', 'L_T_7C_B', 'L_T_7C_E', 'L_T_C7_B',
                       'L_T_C7_E', 'L_RET_8_B', 'L_RET_8_E', 'L_REA_8_B', 'L_REA_8_E', 'L_T_8C_B', 'L_T_8C_E',
                       'L_T_C8_B', 'L_T_C8_E', 'L_T*_A0A1_B', 'L_T*_A0A1_E', 'L_RET*_B0_B', 'L_RET*_B0_E', 'L_T_B1BO_B',
                       'L_T_B1BO_E', 'L_T_C0C2_B', 'L_T_C0C2_E', 'L_RET_C2_B', 'L_RET_C2_E', 'L_REA_C2_B', 'L_REA_C2_E',
                       'L_T_C2C0_B', 'L_T_C2C0_E', 'L_T_B0B2_B', 'L_T_B0B2_E', 'L_RET_B2_B', 'L_RET_B2_E', 'L_REA_B2_B',
                       'L_REA_B2_E', 'L_T_B2B0_B', 'L_T_B2B0_E', 'L_T_A0A2_B', 'L_T_A0A2_E', 'L_RET_A2_B', 'L_RET_A2_E',
                       'L_REA_A2_B', 'L_REA_A2_E', 'L_T_A2A0_B', 'L_T_A2A0_E', 'L_T*_A0A2_B', 'L_T*_A0A2_E',
                       'L_T_T2C0_B', 'L_T_T2C0_E', 'L_REA*_B0_B', 'L_REA*_B0_E', 'L_REA*_B1_B', 'L_REA*_B1_E',
                       'L_RET*_B1_B', 'L_RET*_B1_E', 'L_T*_C2C0_B', 'L_T*_C2C0_E', 'L_T*_B2B0_B', 'L_T*_B2B0_E',
                       'L_RET*_C0_B', 'L_RET*_C0_E', 'L_S_B0_B', 'L_S_B0_E', 'L_RET_B2B0_B', 'L_RET_B2B0_E',
                       'L_RET_A2A0_B', 'L_RET_A2A0_E', 'L_MC_B', 'L_MC_E', 'L_T_B0A2_B', 'L_T_B0A2_E',
                       'L_T_A2B0_B', 'L_T_A2B0_E', 'L_RET*_A2_B', 'L_RET*_A2_E', 'L_T_B0A1_B', 'L_T_B0A1_E',
                       'L_T_A1B0_B', 'L_T_A1B0_E', 'L_T*_B0B1_B', 'L_T*_B0B1_E', 'L_T*_C1C0_B', 'L_T*_C1C0_E',
                       'L_REA*_C0_B', 'L_REA*_C0_E',
                       'P_REST_B', 'P_REST_E', 'P_REA_A0_B', 'P_REA_A0_E', 'P_T_A0A1_B', 'P_T_A0A1_E', 'P_RET_A1_B',
                       'P_RET_A1_E', 'P_REA_A1_B', 'P_REA_A1_E', 'P_T_A1A0_B', 'P_T_A1A0_E', 'P_RET_A0_B', 'P_RET_A0_E',
                       'P_REA_B0_B', 'P_REA_B0_E', 'P_T_B0B1_B', 'P_T_B0B1_E', 'P_RET_B1_B', 'P_RET_B1_E', 'P_REA_B1_B',
                       'P_REA_B1_E', 'P_T_B1I_B', 'P_T_B1I_E', 'P_I_B', 'P_I_E', 'P_T_IB0_B', 'P_T_IB0_E', 'P_RET_B0_B',
                       'P_RET_B0_E', 'P_REA_C0_B', 'P_REA_C0_E', 'P_T_C0C1_B', 'P_T_C0C1_E', 'P_RET_C1_B', 'P_RET_C1_E',
                       'P_REA_C1_B', 'P_REA_C1_E', 'P_T_C1C0_B', 'P_T_C1C0_E', 'P_RET_C0_B', 'P_RET_C0_E', 'P_T_B1B0_B',
                       'P_T_B1B0_E', 'P_REA*_A0_B', 'P_REA*_A0_E', 'P_RET*_A0_B', 'P_RET*_A0_E', 'P_T*_A1A0_B',
                       'P_T*_A1A0_E', 'P_REA_B', 'P_REA_E', 'P_T_B', 'P_T_E', 'P_SM_B', 'P_SM_E', 'P_S_B', 'P_S_E',
                       'P_ST_B', 'P_ST_E', 'P_THM_B', 'P_THM_E', 'P_RET_B', 'P_RET_E', 'P_T*_B', 'P_T*_E', 'P_M_B',
                       'P_M_E', 'P_TC_B', 'P_TC_E', 'P_REA*_B', 'P_REA*_E', 'P_REA_C_B', 'P_REA_C_E', 'P_T_C1_B',
                       'P_T_C1_E', 'P_RET_1_B', 'P_RET_1_E', 'P_REA_1_B', 'P_REA_1_E', 'P_T_1C_B', 'P_T_1C_E',
                       'P_RET_C_B', 'P_RET_C_E', 'P_RET_2_B', 'P_RET_2_E', 'P_REA_2_B', 'P_REA_2_E', 'P_T_2C_B',
                       'P_T_2C_E', 'P_T_C2_B', 'P_T_C2_E', 'P_RET_3_B', 'P_RET_3_E', 'P_REA_3_B', 'P_REA_3_E',
                       'P_T_3C_B', 'P_T_3C_E', 'P_T_C3_B', 'P_T_C3_E', 'P_RET_4_B', 'P_RET_4_E', 'P_REA_4_B',
                       'P_REA_4_E', 'P_T_4C_B', 'P_T_4C_E', 'P_T_C4_B', 'P_T_C4_E', 'P_RET_5_B', 'P_RET_5_E',
                       'P_REA_5_B', 'P_REA_5_E', 'P_T_5C_B', 'P_T_5C_E', 'P_T_C5_B', 'P_T_C5_E', 'P_RET_6_B',
                       'P_RET_6_E', 'P_REA_6_B', 'P_REA_6_E', 'P_T_6C_B', 'P_T_6C_E', 'P_T_C6_B', 'P_T_C6_E',
                       'P_RET_7_B', 'P_RET_7_E', 'P_REA_7_B', 'P_REA_7_E', 'P_T_7C_B', 'P_T_7C_E', 'P_T_C7_B',
                       'P_T_C7_E', 'P_RET_8_B', 'P_RET_8_E', 'P_REA_8_B', 'P_REA_8_E', 'P_T_8C_B', 'P_T_8C_E',
                       'P_T_C8_B', 'P_T_C8_E', 'P_T*_A0A1_B', 'P_T*_A0A1_E', 'P_RET*_B0_B', 'P_RET*_B0_E', 'P_T_B1BO_B',
                       'P_T_B1BO_E', 'P_RET*_B1_B', 'P_RET*_B1_E', 'P_T*_C2C0_B', 'P_T*_C2C0_E', 'P_T*_B2B0_B',
                       'P_T*_B2B0_E', 'P_RET*_C0_B', 'P_RET*_C0_E', 'P_T_C0C2_B', 'P_T_C0C2_E', 'P_RET_C2_B',
                       'P_RET_C2_E', 'P_REA_C2_B', 'P_REA_C2_E', 'P_T_C2C0_B', 'P_T_C2C0_E', 'P_T_B0B2_B', 'P_T_B0B2_E',
                       'P_RET_B2_B', 'P_RET_B2_E', 'P_REA_B2_B', 'P_REA_B2_E', 'P_T_B2B0_B', 'P_T_B2B0_E', 'P_T_A0A2_B',
                       'P_T_A0A2_E', 'P_RET_A2_B', 'P_RET_A2_E', 'P_REA_A2_B', 'P_REA_A2_E', 'P_T_A2A0_B', 'P_T_A2A0_E',
                       'P_T*_A0A2_B', 'P_T*_A0A2_E', 'P_T_T2C0_B', 'P_T_T2C0_E', 'P_REA*_B0_B', 'P_REA*_B0_E',
                       'P_REA*_B1_B', 'P_REA*_B1_E', 'P_S_B0_B', 'P_S_B0_E', 'P_RET_B2B0_B', 'P_RET_B2B0_E',
                       'P_RET_A2A0_B', 'P_RET_A2A0_E', 'P_MC_B', 'P_MC_E', 'P_T_B0A2_B', 'P_T_B0A2_E',
                       'P_T_A2B0_B', 'P_T_A2B0_E', 'P_RET*_A2_B', 'P_RET*_A2_E', 'P_T_B0A1_B', 'P_T_B0A1_E',
                       'P_T_A1B0_B', 'P_T_A1B0_E', 'P_T*_B0B1_B', 'P_T*_B0B1_E', 'P_T*_C1C0_B', 'P_T*_C1C0_E',
                       'P_REA*_C0_B', 'P_REA*_C0_E',
                       'N_REST_B', 'N_REST_E', 'N_REA_A0_B', 'N_REA_A0_E', 'N_T_A0A1_B', 'N_T_A0A1_E', 'N_RET_A1_B',
                       'N_RET_A1_E', 'N_REA_A1_B', 'N_REA_A1_E', 'N_T_A1A0_B', 'N_T_A1A0_E', 'N_RET_A0_B', 'N_RET_A0_E',
                       'N_REA_B0_B', 'N_REA_B0_E', 'N_T_B0B1_B', 'N_T_B0B1_E', 'N_RET_B1_B', 'N_RET_B1_E', 'N_REA_B1_B',
                       'N_REA_B1_E', 'N_T_B1I_B', 'N_T_B1I_E', 'N_I_B', 'N_I_E', 'N_T_IB0_B', 'N_T_IB0_E', 'N_RET_B0_B',
                       'N_RET_B0_E', 'N_REA_C0_B', 'N_REA_C0_E', 'N_T_C0C1_B', 'N_T_C0C1_E', 'N_RET_C1_B', 'N_RET_C1_E',
                       'N_REA_C1_B', 'N_REA_C1_E', 'N_T_C1C0_B', 'N_T_C1C0_E', 'N_RET_C0_B', 'N_RET_C0_E', 'N_T_B1B0_B',
                       'N_T_B1B0_E', 'N_REA*_A0_B', 'N_REA*_A0_E', 'N_RET*_A0_B', 'N_RET*_A0_E', 'N_T*_A1A0_B',
                       'N_T*_A1A0_E', 'N_REA_B', 'N_REA_E', 'N_T_B', 'N_T_E', 'N_SM_B', 'N_SM_E', 'N_S_B', 'N_S_E',
                       'N_ST_B', 'N_ST_E', 'N_THM_B', 'N_THM_E', 'N_RET_B', 'N_RET_E', 'N_T*_B', 'N_T*_E', 'N_M_B',
                       'N_M_E', 'N_TC_B', 'N_TC_E', 'N_REA*_B', 'N_REA*_E', 'N_REA_C_B', 'N_REA_C_E', 'N_T_C1_B',
                       'N_T_C1_E', 'N_RET_1_B', 'N_RET_1_E', 'N_REA_1_B', 'N_REA_1_E', 'N_T_1C_B', 'N_T_1C_E',
                       'N_RET_C_B', 'N_RET_C_E', 'N_RET_2_B', 'N_RET_2_E', 'N_REA_2_B', 'N_REA_2_E', 'N_T_2C_B',
                       'N_T_2C_E', 'N_T_C2_B', 'N_T_C2_E', 'N_RET_3_B', 'N_RET_3_E', 'N_REA_3_B', 'N_REA_3_E',
                       'N_T_3C_B', 'N_T_3C_E', 'N_T_C3_B', 'N_T_C3_E', 'N_RET_4_B', 'N_RET_4_E', 'N_REA_4_B',
                       'N_REA_4_E', 'N_T_4C_B', 'N_T_4C_E', 'N_T_C4_B', 'N_T_C4_E', 'N_RET_5_B', 'N_RET_5_E',
                       'N_REA_5_B', 'N_REA_5_E', 'N_T_5C_B', 'N_T_5C_E', 'N_T_C5_B', 'N_T_C5_E', 'N_RET_6_B',
                       'N_RET_6_E', 'N_REA_6_B', 'N_REA_6_E', 'N_T_6C_B', 'N_T_6C_E', 'N_T_C6_B', 'N_T_C6_E',
                       'N_RET_7_B', 'N_RET_7_E', 'N_REA_7_B', 'N_REA_7_E', 'N_T_7C_B', 'N_T_7C_E', 'N_T_C7_B',
                       'N_T_C7_E', 'N_RET_8_B', 'N_RET_8_E', 'N_REA_8_B', 'N_REA_8_E', 'N_T_8C_B', 'N_T_8C_E',
                       'N_T_C8_B', 'N_T_C8_E', 'N_T*_A0A1_B', 'N_T*_A0A1_E', 'N_RET*_B0_B', 'N_RET*_B0_E', 'N_T_B1BO_B',
                       'N_T_B1BO_E', 'N_RET*_B1_B', 'N_RET*_B1_E', 'N_T*_C2C0_B', 'N_T*_C2C0_E', 'N_T*_B2B0_B',
                       'N_T*_B2B0_E', 'N_RET*_C0_B', 'N_RET*_C0_E', 'N_T_C0C2_B', 'N_T_C0C2_E', 'N_RET_C2_B',
                       'N_RET_C2_E', 'N_REA_C2_B', 'N_REA_C2_E', 'N_T_C2C0_B', 'N_T_C2C0_E', 'N_T_B0B2_B', 'N_T_B0B2_E',
                       'N_RET_B2_B', 'N_RET_B2_E', 'N_REA_B2_B', 'N_REA_B2_E', 'N_T_B2B0_B', 'N_T_B2B0_E', 'N_T_A0A2_B',
                       'N_T_A0A2_E', 'N_RET_A2_B', 'N_RET_A2_E', 'N_REA_A2_B', 'N_REA_A2_E', 'N_T_A2A0_B', 'N_T_A2A0_E',
                       'N_T*_A0A2_B', 'N_T*_A0A2_E', 'N_T_T2C0_B', 'N_T_T2C0_E', 'N_REA*_B0_B', 'N_REA*_B0_E',
                       'N_REA*_B1_B', 'N_REA*_B1_E', 'N_S_B0_B', 'N_S_B0_E', 'N_RET_B2B0_B', 'N_RET_B2B0_E',
                       'N_RET_A2A0_B', 'N_RET_A2A0_E', 'N_MC_B', 'N_MC_E', 'N_T_B0A2_B', 'N_T_B0A2_E',
                       'N_T_A2B0_B', 'N_T_A2B0_E', 'N_RET*_A2_B', 'N_RET*_A2_E', 'N_T_B0A1_B', 'N_T_B0A1_E',
                       'N_T_A1B0_B', 'N_T_A1B0_E', 'N_T*_B0B1_B', 'N_T*_B0B1_E', 'N_T*_C1C0_B', 'N_T*_C1C0_E',
                       'N_REA*_C0_B', 'N_REA*_C0_E',
                       'N_IGNORE_B', 'N_IGNORE_E', 'P_IGNORE_B', 'P_IGNORE_E', 'R_IGNORE_B', 'R_IGNORE_E',
                       'L_IGNORE_B', 'L_IGNORE_E', 'IGNORE_B', 'IGNORE_E']

        self.jointUsed = ['']

        self.segmentUsed = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                            'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand']

        self.vectorsUsed = ['orientation', 'position', 'velocity', 'acceleration', 'angularVelocity',
                            'angularAcceleration']

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