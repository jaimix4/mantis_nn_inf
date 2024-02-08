import numpy as np
import pandas as pd
from CRM_ADAS import Deuterium
from scipy.stats import qmc
import subprocess
import scipy.io as sio

# extract data from matlab 
data = sio.loadmat('SOLPS-ITER/plasma_params_simul.mat')

Te = data['plasma_params_simul']['Te'][0][0]
# flatten the data, from 98x38 to 98*38
original_shape = Te.shape
print(original_shape)
len_arr = Te.shape[0]*Te.shape[1]
print(len_arr)

# 'Te[eV]', 'ne[m-3]', 'no[m-3]', 'nHe[m-3]', 'nHe+[m-3]'
# '   0   ', '   1   ', '   2   ', '   3   ', '    4    ' 
plasma_parms = np.zeros((len_arr, 5))
plasma_parms[:, 0] = np.reshape(data['plasma_params_simul']['Te'][0][0], (len_arr, ))
plasma_parms[:, 1] = np.reshape(data['plasma_params_simul']['ne'][0][0], (len_arr, ))
plasma_parms[:, 2] = np.reshape(data['plasma_params_simul']['n0'][0][0], (len_arr, ))
plasma_parms[:, 3] = np.reshape(data['plasma_params_simul']['nHe'][0][0], (len_arr, ))
plasma_parms[:, 4] = np.reshape(data['plasma_params_simul']['nHeplus'][0][0], (len_arr, ))


# 'emi_3-2','emi_4-2','emi_5-2','emi_7-2','Irate','Rrate','728/668norec','706/668norec','He728norec','He706norec','He668norec','728/668','706/668','He728','He706','He668','Brec3/B3'
# '   0   ','   1   ','   2   ','   3   ','  4  ','  5  ','    6       ','    7       ','    8     ','    9     ','    10    ','  11   ','  12   ','  13 ','  14 ','  15 ','  16    '      
emis = np.zeros((len_arr, 17))

# filling H - ADAS emissions

deu_crm = Deuterium()

for idx in range(len_arr):

    emis[idx, :4], emis[idx, -1] = deu_crm.compute_emissivites_ratio_B3rec(plasma_parms[idx, :3])
    emis[idx, 4:6] = deu_crm.compute_rates(plasma_parms[idx, :3])[:2]

# filling He - Goto emissions
    
nam_tempfile_input = 'temp_emis_SOLPS_cfile_absHe.csv'
nam_tempfile_output = 'temp_emis_SOLPS_cfile_out_absHe.csv'
np.savetxt(nam_tempfile_input, plasma_parms[:, :2], delimiter=',')

print('\nOutput of the c file: ')
subprocess.run(['./emissions_jaime', nam_tempfile_input, nam_tempfile_output])
print('\n c file for He PECS run done. Name of temporary file for data: {} \n this file can be ignored or deleted'.format(nam_tempfile_input))

arr = np.loadtxt(nam_tempfile_output, delimiter=",", dtype=np.float64)

# 728/706 no rec
emis[:, 6] = arr[:, 2]/arr[:, 3]
# 706/668 no rec
emis[:, 7] = arr[:, 2]/arr[:, 4]
# He728 no rec
emis[:, 8] = (plasma_parms[:,1] * plasma_parms[:,3] * arr[:, 2]) 
# He706 no rec
emis[:, 9] = (plasma_parms[:,1] * plasma_parms[:,3] * arr[:, 3]) 
# He728 no rec
emis[:, 10] = (plasma_parms[:,1] * plasma_parms[:,3] * arr[:, 4]) 

# He728
emis[:, 13] = (plasma_parms[:,1] * plasma_parms[:,3] * arr[:, 2]) + (plasma_parms[:,1] * plasma_parms[:,4] * arr[:, 5])
# He706
emis[:, 14] = (plasma_parms[:,1] * plasma_parms[:,3] * arr[:, 3]) + (plasma_parms[:,1] * plasma_parms[:,4] * arr[:, 6])
# He668
emis[:, 15] = (plasma_parms[:,1] * plasma_parms[:,3] * arr[:, 4]) + (plasma_parms[:,1] * plasma_parms[:,4] * arr[:, 7])

# 728/706
emis[:, 11] = emis[:, 13]/emis[:, 14]
# 728/668
emis[:, 12] = emis[:, 13]/emis[:, 15]

emis = emis.reshape((original_shape[0], original_shape[1], 17))
print('jaime')
print(emis.shape)

# save data to .mat
sio.savemat('SOLPS-ITER/forward_emis_adas_goto_SOLPS-ITER_sims.mat', \
    {'emi_3_2': emis[:,:,0], \
        'emi_4_2': emis[:,:,1], \
            'emi_5_2': emis[:,:,2], \
                'emi_7_2': emis[:,:,3], \
                    'Irate': emis[:,:,4], \
                        'Rrate': emis[:,:,5], \
                            'He_lrt_728_706norec': emis[:,:,6], \
                                'He_lrt_728_668norec': emis[:,:,7], \
                                    'He728norec': emis[:,:,8], \
                                        'He706norec': emis[:,:,9], \
                                            'He668norec': emis[:,:,10], \
                                                'He_lrt_728_706': emis[:,:,11], \
                                                    'He_lrt_728_668': emis[:,:,12], \
                                                        'He728': emis[:,:,13], \
                                                            'He706': emis[:,:,14], \
                                                                'He668': emis[:,:,15], \
                                                                    'Brec3_B3': emis[:,:,16]})

print("outputs saved")


# Neural network thingy



