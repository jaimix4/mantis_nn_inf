import tensorflow as tf
from tensorflow.keras.models import load_model
import keras 
import numpy as np
import scipy.io as io
# your_script.py
import sys

# Access command-line arguments
arguments = sys.argv[1:]

# Print the arguments
print("Input arguments:", arguments)

model = load_model("models_matlab/" + arguments[0])

num_networks = int(arguments[0][-1])

print(num_networks)

def test_exp_data_bnn(model_ready, data = 'exp_data_mantis/sh78224.mat'):

    # this function is taylor for data from shot 78224

    data = io.loadmat('sh78224.mat')
    print('succesfully loaded data from shot 78224')

    # here I will take into account the missing time at Da_emi
    cells_num_per_shot = data['Da_emi'].shape[0]
    time_steps = data['Da_emi'].shape[1]
    channels_num = 4
    data_emis = np.zeros((cells_num_per_shot, time_steps, channels_num))

    Da_arr = data['Da_emi']
    # make values equal to zero nan
    # Da_arr[Da_arr == 0.0] = np.nan

    # index of missing time step
    idx_missing = 5

    Dg_arr = data['Dg_emi']
    Dg_arr = np.delete(Dg_arr, idx_missing, axis = 1)
    # Dg_arr[Dg_arr == 0.0] = np.nan
    

    He706_arr = data['He706_emi']
    He706_arr = np.delete(He706_arr, idx_missing, axis = 1)
    # He706_arr[He706_arr == 0.0] = np.nan

    He728_arr = data['He728_emi']
    He728_arr = np.delete(He728_arr, idx_missing, axis = 1)
    # He728_arr[He728_arr == 0.0] = np.nan

    He667_arr = data['He667_emi']
    He667_arr = np.delete(He667_arr, idx_missing, axis = 1)
    # He667_arr[He667_arr == 0.0] = np.nan

    Ratio_728_706 = np.nan_to_num(He728_arr/He706_arr, nan = 0.0, posinf = 0.0, neginf = 0.0)

    Ratio_728_668 = np.nan_to_num(He728_arr/He667_arr, nan = 0.0, posinf = 0.0, neginf = 0.0)

    # Ratio_728_706 = He728_arr/He706_arr
    # Ratio_728_668 = He728_arr/He667_arr

    # oder and He ratios ['emi_3-2', 'emi_5-2', '728/706', '728/668']

    # Dalpha
    data_emis[:, :, 0] = Da_arr
    data_emis[:, :, 1] = Dg_arr
    data_emis[:, :, 2] = Ratio_728_706
    data_emis[:, :, 3] = Ratio_728_668

    # make values equal to zero into nan
    data_emis[data_emis == 0.0] = np.nan

    # create array to store results
    # for now lets hard code it 
    # output array 
    # ['Te', 'ne', 'no', 'Irate', 'Rrate', 'Brec3/B3']
    outputs_mean_network = np.zeros((cells_num_per_shot, time_steps, 6))
    outputs_std_network = np.zeros((cells_num_per_shot, time_steps, 6))

    # go through all cells at each time step 

    for idx in range(time_steps):

        # extract input data
        input = data_emis[:, idx, :].copy()

        # predict output
        result_full = model_ready(input)

        # that seven is because I know that I have 7 outputs
        result_full = np.reshape(result_full, \
            (input.shape[0], 7, num_networks), \
                order='F')

        result_mean = np.mean(result_full.copy(), axis = 2)
        result_std = (100*np.std(result_full.copy(), axis = 2, dtype = np.float64))/result_mean

        # use classification output to put nan into no and Irate 

        # not taking into account values that do not matter 
        # since they were classified as recombination dominant 
        mask_nan = np.ones_like(result_mean[:, -1])
        mask_nan[result_mean[:, -1] < 0.5] = np.nan

        # put nan no and Irate when they are classified as recombination dominant
        #result_mean[:, 0] = result_mean[:, 0] * mask_nan
        result_mean[:, 2] = result_mean[:, 2] * mask_nan
        result_mean[:, 3] = result_mean[:, 3] #* mask_nan

        #result_std[:, 0] = result_std[:, 0] * mask_nan
        result_std[:, 2] = result_std[:, 2] * mask_nan
        result_std[:, 3] = result_std[:, 3] #* mask_nan

        # regression outputs
        outputs_mean_network[:, idx, :-1] = result_mean[:, :-2]
        outputs_std_network[:, idx, :-1] = result_std[:, :-2]

        # classification outputs
        outputs_mean_network[:, idx, -1] = mask_nan
        outputs_std_network[:, idx, -1] = result_std[:, -2]

        # make values of outputs equal to nan where, Dalpha is 0.0
        # mask_Da = np.ones_like(Da_arr)
        #outputs_mean_network[:, idx, :] = Da_arr[:, idx]

    mask_Da = np.ones_like(Da_arr)
    mask_Da[Da_arr == 0.0] = np.nan

    mask_Dg = np.ones_like(Dg_arr)
    mask_Da[Dg_arr == 0.0] = np.nan

    mask_Ratio_728_706 = np.ones_like(Ratio_728_706)
    mask_Ratio_728_706[Ratio_728_706 == 0.0] = np.nan

    #mask_Ratio_728_706[Ratio_728_706 > 0.70] = np.nan

    mask_Ratio_728_668 = np.ones_like(Ratio_728_668)
    mask_Ratio_728_668[Ratio_728_668 == 0.0] = np.nan

    outputs_mean_network[:, :, 0] = outputs_mean_network[:, :, 0] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_mean_network[:, :, 1] = outputs_mean_network[:, :, 1] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_mean_network[:, :, 2] = outputs_mean_network[:, :, 2] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_mean_network[:, :, 3] = outputs_mean_network[:, :, 3] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_mean_network[:, :, 4] = outputs_mean_network[:, :, 4] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_mean_network[:, :, 5] = outputs_mean_network[:, :, 5] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668

    outputs_std_network[:, :, 0] = outputs_std_network[:, :, 0] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_std_network[:, :, 1] = outputs_std_network[:, :, 1] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_std_network[:, :, 2] = outputs_std_network[:, :, 2] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_std_network[:, :, 3] = outputs_std_network[:, :, 3] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_std_network[:, :, 4] = outputs_std_network[:, :, 4] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668
    outputs_std_network[:, :, 5] = outputs_std_network[:, :, 5] * mask_Da * mask_Dg * mask_Ratio_728_706 * mask_Ratio_728_668

    # outputs_mean_network = np.nan_to_num(outputs_mean_network, nan = 0.0, posinf = 0.0, neginf = 0.0)
    # outputs_std_network = np.nan_to_num(outputs_std_network, nan = 0.0, posinf = 0.0, neginf = 0.0)
    return outputs_mean_network, outputs_std_network

outputs_mean_network, outputs_std_network = test_exp_data_bnn(model)

io.savemat('exp_data_mantis/output_' + arguments[0] + '.mat', \
    {'Te_mean': outputs_mean_network[:, :, 0], \
        'ne_mean': outputs_mean_network[:, :, 1], \
            'no_mean': outputs_mean_network[:, :, 2], \
                'Irate_mean': outputs_mean_network[:, :, 3], \
                    'Rrate_mean': outputs_mean_network[:, :, 4], \
                        'rec_dom_mean': outputs_mean_network[:, :, 5],\
                            'Te_std': outputs_std_network[:, :, 0], \
                                'ne_std': outputs_std_network[:, :, 1], \
                                    'no_std': outputs_std_network[:, :, 2], \
                                        'Irate_std': outputs_std_network[:, :, 3], \
                                            'Rrate_std': outputs_std_network[:, :, 4], \
                                                'rec_dom_std': outputs_std_network[:, :, 5]})
print("outputs saved")


    
