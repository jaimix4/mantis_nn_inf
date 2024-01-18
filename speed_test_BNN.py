import matplotlib.pyplot as plt
from   matplotlib        import rc
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.ticker as tick
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
# for loading models 
import tensorflow as tf
import models
import timeit

# # make plot latex
# rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 9})
# rc('text', usetex=True)

# # Size in centimeters
# width_cm = 7.2
# height_cm = 3.5

# # Convert size from centimeters to inches
# width_in = width_cm / 2.54
# height_in = height_cm / 2.54

# plt.figure(figsize=(width_in, height_in))

# load model 

model_path = 'models_v3/experiments_11/trim-serenity-8'
model_ensemble = models.BNN_crazy_ensemble_DNN(model_path=model_path)

# create ensemble of model to test 

num_networks = 1
model_ready = model_ensemble.ensemble_DNN_single_input(num_networks=num_networks)

# load test set for testing model 
X_test, y_test = model_ensemble.load_test_set()

# and now trimmed to a numbers of cells
n_cells = 32500
X_test = X_test[:n_cells, :]
y_test = y_test[:n_cells, :]

# for now make separation of data set (class) here
y_test_class = y_test[:, -1].copy()
y_test_class[y_test[:, -1] < model_ensemble.config_params['rec3_class']] = 1.0
y_test_class[y_test[:, -1] >= model_ensemble.config_params['rec3_class']] = 0.0

# outputs_labels = model_ensemble.config_params['outputs_labels'][:-1]
# print(outputs_labels)

# this is what needs to be timed 
# create a function that does the inference

# tf.debugging.set_log_device_placement(True)

for k in range(1, 11):

    def forward_pass():
        for i in range(k):
            predictions_raw = model_ready(X_test)


    for i in range(10):
        predictions_raw = forward_pass()

    # time it now that is has been compiled
        
    num_runs = 100
    total_time = timeit.timeit(forward_pass, number=num_runs)
    average_time = total_time/num_runs
    # print average time in ms 
    print(f'Average time for forward pass for k = {k}, n_cells = {n_cells}; t -> {average_time*1e3:.3f} ms')



# predictions_raw = np.reshape(predictions_raw, (y_test.shape[0], y_test.shape[1] + 1, num_networks), order='F')

# y_predict_realspace = np.mean(predictions_raw.copy(), axis = 2)
# y_predict_std_realspace = (100*np.std(predictions_raw.copy(), axis = 2, dtype = np.float64))/y_predict_realspace

# # not taking into account values that do not matter 
# # since they were classified as recombination dominant 
# mask_nan = np.ones_like(y_predict_realspace[:, -1])
# mask_nan[y_predict_realspace[:, -1] < 0.5] = np.nan

# #y_predict_realspace[:, 0] = y_predict_realspace[:, 0] * mask_nan
# # y_predict_realspace[:, 0] = y_predict_realspace[:, 0] * mask_nan
# # y_predict_realspace[:, 1] = y_predict_realspace[:, 1] * mask_nan
# # y_predict_realspace[:, 4] = y_predict_realspace[:, 4] * mask_nan

# y_predict_realspace[:, 2] = y_predict_realspace[:, 2] * mask_nan
# y_predict_realspace[:, 3] = y_predict_realspace[:, 3] * mask_nan
# y_predict_realspace[:, 4] = y_predict_realspace[:, 4] * mask_nan

# y_class_predict = np.ones_like(y_test[:, -1])
# y_class_predict[y_predict_realspace[:, -1] < 0.5] = 0










