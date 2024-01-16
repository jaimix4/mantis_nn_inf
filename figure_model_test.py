import models
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
from tensorflow.keras.models import load_model
import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score as r2_score_metric
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

# make plot latex
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 5})
rc('text', usetex=True)

model_path = 'models_v2/experiments_8/fancy-wind-12'
model_ensemble = models.BNN_crazy_ensemble_DNN(model_path=model_path)

model_path = 'models_v2/experiments_8/fancy-wind-12_ensemble_DNN_6'
model_ready = load_model(model_path)


num_networks = int(model_path.split('_')[-1])

# loading test data

X_test, y_test = model_ensemble.load_test_set()

# for now make separation of data set (class) here
y_test_class = y_test[:, -1].copy()
y_test_class[y_test[:, -1] < model_ensemble.config_params['rec3_class']] = 1.0
y_test_class[y_test[:, -1] >= model_ensemble.config_params['rec3_class']] = 0.0

outputs_labels = model_ensemble.config_params['outputs_labels'][:-1]
print(outputs_labels)

# #predictions_raw = model_ready([X_test]*num_networks)
predictions_raw = model_ready(X_test)
# remeber that output shape now is (..., 7, num_networks)
predictions_raw = np.reshape(predictions_raw, (y_test.shape[0], y_test.shape[1] + 1, num_networks), order='F')

y_predict_realspace = np.mean(predictions_raw.copy(), axis = 2)
y_predict_std_realspace = (100*np.std(predictions_raw.copy(), axis = 2, dtype = np.float64))/y_predict_realspace

# not taking into account values that do not matter 
# since they were classified as recombination dominant 
mask_nan = np.ones_like(y_predict_realspace[:, -1])
mask_nan[y_predict_realspace[:, -1] < 0.5] = np.nan

#y_predict_realspace[:, 0] = y_predict_realspace[:, 0] * mask_nan
# y_predict_realspace[:, 0] = y_predict_realspace[:, 0] * mask_nan
# y_predict_realspace[:, 1] = y_predict_realspace[:, 1] * mask_nan
# y_predict_realspace[:, 4] = y_predict_realspace[:, 4] * mask_nan

y_predict_realspace[:, 2] = y_predict_realspace[:, 2] * mask_nan
y_predict_realspace[:, 3] = y_predict_realspace[:, 3] * mask_nan
y_predict_realspace[:, 4] = y_predict_realspace[:, 4] * mask_nan

y_class_predict = np.ones_like(y_test[:, -1])
y_class_predict[y_predict_realspace[:, -1] < 0.5] = 0


# figure settings

# Size in centimeters
width_cm = 19
height_cm = 4.5

# Convert size from centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54


# fig, ax = plt.subplots(2, y_test.shape[1] - 1, figsize = (width_in, height_in), gridspec_kw={'height_ratios': [4,1], 'width_ratios': [1,1,1,1,1]}) 

plt.figure(figsize=(width_in, height_in))
plt.tight_layout()
gs = gridspec.GridSpec(5, y_test.shape[1] - 1, height_ratios=[3, 0.3, 2/8, 2/5, 2/8], width_ratios=[1,1,1,1,1], wspace=0.38, hspace=0.1)


# define an ax for color bar whole bottom row
ax_cb = plt.subplot(gs[1, :])

ax_cm = plt.subplot(gs[2:, :])

ax = []
for i in range(y_test.shape[1] - 1):
    ax.append(plt.subplot(gs[0, i]))


latex_output_labels = [r'$T_e$', r'$n_e$', r'$n_o$', r'$I_{\mathrm{rate}}$',  r'$R_{\mathrm{rate}}$', 'B_{rec3}/B_3']


for i in range(y_test.shape[1] - 1):

    print("  " + outputs_labels[i], end= ": ")
    # if i == 0 or i == 1 or i == 4:
    # if i == 1 or i == 4:
    #     r2_score = r2_score_metric(y_test[:,i], y_predict_realspace[:,i])
    # else:
    y_true_for_r2 = y_test[:,i].copy()
    y_pred_for_r2 = y_predict_realspace[:,i].copy()

    non_nan_indices = ~np.isnan(y_pred_for_r2)

    y_true_no_nan = y_true_for_r2[non_nan_indices]
    y_pred_no_nan = y_pred_for_r2[non_nan_indices]

    r2_score = r2_score_metric(y_true_no_nan, y_pred_no_nan)
    print('{:.6f}'.format(r2_score), end = ' // ')

    error_line_arr = np.linspace(np.min(y_test[:,i]), np.max(y_test[:,i]), 100)
        
    #ax[i].plot(error_line_arr, error_line_arr, 'k-', linewidth = 0.5)
    ax[i].plot(error_line_arr, error_line_arr*1.4, 'r--', linewidth = 0.5)
    ax[i].plot(error_line_arr, error_line_arr*0.6, 'r--', linewidth = 0.5, label = r'$\sigma_{\mathrm{rsd}} \pm 40\%$')
    # figure out error bars
    # ax.errorbar(y_test, mean, yerr=np.std(predictions, axis=1), fmt='o', alpha = 0.1)

    #ax[i].plot(y_test_truth_realspace[:,i], y_predict_realspace[:,i], 'o', alpha = 0.05, markersize=2)
    pcm = ax[i].scatter(y_test[:,i], y_predict_realspace[:,i], \
        c = y_predict_std_realspace[:,i], cmap = 'jet', alpha = 0.08, s = 0.08, \
            vmin = 0.0, vmax = 40.0)
    
    #ax[i].set_title('{}, r2_score = {:.4f}'.format(outputs_labels[i], r2_score))
    ax[i].set_title(latex_output_labels[i], fontsize = 10)

    # ax[i].text(0.1, 0.9, latex_output_labels[i], \
    #     transform=ax[i].transAxes, fontsize=10, verticalalignment='top')

    # put r2 score in the plot
    # ax[i].text(0.35, 0.2, r'$r^2 = $' + '{:.3f}'.format(r2_score), \
    #     transform=ax[i].transAxes, fontsize=8, verticalalignment='top')
    
    ax[i].text(0.1, 0.9, r'$r^2 = $' + '{:.3f}'.format(r2_score), \
        transform=ax[i].transAxes, fontsize=8, verticalalignment='top')

    ax[i].set_xlabel('True', fontsize = 8)
    if i == 0:
        ax[i].set_ylabel('Predicted', fontsize = 8)
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')

    # if i == 0:

        # line = mlines.Line2D([], [], color='red', marker='_', linestyle='--',
        #               markersize=10, label='Line Symbol')

        # # Create a legend for the line symbol
        # legend1 = ax[i].legend(handles=[line], loc='upper left')

        # #ax[i].legend(loc = 'lower right', fontsize = 4)

        # ax[i] = plt.gca().add_artist(legend1)

        # ax[i].legend(['Label'], loc='lower left')
    
    #if i == 2:

        #ax[i].set_title('Regression', fontsize = 10)

    if i == 3:

        ax[i].set_xlim(1e19, 3e25)
        ax[i].set_ylim(1e19, 3e25)

    if i == 4:

        ax[i].set_xlim(1e16, 3e22)
        ax[i].set_ylim(1e16, 3e22)

    # if i + 1 == y_test.shape[1] - 1:
    #     # add color bar

    #     # set the min and max of the color from 0 to 40

norm = Normalize(vmin=0.0, vmax=40.0)

sm = ScalarMappable(cmap='jet', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax = ax_cb, orientation = 'horizontal')#, location = 'top')

#cbar = fig.colorbar(pcm, ax = ax, orientation = 'horizontal', location = 'bottom')
# cbar.set_label(r'$\sigma_{\mathrm{rsd}} \: \: [\%]$', fontsize = 9)

cm = confusion_matrix(y_test_class, y_class_predict)
# print confusion matrix
print('\nConfusion matrix:')
print(cm)

ax_cm = plt.subplot(gs[2:, :-1])
cax_cb = plt.subplot(gs[3, -1])


# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rec. +', 'rec. -'])
ax_cm.set_aspect('auto')
cax = ax_cm.matshow(cm, cmap='Blues', aspect='auto')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i+0.1, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

# Create a divider for the existing axes instance
# divider = make_axes_locatable(ax_cm)

# Append axes to the right of ax_cm, with 20% width of ax_cm
# cax_cb = divider.append_axes("right", size="20%", pad=0.05)

plt.colorbar(cax, cax=cax_cb, orientation = 'horizontal', fraction = 0.01, pad = 0.05, ticks = [500, 150000])
# use scientific notation in the ticks of the colorbar above and just one decimal place
# cax_cb.xaxis.set_major_formatter(tick.ScalarFormatter('%.1e'))
# cax_cb.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


# Add labels, title and ticks
ax_cm.set_xlabel('Predicted', fontsize = 8)
ax_cm.set_ylabel('True', fontsize = 8)
# ax_cm.set_title('Classification', fontsize = 8)


ax_cm.set_xticklabels([''] + ['non-ionising', 'ionising'])
ax_cm.xaxis.set_ticks_position('bottom')
ax_cm.xaxis.set_tick_params(labeltop='off')
# get rid of ticks labels on the x axis top
ax_cm.xaxis.set_tick_params(labeltop='off', labelbottom='on')
# get rid of ticks on the x axis top


# Set the y-tick labels
ax_cm.set_yticklabels([''] + ['non-ionising', 'ionising'])
# rotate the tick labels and set their alignment
plt.setp(ax_cm.get_yticklabels(), rotation=70, ha="right",
         rotation_mode="anchor")

# Use the below line if you want to show numbers in each cell
# ax_cm.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

# I want

# Adjust the position of the second subplot
pos1 = ax_cb.get_position() # get the original position 
pos2 = [pos1.x0, pos1.y0 - 0.23,  pos1.width, pos1.height] 
ax_cb.set_position(pos2) # set a new position

pos1 = ax_cm.get_position() # get the original position 
pos2 = [pos1.x0, pos1.y0 - 0.5,  pos1.width, pos1.height] 
ax_cm.set_position(pos2) # set a new position

pos1 = cax_cb.get_position() # get the original position 
pos2 = [pos1.x0, pos1.y0 - 0.5,  pos1.width, pos1.height] 
cax_cb.set_position(pos2) # set a new position

# disp.plot(ax = ax_cm, cmap = 'Blues', values_format = 'd')

plt.savefig('figures_paper/fig_model_test_' + 'fancy-wind-12_ensemble_DNN_6.png', dpi = 300, bbox_inches='tight', transparent = True)



