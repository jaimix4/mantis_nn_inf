import matplotlib.pyplot as plt
from   matplotlib        import rc

import numpy as np
import pandas as pd

import sys
from pathlib import Path
import pyarrow.parquet as pq

data_folder = 'raw_datasets/dataset_exp_master_'
inputs_labels = ['emi_3-2', 'emi_5-2', '728/706', '728/668']
outputs_labels = ['Te', 'ne', 'no', 'Irate', 'Rrate', 'Brec3/B3']

if Path(data_folder + 'TRAIN.parquet').is_file():

    data_set_train = pq.read_table(data_folder + 'TRAIN.parquet').to_pandas()
    data_set_val = pq.read_table(data_folder + 'VAL.parquet').to_pandas()
    data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()
   

    print("----PARQUET FILES FOUND-----")

# assign all inputs to X_full
X_train = data_set_train[inputs_labels].values
X_val = data_set_val[inputs_labels].values
X_test = data_set_test[inputs_labels].values

# assign all outputs to y_full
y_train = data_set_train[outputs_labels].values
y_val = data_set_val[outputs_labels].values
y_test = data_set_test[outputs_labels].values

X = np.vstack((X_train, X_val, X_test))
y = np.vstack((y_train, y_val, y_test))

# make plot latex
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 9})
rc('text', usetex=True)


rec3 = y[:200000, 5].copy()
Irate = y[:200000, 3].copy()
Rrate = y[:200000, 4].copy()

Te = y[:200000, 0].copy()

# fig = plt.figure(figsize=(5, 4), dpi = 300)

# 
split = 0.95

# Size in centimeters
width_cm = 8
height_cm = 7

# Convert size from centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54

fig, axs = plt.subplots(2, 1, figsize=(width_in, height_in), dpi = 300, gridspec_kw={'height_ratios': [2.2,1]})
# plt.tight_layout()
# fig.tight_layout(pad = 2.0)
# 
# fig.tight_layout(rect=[0.05, 0.1, 0.05, 1.0])

# first plot 

axs[0].plot(rec3, Irate, 'bo', markersize = 0.3, alpha=0.1, label=r'$I_{\rm rate}$')
axs[0].plot(rec3, Rrate, 'ro', markersize = 0.3, alpha=0.1, label=r'$R_{\rm rate}$')
axs[0].set_yscale('log')
axs[0].set_ylim([1e13, 1e26])
axs[0].set_xlim([-0.02, 1.02])
axs[0].set_ylabel(r'$[{\rm m^{-3} s^{-1}}]$') 
# keep x ticks marks but numbers are not visible
axs[0].set_xticklabels([' ', ' ', ' '])
# add grid lines 
# axs[0].grid(True, which='both', alpha = 0.5, zorder = 0)




# plot vertical line 
axs[0].axvline(x=split, color='k', linestyle='--', linewidth=1.5)
#plt.axvline(x=0.02, color='k', linestyle='--', linewidth=0.5)
# axs[0].legend()

# axs[0].fill_betweenx(Rrate, 0.95, 1,  facecolor='green', alpha=.5)
# square_x = [0.95, 1.0, 1.0, 0.95, 0.95]
# square_y = [1e12, 1e12, 1e26, 1e26, 1e12]
# axs[0].fill(square_x, square_y, "b", alpha = 0.1)


# second plot

axs[1].plot(rec3, Te, 'go', markersize=0.3, alpha=0.2) #, label='r$T_e$')
axs[1].set_yscale('log')
axs[1].axvline(x=split, color='k', linestyle='--', linewidth=1.5, label = str(split))
axs[1].set_ylabel(r'$T_e \: \: [{\rm eV}]$') 
axs[1].set_xlabel(r'$D^{\rm rec}_{3 \rightarrow 2}$') 
# axs[1].legend()
axs[1].set_xlim([-0.02, 1.02])
# axs[0].set_xticklabels([' ', ' ', ' '])
# add grid lines but just on specifics on the y axis
# axs[1].grid(True, which='both', alpha = 0.5, zorder = 0)


# square_x = [split, 1.0, 1.0, split, split]
# square_y = [0, 0, 2e2, 2e2, 0]
# axs[1].fill(square_x, square_y, "b", alpha = 0.1)


#
fig.subplots_adjust(hspace=0.03)
plt.savefig('figures_paper/figure_irate_d3rec.png', dpi = 600, bbox_inches='tight', transparent = True)
#plt.show()


# count the number of values below 0.05 and above 0.95
print('Number of values below 0.02: ', np.sum(rec3 < 0.02))
print('Number of values above {}: '.format(split), np.sum(rec3 > split))
# count number of values between 0.05 and 0.95
print('Number of values between 0.02 and {}: '.format(split), np.sum((rec3 > 0.02) & (rec3 < split)))