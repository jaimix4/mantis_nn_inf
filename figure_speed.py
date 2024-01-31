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
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 9})
rc('text', usetex=True)

# set width of bar 
barWidth = 0.25

# Size in centimeters
width_cm = 7.5
height_cm = 7.5

# Convert size from centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54

fig = plt.figure(figsize=(width_in, height_in), dpi = 300)

# ensemble DNN M1 Max
# Average time for forward pass for k = 1, n_cells = 32500; t -> 8.765 ms
# Average time for forward pass for k = 2, n_cells = 32500; t -> 14.332 ms
# Average time for forward pass for k = 3, n_cells = 32500; t -> 19.625 ms
# Average time for forward pass for k = 4, n_cells = 32500; t -> 25.064 ms
# Average time for forward pass for k = 5, n_cells = 32500; t -> 31.068 ms
# Average time for forward pass for k = 6, n_cells = 32500; t -> 35.880 ms
# Average time for forward pass for k = 7, n_cells = 32500; t -> 38.765 ms
# Average time for forward pass for k = 8, n_cells = 32500; t -> 47.175 ms
# Average time for forward pass for k = 9, n_cells = 32500; t -> 50.055 ms
# Average time for forward pass for k = 10, n_cells = 32500; t -> 54.380 ms

# BNN M1 Max
# Average time for forward pass for k = 1, n_cells = 32500; t -> 9.325 ms
# Average time for forward pass for k = 2, n_cells = 32500; t -> 19.204 ms
# Average time for forward pass for k = 3, n_cells = 32500; t -> 27.366 ms
# Average time for forward pass for k = 4, n_cells = 32500; t -> 37.395 ms
# Average time for forward pass for k = 5, n_cells = 32500; t -> 46.391 ms
# Average time for forward pass for k = 6, n_cells = 32500; t -> 55.494 ms
# Average time for forward pass for k = 7, n_cells = 32500; t -> 65.604 ms
# Average time for forward pass for k = 8, n_cells = 32500; t -> 75.248 ms
# Average time for forward pass for k = 9, n_cells = 32500; t -> 85.015 ms
# Average time for forward pass for k = 10, n_cells = 32500; t -> 93.492 ms

# BNN A100
# Average time for forward pass for k = 1, n_cells = 32500; t -> 4.197 ms
# Average time for forward pass for k = 2, n_cells = 32500; t -> 17.899 ms
# Average time for forward pass for k = 3, n_cells = 32500; t -> 25.841 ms
# Average time for forward pass for k = 4, n_cells = 32500; t -> 18.027 ms
# Average time for forward pass for k = 5, n_cells = 32500; t -> 17.502 ms
# Average time for forward pass for k = 6, n_cells = 32500; t -> 21.007 ms
# Average time for forward pass for k = 7, n_cells = 32500; t -> 24.338 ms
# Average time for forward pass for k = 8, n_cells = 32500; t -> 27.636 ms
# Average time for forward pass for k = 9, n_cells = 32500; t -> 34.534 ms
# Average time for forward pass for k = 10, n_cells = 32500; t -> 34.050 ms

# BNN A100
# Average time for forward pass for k = 1, n_cells = 32500; t -> 3.575 ms
# Average time for forward pass for k = 2, n_cells = 32500; t -> 7.626 ms
# Average time for forward pass for k = 3, n_cells = 32500; t -> 11.649 ms
# Average time for forward pass for k = 4, n_cells = 32500; t -> 14.349 ms
# Average time for forward pass for k = 5, n_cells = 32500; t -> 18.345 ms
# Average time for forward pass for k = 6, n_cells = 32500; t -> 21.037 ms
# Average time for forward pass for k = 7, n_cells = 32500; t -> 42.825 ms
# Average time for forward pass for k = 8, n_cells = 32500; t -> 28.579 ms
# Average time for forward pass for k = 9, n_cells = 32500; t -> 40.824 ms
# Average time for forward pass for k = 10, n_cells = 32500; t -> 61.190 ms

# ensemble DNN A100
# Average time for forward pass for k = 1, n_cells = 32500; t -> 3.645 ms
# Average time for forward pass for k = 2, n_cells = 32500; t -> 6.284 ms
# Average time for forward pass for k = 3, n_cells = 32500; t -> 8.906 ms
# Average time for forward pass for k = 4, n_cells = 32500; t -> 12.645 ms
# Average time for forward pass for k = 5, n_cells = 32500; t -> 14.899 ms
# Average time for forward pass for k = 6, n_cells = 32500; t -> 36.661 ms
# Average time for forward pass for k = 7, n_cells = 32500; t -> 20.227 ms
# Average time for forward pass for k = 8, n_cells = 32500; t -> 59.929 ms
# Average time for forward pass for k = 9, n_cells = 32500; t -> 24.837 ms
# Average time for forward pass for k = 10, n_cells = 32500; t -> 27.343 ms


# set height of bar 
# speed in ms for k = 10
# IT = [93.492, 34.050, 54.380, 27.343, 1] 
IT = [93.492, 61.190, 54.380, 27.343, 0] 
# speed in ms for k = 6
ECE = [55.494, 21.037, 35.880, 36.661, 0] 
# speed in ms for k = 4
CSE = [37.395, 14.349, 25.064, 12.645, 0.9] 
 
# Set position of bar on X axis 
br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
 
# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth, 
        edgecolor ='black', label =r'$k = 10$') 
plt.bar(br2, ECE, color ='g', width = barWidth, 
        edgecolor ='black', label =r'$k = 6$') 
plt.bar(br3, CSE, color ='b', width = barWidth, 
        edgecolor ='black', label =r'$k = 4$') 
 
# Adding Xticks 
# plt.xlabel(' ', fontweight ='bold', fontsize = 15) 
plt.ylabel('inference time (ms)', fontsize = 12) 
plt.xticks([r + barWidth for r in range(len(IT))], 
        ['BNN M1 Max', 'BNN A100', 'ens. DNN M1 Max', 'ens. DNN A100', 'ens. DNN A100 TensorRT'], \
                rotation = 30, rotation_mode = 'anchor', ha = 'right')


plt.yscale('log')
plt.legend()


plt.tight_layout()

# plt.suptitle(r'${\rm Goto-CRM}, \: B_0 = 1.4 \: [{\rm T}]$', fontsize=fontsize_labels+2)

plt.savefig('figures_paper/fig_speed_test.png', dpi = 600, bbox_inches='tight', transparent = True)