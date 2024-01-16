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


# set height of bar 
# speed in ms for k = 12
IT = [50, 30, 15, 6, 1] 
# speed in ms for k = 8
ECE = [25, 12, 8, 3, 0.6] 
# speed in ms for k = 4
CSE = [18, 10, 5, 0.5, 0.2] 
 
# Set position of bar on X axis 
br1 = np.arange(len(IT)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
 
# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth, 
        edgecolor ='black', label =r'$k = 12$') 
plt.bar(br2, ECE, color ='g', width = barWidth, 
        edgecolor ='black', label =r'$k = 8$') 
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

plt.savefig('figures_paper/fig_speed_test.png', dpi = 300, bbox_inches='tight', transparent = True)