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

# make plot latex
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 9})
rc('text', usetex=True)

# Size in centimeters
width_cm = 7.2
height_cm = 3.5

# Convert size from centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54

plt.figure(figsize=(width_in, height_in))

# plt.figure(figsize=(11,3))

# fig, axs = plt.subplots(1, 2, figsize=(width_in, height_in), dpi = 300)

df_Tene_ratios = pd.read_csv('BI_He_Tene_728_706_728_668.csv')

Te_new = df_Tene_ratios['Te'].values
ne_new = df_Tene_ratios['ne'].values
R_728_706 = df_Tene_ratios['728/706'].values
R_728_668 = df_Tene_ratios['728/668'].values

Te_arr = Te_new.reshape((52,40))

ne_arr = ne_new.reshape((52,40))

He728_706_arr = R_728_706.reshape((52,40))
He728_668_arr = R_728_668.reshape((52,40))


fontsize_labels = 10

levels_plot_1 = np.geomspace(0.04, 1.7, num = 100)

levels_plot_2 = np.geomspace(0.08, 1.6, num = 100)

levels_plot_better = np.geomspace(0.04, 1.7, num = 200)

te = np.linspace(0.7, 40, 100)

ne = np.linspace(1e18, 1e20, 100)

X, Y = np.meshgrid(te, ne)

Z = np.zeros_like(X)

# levels_plot = np.geomspace(0.04, 1.5, num = 100)
# 0.04, 1.64

# 0.08, 1.68

# plt.subplot(1, 2, 1)

gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.1)

ax1 = plt.subplot(gs[0])

plt.title(r'${\rm He}^{\rm LRT}_{728/706}$', fontsize=fontsize_labels + 1)

plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
plt.ylabel(r'$n_e \: [{m\rm }^{-3}]$', fontsize=fontsize_labels)


# get the min and max value from levels_plot_1 and levels_plot_2
min_value = np.min([np.min(levels_plot_1), np.min(levels_plot_2)])
max_value = np.max([np.max(levels_plot_1), np.max(levels_plot_2)])

norm = LogNorm(vmin=min_value, vmax=max_value)
# norm = Normalize(vmin=min_value, vmax=max_value)
cp = ax1.contourf(Te_arr, ne_arr, He728_706_arr, levels = levels_plot_better, cmap = 'jet', norm = LogNorm())
#cp.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
plt.grid(alpha = 0.3)
# plt.colorbar(cp, format=tick.FormatStrFormatter('$%.2f$'))
plt.xscale('log')
plt.yscale('log')

# plt.subplot(1, 2, 2)
ax2 = plt.subplot(gs[1])

# ax2.set_yticklabels([' ', ' '])
ax2.yaxis.set_tick_params(labelleft=False)

plt.title(r'${\rm He}^{\rm LRT}_{728/668}$', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
#plt.ylabel('$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = ax2.contourf(Te_arr, ne_arr, He728_668_arr, levels = levels_plot_better, cmap = 'jet', norm = norm)
plt.grid(alpha = 0.3)


# sm = ScalarMappable(cmap='jet', norm=norm)
# sm.set_array([])

# def format_tick(x, pos):
#     return '$%.2f$' % x

# cbar = plt.colorbar(sm)
# cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_tick))


# plt.colorbar(sm, format=tick.FormatStrFormatter('$%.2f$'))

#cbar = plt.colorbar(cp,format=tick.FormatStrFormatter('$%.2f$'))#, ticks = [0.04, 0.1, 0.2, 0.5, 1.0, 1.5])
#cbar.ax.tick_params(labelsize=fontsize_labels - 2)

plt.xscale('log')
plt.yscale('log')
# take out the y tick labels

cax = plt.subplot(gs[2])
cbar = plt.colorbar(cp, cax=cax, format=tick.FormatStrFormatter('$%.2f$'))
cbar.ax.tick_params(labelsize=fontsize_labels - 2)



plt.tight_layout()

# plt.suptitle(r'${\rm Goto-CRM}, \: B_0 = 1.4 \: [{\rm T}]$', fontsize=fontsize_labels+2)

plt.savefig('figures_paper/ratios_He_evolve.png', dpi = 300, bbox_inches='tight', transparent = True)
#plt.show()


