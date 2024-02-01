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
import matplotlib.patches as mpatches

# make plot latex
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 8})
rc('text', usetex=True)

# Size in centimeters
width_cm = 7.2
height_cm = 3.5

# Convert size from centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54

fig = plt.figure(figsize=(width_in, height_in))

# plt.figure(figsize=(11,3))

# fig, axs = plt.subplots(1, 2, figsize=(width_in, height_in), dpi = 300)

# df_Tene_ratios = pd.read_csv('BI_He_Tene_728_706_728_668.csv')
df_Tene_ratios = pd.read_csv('data_figure_He_ratios.csv')

Te_new = df_Tene_ratios['Te'].values
ne_new = df_Tene_ratios['ne'].values
R_728_706 = df_Tene_ratios['728/706'].values
R_728_668 = df_Tene_ratios['728/668'].values

Te_arr = Te_new.reshape((200,200))

ne_arr = ne_new.reshape((200,200))

He728_706_arr = R_728_706.reshape((200,200))
He728_668_arr = R_728_668.reshape((200,200))

# figure the min and max values among both ratios
# to use in the contour plots





fontsize_labels = 9

levels_plot_1 = np.geomspace(0.04, 1.7, num = 100)

levels_plot_2 = np.geomspace(0.08, 1.6, num = 100)

levels_plot_better = np.geomspace(0.04, 1.7, num = 200)



# highlight TCV operational range

len_arr_fill = 100

Te_tcv = np.linspace(1.0, 80, len_arr_fill)
ne_tcv = np.linspace(1e18, 4e19, len_arr_fill)

Te_tcv_fill  = np.ones(len_arr_fill**2) 
ne_tcv_fill  = np.ones(len_arr_fill**2)

for i in range(len_arr_fill):
    Te_tcv_fill[i*len_arr_fill:(i+1)*len_arr_fill] = Te_tcv
    ne_tcv_fill[i*len_arr_fill:(i+1)*len_arr_fill] = ne_tcv[i]

Te_tcv_fill_2 = np.array([1.5, 1.3, 0.7, 1.3, 2.1, 8.0, 80.0, 40.0, 1.5])
ne_tcv_fill_2 = np.array([1e18, 2e18, 8e18, 3.5e19, 4e19, 4e19, 8e18, 1.3e18, 1e18])

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

min_value = np.min([np.min(He728_706_arr), np.min(He728_668_arr)])
max_value = np.max([np.max(He728_706_arr), np.max(He728_668_arr)])

levels_plot_better = np.geomspace(min_value, max_value, num = 200)

norm = LogNorm(vmin=min_value, vmax=max_value)
# norm = Normalize(vmin=min_value, vmax=max_value)
cp = ax1.contourf(Te_arr, ne_arr, He728_706_arr, levels = levels_plot_better, cmap = 'jet', norm = LogNorm())
#cp.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
plt.grid(alpha = 0.3)

# TCV params
# plt.fill(Te_tcv_fill_2, ne_tcv_fill_2, color='gray', alpha=0.1, edgecolor='black')
# plt.plot(Te_tcv_fill_2, ne_tcv_fill_2, '--w', linewidth=1.2)

plt.fill(Te_tcv_fill_2, ne_tcv_fill_2, color='gray', alpha=0.05, edgecolor='black', hatch='//')
plt.plot(Te_tcv_fill_2, ne_tcv_fill_2, color='black', label = 'TCV SOL')

# put legend on specific location
#plt.legend(loc='upper left', bbox_to_anchor=(0.05, 1.025), fontsize=6, framealpha=0.9)

# plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.025), fontsize=6, framealpha=0.9)

# set limits
plt.xlim([np.min(Te_arr), np.max(Te_arr)])
plt.ylim([np.min(ne_arr), np.max(ne_arr)])
plt.ylim([8e17, np.max(ne_arr)])

# plt.colorbar(cp, format=tick.FormatStrFormatter('$%.2f$'))
plt.xscale('log')
plt.yscale('log')

# put text in the plot
#plt.text(0.05, 0.90, 'TCV SOL -', transform=ax1.transAxes, fontsize=8, color='white')

# plt.subplot(1, 2, 2)
ax2 = plt.subplot(gs[1])

# ax2.set_yticklabels([' ', ' '])
ax2.yaxis.set_tick_params(labelleft=False)

plt.title(r'${\rm He}^{\rm LRT}_{728/668}$', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
#plt.ylabel('$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = ax2.contourf(Te_arr, ne_arr, He728_668_arr, levels = levels_plot_better, cmap = 'jet', norm = norm)
plt.grid(alpha = 0.3)

# TCV params
plt.fill(Te_tcv_fill_2, ne_tcv_fill_2, color='gray', alpha=0.05, edgecolor='black', hatch='//')
plt.plot(Te_tcv_fill_2, ne_tcv_fill_2, color='black')

# set limits 
plt.xlim([np.min(Te_arr), np.max(Te_arr)])
plt.ylim([np.min(ne_arr), np.max(ne_arr)])
plt.ylim([8e17, np.max(ne_arr)])
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


# Create a new axes for the legend
legend_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
legend_ax.axis('off')
line1 = mpatches.Patch(color='black', label='TCV SOL parameters', linewidth=0.5, linestyle='-')
# Add the legend to the new axes
legend_ax.legend(handles=[line1], loc='center', bbox_to_anchor=(0.5, 0.885), fontsize = 6, framealpha=0.95)


plt.tight_layout()

# plt.suptitle(r'${\rm Goto-CRM}, \: B_0 = 1.4 \: [{\rm T}]$', fontsize=fontsize_labels+2)

plt.savefig('figures_paper/ratios_He_evolve.png', dpi = 600, bbox_inches='tight', transparent = True)
#plt.show()

levels_He = np.geomspace(0.001, 32.0, num = 200)

levels_He = np.linspace(0.001, 10.0, num = 200)

# plot the ratio of the ratios

plt.figure(figsize=(4,3))

plt.title(r'$ {\rm He}^{\rm LRT}_{728/668} / {\rm He}^{\rm LRT}_{728/706}$', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
plt.ylabel('$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = plt.contourf(Te_arr, ne_arr, 2*Te_arr*ne_arr, levels = 100, cmap = 'jet')
plt.grid(alpha = 0.3)
plt.colorbar(cp,format=tick.FormatStrFormatter('$%.2f$'))

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()

plt.figure(figsize=(4,3))

plt.title(r'$ {\rm He}^{\rm LRT}_{728/668} / {\rm He}^{\rm LRT}_{728/706}$', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
plt.ylabel('$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = plt.contourf(Te_arr, ne_arr, He728_668_arr/He728_706_arr, levels = levels_He, cmap = 'jet')
plt.grid(alpha = 0.3)
plt.colorbar(cp,format=tick.FormatStrFormatter('$%.2f$'))

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
# plt.show()

plt.figure(figsize=(4,3))

plt.title(r'${\rm He}^{\rm LRT}_{728/706} / {\rm He}^{\rm LRT}_{728/668} $', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
plt.ylabel(r'$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = plt.contourf(Te_arr, ne_arr, He728_706_arr/He728_668_arr, levels = levels_He, cmap = 'jet')
plt.grid(alpha = 0.3)
plt.colorbar(cp,format=tick.FormatStrFormatter('$%.2f$'))

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
# plt.show()

plt.figure(figsize=(4,3))

plt.title(r'$2 * {\rm He}^{\rm LRT}_{728/706} / {\rm He}^{\rm LRT}_{728/668} $', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
plt.ylabel(r'$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = plt.contourf(Te_arr, ne_arr, (2*He728_706_arr)/He728_668_arr, levels = levels_He, cmap = 'jet')
plt.grid(alpha = 0.3)
plt.colorbar(cp,format=tick.FormatStrFormatter('$%.2f$'))

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
# plt.show()

plt.figure(figsize=(4,3))

plt.title(r'${\rm He}^{\rm LRT}_{728/668} / 2 * {\rm He}^{\rm LRT}_{728/706}  $', fontsize=fontsize_labels + 1)
plt.xlabel(r'$T_e \: [{ \rm eV}]$', fontsize=fontsize_labels)
plt.ylabel(r'$n_e$ [m$^{-3}$]', fontsize=fontsize_labels)
cp = plt.contourf(Te_arr, ne_arr, He728_668_arr/ (2*He728_706_arr), levels = levels_He, cmap = 'jet')
plt.grid(alpha = 0.3)
plt.colorbar(cp,format=tick.FormatStrFormatter('$%.2f$'))

plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()


