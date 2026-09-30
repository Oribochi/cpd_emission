# Code to generate cornerplots 
import matplotlib.pyplot as plt
import numpy as np
import corner
from units_astro import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from numpy import log10
plt.style.use('tableau-colorblind10')

# now making the plot in the upper right corner bigger
# Main figure and grid

# This function receives the points, log_w, nu_arr, F_arr_ff, Fnus, sFnus and makes the corner plot in the bottom left corner 
# and the function plot in the upper right corner with the data points and error bars
def fig_corner(points, log_w, nu_arr, F_arr_ff, nus, Fnus, sFnus, labels=['a', 'b'], model='Model', filename='cornerplot.pdf'):
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = GridSpec(3, 3, figure=fig)

    # Create nested 2x2 GridSpec for the corner plot
    corner_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1:, :2])

    # Allocate all 2x2 axes, even if some stay unused
    corner_axes = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            corner_axes[i, j] = fig.add_subplot(corner_spec[i, j])

    # Flatten and pass to corner
    corner.corner(points, weights=np.exp(log_w), fig=fig, color='C1', bins=20, range=np.repeat(0.999, 2),
                labels=labels, plot_datapoints=False, labelpad=0.05,
                axes=corner_axes.flatten().tolist(), label_kwargs={'fontsize': 14})  # Must be a flat list)

    yerr_log = np.array([[log10(Fnus[i]) - log10(Fnus[i] - sFnus[i]) for i in range(len(Fnus))],[log10(Fnus[i] + sFnus[i]) 
    - log10(Fnus[i]) for i in range(len(Fnus))]])

    # Another plot in top-right 2 rows of last column
    ax_function = fig.add_subplot(gs[:2, 1:])
    ax_function.errorbar(np.array(nus)/1E9, log10(Fnus), yerr=yerr_log, fmt='o', color='C0', markersize=5, label='Data')
    ax_function.plot(nu_arr/10**9, log10(F_arr_ff), label=model, color='C1')
    ax_function.set_xscale('log')
    ax_function.set_xlabel(r'$\nu/{\rm GHz}$', labelpad=5, fontsize=14)
    ax_function.set_ylabel(r'$\log(F_{\nu}/ {\rm mJy})$', labelpad=5, fontsize=14)
    ax_function.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=12)
    ax_function.tick_params(axis='both', which='both', labelsize=12)
    ax_function.xaxis.tick_top()
    ax_function.yaxis.tick_right()

    ax_function.tick_params(axis='y', direction='in', pad=0, labelsize=12)
    # ax_function.set_yticks([-5,-3, -1, 1])
    # ax_function.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax_function.get_yaxis().set_minor_formatter(plt.NullFormatter())
    # ax_function.set_xticks([100, 300, 700])
    # Remove bottom and left ticks
    ax_function.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_function.get_xaxis().set_minor_formatter(plt.NullFormatter())
    # Custom ticks (ensures you get what you want)
    ax_function.legend(loc='best', fontsize=12)
    ax_function.xaxis.set_label_position('top') 
    ax_function.yaxis.set_label_position('right')
    # now the bottom left corner
    # corner_axes[1, 0].set_xticks([-5.8,-5.7,-5.6,-5.5])
    corner_axes[1, 0].tick_params(axis='both',labelsize=12)
    corner_axes[1, 1].tick_params(axis='x', labelsize=12)
    corner_axes[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    corner_axes[1, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())

    # Hide or use bottom-right slot
    ax_unused = fig.add_subplot(gs[2, 2])
    ax_unused.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


# now the same but with two cornerplots in the same figure
# This function receives a list of two sets of points, log_w, nu_arr, F_arr_ff, Fnus, sFnus and makes the corner plot in the bottom left corner 
# and the function plot in the upper right corner with the data points and error bars
def fig_2corner(arr_points, arr_log_w, arr_nu_arr, arr_F_arr_ff, arr_nus, arr_Fnus, arr_sFnus, labels=['a', 'b'], models=['Model1', 'Model2'], filename='2cornerplots.pdf'):
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = GridSpec(3, 3, figure=fig)

    # Create nested 2x2 GridSpec for the corner plot
    corner_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1:, :2])

    # Allocate all 2x2 axes, even if some stay unused
    corner_axes = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            corner_axes[i, j] = fig.add_subplot(corner_spec[i, j])

    points1=arr_points[0]
    log_w1=arr_log_w[0]
    nu_arr1=arr_nu_arr[0]
    F_arr_ff1=arr_F_arr_ff[0]
    nus1=arr_nus[0]
    Fnus1=arr_Fnus[0]
    sFnus1=arr_sFnus[0]

    points2=arr_points[1]
    log_w2=arr_log_w[1]
    nu_arr2=arr_nu_arr[1]
    F_arr_ff2=arr_F_arr_ff[1]
    nus2=arr_nus[1]
    Fnus2=arr_Fnus[1]
    sFnus2=arr_sFnus[1]

    # makes the first corner plot in the bottom left corner

    corner.corner(points1, weights=np.exp(log_w1), fig=fig, color='C1', bins=20, range=[(-10, -4.5), (-10, 0)],
                labels=labels, plot_datapoints=False, labelpad=0.05,
                axes=corner_axes.flatten().tolist(), label_kwargs={'fontsize': 14})  # Must be a flat list)

    # makes the second corner plot in the bottom left corner
    corner.corner(points2, weights=np.exp(log_w2), fig=fig, color='C3', bins=20, range=[(-10, -4.5), (-10, 0)],
                labels=labels, plot_datapoints=False, labelpad=0.05,
                axes=corner_axes.flatten().tolist(), label_kwargs={'fontsize': 14})  
    
    # makes the error bars for the first corner plot
    yerr_log1 = np.array([[log10(Fnus1[i]) - log10(Fnus1[i] - sFnus1[i]) for i in range(len(Fnus1))],[log10(Fnus1[i] + sFnus1[i]) 
    - log10(Fnus1[i]) for i in range(len(Fnus1))]])

    # makes the error bars for the second corner plot
    yerr_log2 = np.array([[log10(Fnus2[i]) - log10(Fnus2[i] - sFnus2[i]) for i in range(len(Fnus2))],[log10(Fnus2[i] + sFnus2[i]) 
    - log10(Fnus2[i]) for i in range(len(Fnus2))]])

    # Another plot in top-right 2 rows of last column
    ax_function = fig.add_subplot(gs[:2, 1:])
    ax_function.errorbar(np.array(nus1)/1E9, log10(Fnus1), yerr=yerr_log1, fmt='o', color='C0', markersize=5)
    ax_function.plot(nu_arr1/10**9, log10(F_arr_ff1), label=models[0], color='C1')
    ax_function.errorbar(np.array(nus2)/1E9, log10(Fnus2), yerr=yerr_log2, fmt='o', color='C2', markersize=5)
    ax_function.plot(nu_arr2/10**9, log10(F_arr_ff2), label=models[1], color='C3')
    ax_function.set_xscale('log')
    ax_function.set_xlabel(r'$\nu/{\rm GHz}$', labelpad=5, fontsize=14)
    ax_function.set_ylabel(r'$\log(F_{\nu}/ {\rm mJy})$', labelpad=5, fontsize=14)
    ax_function.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=12)
    ax_function.tick_params(axis='both', which='both', labelsize=12)
    ax_function.xaxis.tick_top()
    ax_function.yaxis.tick_right()

    ax_function.tick_params(axis='y', direction='in', pad=0, labelsize=12)
    ax_function.set_yticks([-5,-3, -1, 1])
    ax_function.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax_function.get_yaxis().set_minor_formatter(plt.NullFormatter())
    # Remove bottom and left ticks
    ax_function.set_xticks([100, 300, 700])
    ax_function.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_function.get_xaxis().set_minor_formatter(plt.NullFormatter())
    # Custom ticks (ensures you get what you want)
    ax_function.legend(loc='best', fontsize=12, frameon=False)
    ax_function.xaxis.set_label_position('top') 
    ax_function.yaxis.set_label_position('right')
    # now the bottom left corner
    corner_axes[1, 0].set_xticks([-9,-7,-5])
    corner_axes[1, 0].tick_params(axis='both',labelsize=12)
    corner_axes[1, 1].tick_params(axis='x', labelsize=12)
    corner_axes[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    corner_axes[1, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())

    # Hide or use bottom-right slot
    ax_unused = fig.add_subplot(gs[2, 2])
    ax_unused.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
#
#
# # Example
# log_w=np.load('logW1.npy')
# points=np.load('logP1.npy')
# nu_arr, F_arr_ff = np.loadtxt('fluxes_zhu.txt', unpack=True)
# # First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
# nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
# Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
# sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# # To mJy
# nu = np.array(nu)
# Fnus = np.array(Fnus)*1E3
# sFnus = np.array(sFnus)*1E3
# fig_corner(points, log_w, nu_arr, F_arr_ff, nu, Fnus, sFnus, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', 
# r'$\log(\alpha)$'], model='Dusty disc', filename='cornerplot_dusty_disk.pdf')

# Example 2

from flux_dusty_disk import F as F_zhu

# Ṁp = 10−9 MJup yr−1 and α = 10−3
Mpdot1=1e-9*Mj/yr
alpha1=1e-3

nus1=np.array([97.5E9, 145005402197.7, 343.5E9, 671E9])
Fnus1=np.array([F_zhu(10*c/nu, Mpdot1, alpha1) for nu in nus1])
sFnus1=0.1*Fnus1 # 10% error

# Now the same but with Ṁp = 10−5 MJup yr−1 and α = 10−3
Mpdot2=1e-5*Mj/yr
alpha2=1e-3

nus2=np.array([97.5E9, 145005402197.7, 343.5E9, 671E9])
Fnus2=np.array([F_zhu(10*c/nu, Mpdot2, alpha2) for nu in nus2])
sFnus2=0.1*Fnus2 # 10% error


log_w1=np.load('logW1_zhu.npy')
points1=np.load('logP1_zhu.npy')
nu_arr1, F_arr_ff1 = np.loadtxt('fluxes_zhu.txt', unpack=True)

log_w2=np.load('logW2_zhu.npy')
points2=np.load('logP2_zhu.npy')
nu_arr2, F_arr_ff2 = np.loadtxt('fluxes2_zhu.txt', unpack=True)

fig_2corner(arr_points=[points1, points2], arr_log_w=[log_w1, log_w2], arr_nu_arr=[nu_arr1, nu_arr2], arr_F_arr_ff=[F_arr_ff1, F_arr_ff2], arr_nus=[nus1, nus2], arr_Fnus=[Fnus1, Fnus2], arr_sFnus=[sFnus1, sFnus2], labels=[r'$\log\left( \frac{dM_{\rm p}/dt}{M_{\rm Jup}/{\rm yr}} \right)$', r'$\log(\alpha)$'], models=['Optically thin', 'Optically thick'], filename='cornerplot_degeneracy.pdf')