# Artificial case
# Using Nautilus nested sampling to obtain the best fit parameters for the Zhu model given by our real data

import numpy as np
import matplotlib.pyplot as plt
from dustyCPD import * # my flux function
from flux_dusty_disk import F, F_thin, S
from nautilus import Prior
from units_astro import *
from nautilus import Sampler
import corner
from numpy import exp, log, log10, sqrt
import smplotlib
import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

Mpdot1=1E-9*Mj/yr
alpha1=1e-3

# To mJy
nus = np.array(nu)
Fnus = np.array(F(10*c/nus, Mpdot1, alpha1))# [F1, F2] are the y-axis values
sFnus = Fnus/10 # [sF1, sF2] are the error bars

# To mJy
Fnus1 = Fnus
sFnus1 = sFnus


array_nu=np.logspace(log10(nu[0]),log10(nu[-1]),100)

# Now with the model of F that depends oof Mpdot and alpha we fit the likelihood
prior = Prior()
prior.add_parameter('log_Mpdot', dist=(-10, -4.5))
prior.add_parameter('log_alpha', dist=(-8, 0))

def likelihood_F(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha)
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

sampler_zhu = Sampler(prior, likelihood_F, n_live=1000, pool = 16)             
sampler_zhu.run(verbose=True)


# Hacemos el corner plot
figure1, axes = plt.subplots(2, 2, figsize=(4.5, 4.5))

points, log_w, log_l = sampler_zhu.posterior()

Mpdot=10**points[np.argmax(log_l)][0]*Mj/yr
alpha=10**points[np.argmax(log_l)][1]

fluxes1 = np.zeros(len(array_nu))
for i in range(len(array_nu)):
    fluxes1[i]=F(10*c/array_nu[i], Mpdot, alpha)

corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$',r'$\log(\alpha)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=figure1, labelpad=0.05) #, label_kwargs={'fontsize': 20}

# adding the data points in the corner plot
axes[0, 1].errorbar(nus/1E9,
                    Fnus1,
                    yerr=sFnus1,
                    fmt='o', color='C0', markersize=3, alpha=0.5)
axes[0, 1].plot(array_nu/1E9, fluxes1, color='C1', label='Optically thin', alpha=0.5)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel(r'$\nu/{\rm GHz}$', labelpad=-10, fontsize=10)
axes[0, 1].set_ylabel(r'$F_{\nu}/{\rm 0.1 \mu Jy}$', labelpad=-10, fontsize=10)
axes[0, 1].tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=8)
axes[0, 1].tick_params(axis='both', which='both', labelsize=8)

axes[0, 1].xaxis.tick_top()
axes[0, 1].yaxis.tick_right()
axes[0, 1].tick_params(axis='y', direction='in', pad=5, labelsize=8)
# Remove bottom and left ticks
axes[0, 1].tick_params(bottom=False, left=False)
axes[0, 1].set_xticks([100, 300, 700])
axes[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[0, 1].get_xaxis().set_minor_formatter(plt.NullFormatter())
# Custom ticks (ensures you get what you want)
axes[0, 1].legend(loc='upper left', frameon=True, framealpha=0.8, fontsize=10)

plt.savefig('corner_plot_synthetic1.png', bbox_inches='tight', dpi=300)

# axes[0, 1].yaxis.tick_right()
# axes[0, 1].tick_params(axis='y', direction='in', pad=0, labelsize=10)
# axes[0, 1].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, 
#                        labeltop=True, labelright=True, labelsize=10)

# # the best fit parameters
# print('log(Z) =', log_l)
# print('Evidence =', np.exp(log_l))9
# print('Best fit parameters =', np.mean(points, axis=0))
# print('Errors =', np.std(points, axis=0))
# print('The log likelihood is', likelihood_F({'log_Mpdot': np.mean(points, axis=0)[0], 'log_alpha': np.mean(points, axis=0)[1]}))
# print('The maximum likelihood parameters are', points[np.argmax(log_l)])
# print('The maximum likelihood is ', np.max(log_l))

# # vs the real data
# # now making our spectral index
# Mpdot=10**points[np.argmax(log_l)][0]*Mj/yr
# alpha=10**points[np.argmax(log_l)][1]

# fluxes1 = np.zeros(len(nus))
# for i in range(len(nus)):
#     fluxes1[i]=F(10*c/nus[i], Mpdot, alpha)

# print(fluxes[-1])
# # to inches the column and the text width
# column_width=256.0748*0.0138888889
# text_width=523.5307*0.0138888889

# # vs the real data
# fig = plt.figure(layout='constrained', figsize=(column_width, column_width/(1.3)), dpi=300)
# plt.errorbar(np.array(nu)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
# plt.plot(np.array(nus)/10**9, fluxes, label='Best fit', alpha=0.5, color='red')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=8)
# plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=8)
# plt.xticks(fontsize=6, minor=True)
# plt.yticks(fontsize=6)
# plt.xticks(fontsize=6, minor=False)
# plt.legend(loc='upper left', fontsize=8)
# plt.show()


# # profiles
# T_ext_arr=[T_ext(r, Mpdot) for r in R]
# T_eff_arr=[T_eff(r,Mpdot) for r in R]
# T_visc_arr=[T_visc(r, Sigma(r, Mpdot, alpha), Mpdot) for r in R]
# T_c_arr=[T_c_approx(r, Mpdot, alpha) for r in R]

# plt.plot(R/au, T_ext_arr)
# plt.plot(R/au, T_eff_arr)
# plt.plot(R/au, T_visc_arr)
# plt.plot(R/au, T_c_arr)
# plt.ylim(10, 5*np.max(T_c_arr))
# plt.loglog()
# plt.legend(['T_ext', 'T_eff', 'T_visc','T_c'])
# plt.show()

# # and the flux
# S_arr=[S(10*c/nu[-1], r, Mpdot, alpha) for r in R]

# S_arr_R=np.array(S_arr)*R**2
# plt.plot(R/au,S_arr_R)
# plt.loglog()
# plt.ylim(np.max(S_arr_R)*1E-10, np.max(S_arr_R)*5)
# plt.show()

# # # Now the same but with Mpdot =1E-5 and alpha = 1E-2

Mpdot2=1E-5*Mj/yr
alpha2=1E-3

nus = np.array(nu)
Fnus2 = np.array(F(10*c/nus, Mpdot2, alpha2))# [F1, F2] are the y-axis values
sFnus2 = Fnus2/10 # [sF1, sF2] are the error bars


prior = Prior()
prior.add_parameter('log_Mpdot', dist=(-10, -4.5))
prior.add_parameter('log_alpha', dist=(-8, 0))

def log_likelihood(param_dict): 
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus2))
    Xis = np.zeros(len(Fnus2))
    for i in range(len(Fnus2)):
        ymodels[i] = F(10*c/nu[i], Mpdot*Mj/yr, alpha)
        Xis[i] = -0.5*((Fnus2[i]-ymodels[i])/sFnus2[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

sampler_zhu2 = Sampler(prior, log_likelihood, n_live=1000, pool = 16)
sampler_zhu2.run(verbose=True)

# to inches the column and the text width
column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889

# Hacemos el corner plot
figure2, axes2 = plt.subplots(2, 2, figsize=(4.5, 4.5))
points2, log_w2, log_l2 = sampler_zhu2.posterior()
corner.corner(points2, weights=np.exp(log_w2), bins=20, labels=[r'$\log \left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$',r'$\log(\alpha)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=figure2, labelpad=0.05) # , label_kwargs={'fontsize': 20}

fluxes2 = np.zeros(len(array_nu))
for i in range(len(array_nu)):
    fluxes2[i] = F(10*c/array_nu[i], 10**points2[np.argmax(log_l2)][0]*Mj/yr, 10**points2[np.argmax(log_l2)][1])

# adding the data points in the corner plot
axes2[0, 1].errorbar(nus/1E9,
                    Fnus2,
                    yerr=sFnus2,
                    fmt='o', color='C0', markersize=3, alpha=0.5)
axes2[0, 1].plot(array_nu/1E9, fluxes2, color='C1', label='Optically thick', alpha=0.5)
axes2[0, 1].set_xscale('log')
axes2[0, 1].set_yscale('log')
axes2[0, 1].set_xlabel(r'$\nu/{\rm GHz}$', labelpad=-10, fontsize=10)
axes2[0, 1].set_ylabel(r'$F_{\nu}/{\rm mJy}$', labelpad=-10, fontsize=10)
axes2[0, 1].tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=8)
axes2[0, 1].tick_params(axis='both', which='both', labelsize=8)

axes2[0, 1].xaxis.tick_top()
axes2[0, 1].yaxis.tick_right()
axes2[0, 1].tick_params(axis='y', direction='in', pad=0, labelsize=8)
# Remove bottom and left ticks
axes2[0, 1].tick_params(bottom=False, left=False)
axes2[0, 1].set_xticks([100, 300, 700])
axes2[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes2[0, 1].get_xaxis().set_minor_formatter(plt.NullFormatter())
# Custom ticks (ensures you get what you want)
axes2[0, 1].legend(loc='upper left', frameon=True, framealpha=0.8, fontsize=10)
plt.savefig('corner_plot_synthetic2.png', bbox_inches='tight', dpi=300)

plt.show()

# Now a corner plot with both
points1, log_w1, log_l1 = sampler_zhu.posterior()
points2, log_w2, log_l2 = sampler_zhu2.posterior()
data1 = points1
data2 = points2

# Create a figure with subplots
fig, axes3 = plt.subplots(2, 2, figsize=(4.5, 4.5))
# Plot the first corner plot
corner.corner(data1, weights=np.exp(log_w1), fig=fig, color='C1', bins=20, alpha=0.7, labels=[r'$\log \left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log(\alpha)$'],plot_datapoints=False, labelpad=0)

# Plot the second corner plot
corner.corner(data2, weights=np.exp(log_w2), fig=fig, color='C3',  bins=20, alpha=0.7, labels=[r'$\log \left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log(\alpha)$'],plot_datapoints=False, labelpad=0)

axes3[0, 1].errorbar(nus/1E9, log10(Fnus1), yerr=sFnus1/(log(10)*Fnus1), fmt='o', color='C0', markersize=3, alpha=0.5)
axes3[0, 1].plot(array_nu/1E9, log10(fluxes1), color='C1', label=r'Optically thin', alpha=0.7)
axes3[0, 1].errorbar(nus/1E9, log10(Fnus2), yerr=sFnus2/(log(10)*Fnus2), fmt='o', color='C2', markersize=3, alpha=0.5)
axes3[0, 1].plot(array_nu/1E9, log10(fluxes2), color='C3', label=r'Optically thick', alpha=0.7)
axes3[0, 1].set_xscale('log')
axes3[0, 1].set_xlabel(r'$\nu/{\rm GHz}$', labelpad=-10, fontsize=10)
axes3[0, 1].set_ylabel(r'$\log(F_{\nu}/ {\rm mJy})$', labelpad=-8, fontsize=10)
axes3[0, 1].tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=10)
axes3[0, 1].tick_params(axis='both', which='both', labelsize=10)

axes3[0, 1].xaxis.tick_top()
axes3[0, 1].yaxis.tick_right()
axes3[0, 1].tick_params(axis='y', direction='in', pad=0, labelsize=10)
axes3[0, 1].set_yticks([-5,-3, -1, 1])
axes3[0, 1].get_yaxis().set_major_formatter(plt.ScalarFormatter())
axes3[0, 1].get_yaxis().set_minor_formatter(plt.NullFormatter())
# Remove bottom and left ticks
axes3[0, 1].set_xticks([100, 300, 700])
axes3[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes3[0, 1].get_xaxis().set_minor_formatter(plt.NullFormatter())
# now the bottom left corner
axes3[1, 0].set_xticks([-10,-8,-6,-4])
axes3[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes3[1, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())

axes3[0, 1].tick_params(which='both', direction='in', labelsize=10)
axes3[0, 1].tick_params(bottom=False, left=False)
# Custom ticks (ensures you get what you want)
axes3[0, 1].legend(loc='best', fontsize=9)
plt.savefig('corner_plot_synthetic_both.png', bbox_inches='tight', dpi=300)
plt.show()


# now making the plot in the upper right corner bigger
# Main figure and grid
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
corner.corner(data1, weights=np.exp(log_w1), fig=fig, color='C1', bins=20, alpha=0.7, 
              labels=[r'$\log \left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log(\alpha)$'],plot_datapoints=False, labelpad=0.05,
              axes=corner_axes.flatten().tolist())  # Must be a flat list)

corner.corner(data2, weights=np.exp(log_w2), fig=fig, color='C3', bins=20, alpha=0.7,
                labels=[r'$\log \left(\dot{M}_{\rm p}/(M_{\rm Jup}/{\rm yr})\right)$', r'$\log(\alpha)$'],plot_datapoints=False, labelpad=0.05,
                axes=corner_axes.flatten().tolist())  # Must be a flat list
             
# Another plot in top-right 2 rows of last column
ax_function = fig.add_subplot(gs[:2, 1:])
ax_function.errorbar(nus/1E9, log10(Fnus1), yerr=sFnus1/(log(10)*Fnus1), fmt='o', color='C0', markersize=3, alpha=0.5)
ax_function.plot(array_nu/1E9, log10(fluxes1), color='C1', label=r'Optically thin', alpha=0.7)
ax_function.errorbar(nus/1E9, log10(Fnus2), yerr=sFnus2/(log(10)*Fnus2), fmt='o', color='C2', markersize=3, alpha=0.5)
ax_function.plot(array_nu/1E9, log10(fluxes2), color='C3', label=r'Optically thick', alpha=0.7)
ax_function.set_xscale('log')
ax_function.set_xlabel(r'$\nu/{\rm GHz}$', labelpad=5, fontsize=15)
ax_function.set_ylabel(r'$\log(F_{\nu}/ {\rm mJy})$', labelpad=5, fontsize=15)
ax_function.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=15)
ax_function.tick_params(axis='both', which='both', labelsize=15)
ax_function.xaxis.tick_top()
ax_function.yaxis.tick_right()

ax_function.tick_params(axis='y', direction='in', pad=0, labelsize=15)
ax_function.set_yticks([-5,-3, -1, 1])
ax_function.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax_function.get_yaxis().set_minor_formatter(plt.NullFormatter())
# Remove bottom and left ticks
ax_function.set_xticks([100, 300, 700])
ax_function.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_function.get_xaxis().set_minor_formatter(plt.NullFormatter())
# Custom ticks (ensures you get what you want)
ax_function.legend(loc='best', fontsize=15)
ax_function.xaxis.set_label_position('top') 
ax_function.yaxis.set_label_position('right')
# now the bottom left corner
corner_axes[1, 0].set_xticks([-9,-7,-5])
corner_axes[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
corner_axes[1, 0].get_xaxis().set_minor_formatter(plt.NullFormatter())
# Hide or use bottom-right slot
ax_unused = fig.add_subplot(gs[2, 2])
ax_unused.axis('off')
plt.savefig('corner_plot_synthetic_both_big.png', bbox_inches='tight', dpi=300)
plt.show()
# # the best fit parameters
# print('log(Z) =', log_l)
# print('Evidence =', np.exp(log_l))
# print('Best fit parameters =', np.mean(points, axis=0))
# print('Errors =', np.std(points, axis=0))
# print('The log likelihood is', log_likelihood({'log_Mpdot': np.mean(points, axis=0)[0], 'log_alpha': np.mean(points, axis=0)[1]}))
# print('The maximum likelihood parameters are', points[np.argmax(log_l)])
# print('The maximum likelihood is ', np.max(log_l))
# # vs the real data
# # now making our spectral index
# Mpdot=10**points[np.argmax(log_l)][0]*Mj/yr
# alpha=10**points[np.argmax(log_l)][1]

# fluxes2 = np.zeros(len(nus))
# for i in range(len(nus)):
#     fluxes2[i]=F(10*c/nus[i], Mpdot, alpha)
#     if i>0:
#         print(log10(fluxes2[i]/fluxes2[i-1])/log10(nus[i]/nus[i-1]))
# print(fluxes2[-1])

# # to inches the column and the text width
# column_width=256.0748*0.0138888889
# text_width=523.5307*0.0138888889
# # vs the real data
# fig = plt.figure(layout='constrained', figsize=(column_width, column_width/(1.3)), dpi=300)
# plt.errorbar(np.array(nu)/10**9, Fnus2, yerr=sFnus2, fmt='o', label='Data', color='blue')
# plt.plot(nus/10**9, fluxes2, label='Best fit', alpha=0.5, color='red')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=8)
# plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=8)
# plt.xticks(fontsize=6, minor=True)
# plt.yticks(fontsize=6)
# plt.xticks(fontsize=6, minor=False)
# plt.legend(loc='upper left', fontsize=8)
# plt.show()

# # profiles
# T_ext_arr2=[T_ext(r, Mpdot) for r in R]
# T_eff_arr2=[T_eff(r,Mpdot) for r in R]
# T_visc_arr2=[T_visc(r, Sigma(r, Mpdot, alpha), Mpdot) for r in R]
# T_c_arr2=[T_c_approx(r, Mpdot, alpha) for r in R]
# plt.plot(R/au, T_ext_arr2)
# plt.plot(R/au, T_eff_arr2)
# plt.plot(R/au, T_visc_arr2)
# plt.plot(R/au, T_c_arr2)
# plt.ylim(10, 5*np.max(T_c_arr2))
# plt.loglog()
# plt.legend(['T_ext', 'T_eff', 'T_visc','T_c'])
# plt.show()

# # and the flux
# S_arr2=[S(10*c/nu[-1], r, Mpdot, alpha) for r in R]
# S_arr_R2=np.array(S_arr2)*R**2
# plt.plot(R/au,S_arr_R2)
# plt.loglog()
# plt.ylim(np.max(S_arr_R2)*1E-10, np.max(S_arr_R2)*5)
# plt.show()