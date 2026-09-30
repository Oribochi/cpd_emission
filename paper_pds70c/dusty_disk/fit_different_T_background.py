# Using Nautilus nested sampling to obtain the best fit parameters for the zhu model given by our real data

import numpy as np
import matplotlib.pyplot as plt
from flux_dusty_disk import F # my flux function
from nautilus import Prior
from units_astro import *
from nautilus import Sampler
import corner
from numpy import exp, log, log10, sqrt
import smplotlib
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

nus=np.logspace(log10(nu[0]),log10(nu[-1]),100)

column_width=256.0748*0.0138888889
text_width=523.5307*0.0138888889


def fit(Tb):

    # we define the likelihood
    def likelihood_F(param_dict):
    # for a constant beta
        
        Mpdot = 10**param_dict['log_Mpdot']*Mj/yr
        alpha = 10**param_dict['log_alpha']


        ymodels = np.zeros(len(Fnus))
        Xis = np.zeros(len(Fnus))
        for i in range(len(Fnus)):
            ymodels[i] = F(10*c/nu[i],Mpdot, alpha, TISM=Tb)
            Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2
        
        return np.sum(Xis)


    prior = Prior()

    prior.add_parameter('log_Mpdot', dist=(-15, 15))
    prior.add_parameter('log_alpha', dist=(-15, 15))


    sampler = Sampler(prior, likelihood_F, n_live=1000, pool = 16)             
    sampler.run(verbose=True)
    
    points, log_w, log_l = sampler.posterior()
    # corner.corner(
    #     points, weights=np.exp(log_w), bins=20, labels=[r'$\log(M_{\rm tot}/M_{\rm j})$',  r'$\gamma$', r'$\beta$'], color='purple',
    #     plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

    # # Save the log Z and the corner plot
    # np.save('logZ_zhu.npy', log_l)
    # plt.savefig('corner_plot_Tb_'+str(Tb)+'.png', bbox_inches='tight')

    # the best fit parameters
    print('log(Z) =', log_l)
    print('Evidence =', np.exp(log_l))
    print('Best fit parameters =', np.mean(points, axis=0))
    print('Errors =', np.std(points, axis=0))
    print('The log likelihood is', likelihood_F({'log_Mpdot': np.mean(points, axis=0)[0], 'log_alpha': np.mean(points, axis=0)[1]}))
    print('The maximum likelihood parameters are', points[np.argmax(log_l)])
    print('The maximum likelihood is ', np.max(log_l))


    # text1=np.array(['log_Mp =', Mpdot_best,'+', Mpdot_max, '-', -Mpdot_min, 'Mj', '\n', 'gamma =', gamma_best,'+', gamma_max, '-', -gamma_min, '\n', 'beta =', beta_best,'+', beta_max, '-',
    # -beta_min])

    # # save the text file
    # np.savetxt('param_fit_Tb_'+str(Tb)+'.txt', text1)
    Mpdot = 10**points[np.argmax(log_l)][0]*Mj/yr
    alpha = 10**points[np.argmax(log_l)][1]

    fluxes = np.zeros(len(nu))
    for i in range(len(nu)):
        fluxes[i]=F(10*c/nu[i], Mpdot, alpha, TISM=Tb)
        print('Flux at', round(nus[i]/1E9, 1), 'GHz =', fluxes[i])
        if i>0:
            print('Spectral index is ', (log(fluxes[i])-log(fluxes[i-1]))/(log(nu[i])-log(nu[i-1])))

    flux_arr = np.zeros(len(nus))

    for i in range(len(nus)):
        flux_arr[i]=F(10*c/nus[i], Mpdot, alpha, TISM=Tb)

    #size box

    
    # vs the real data and the corner in the same plot
    # fig = plt.figure(layout='constrained', figsize=(text_width, text_width/(2.5)), dpi=300)
    # subfigs = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.15)

    # ax1 = subfigs[0].subplots(1, 1, sharey=True)
    # ax1.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
    # #ax1.plot(nu_arr/10**9, F_arr_ff, label='Free-free', linestyle=':', color='black')
    # #ax1.plot(nu_arr/10**9, F_arr_zhu, label='Dust',linestyle='--', color='green')
    # ax1.plot(nu_arr/10**9, flux_arr, label='Model', alpha=0.5, color='red')
    # ax1.loglog()
    # ax1.tick_params(axis='both', which='minor', labelsize=6)
    # ax1.tick_params(axis='both', which='major', labelsize=6)
    # ax1.set_xlabel(r'$\nu$ (GHz)', fontsize=8)
    # ax1.set_ylabel(r'$F_{\nu}$ (mJy)', fontsize=8)
    # ax1.legend(loc='upper left', fontsize=8)


    # ax2 = subfigs[1]

    # corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log(M_{\rm tot}/M_{\rm j})$', r'$\gamma$', r'$\beta$'], color='purple',
    #     plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=ax2, label_kwargs={'fontsize': 8})

    # for axs in ax2.get_axes():
    #     axs.tick_params(axis='both', labelsize=6)

    # plt.savefig('fig_envelope_fit_Tb_'+str(Tb)+'.png', bbox_inches='tight')
    
    # #plot
    # plt.plot(nu, [R1,R2,R3,R4], color='black', marker='o', linestyle='None', label='Data')
    # plt.plot(nu, R1*nu/nu[0], color='red', linestyle='--', label='f(x)=x')
    # plt.plot(nu, [Rin/au]*4, color='blue', linestyle='--', label='Rin')
    # plt.loglog()
    # plt.legend()
    # plt.show()
    # penalty=mass_envelope(rho_0, beta)/(Mj)
    # print('Penalty =', penalty)
    # print('T1 =', str(T_profile(R1*au, T_0, gamma)))
    # print('T2 =', str(T_profile(R2*au, T_0, gamma)))
    # print('T3 =', str(T_profile(R3*au, T_0, gamma)))
    # print('T4 =', str(T_profile(R4*au, T_0, gamma)))
    # save text file with radius and temperature

    return flux_arr

flux_0=fit(0)
flux_10=fit(10)
flux_15=fit(15)
flux_20=fit(20)


fig = plt.figure(layout='constrained', figsize=(4.8, 3.8), dpi=300)
plt.errorbar(np.array(nu)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='C0')
#plt.plot(nus/10**9, flux_0, label=r'$T_{\rm floor}=0 ~ \rm K$', color='C1')
plt.plot(nus/10**9, flux_10, label=r'$T_{\rm floor}=10 ~ \rm K$', color='C1')
plt.plot(nus/10**9, flux_15, label=r'$T_{\rm floor}=15 ~ \rm K$', color='C2')
plt.plot(nus/10**9, flux_20, label=r'$T_{\rm floor}=20 ~ \rm K$', color='C3')
plt.loglog()
plt.xlabel(r'$\nu$ [GHz]')
plt.ylabel(r'$F_{\nu}$ [mJy]')
plt.legend(loc='best')
plt.savefig('all_fit.png',bbox_inches='tight', dpi=300)
plt.show()
