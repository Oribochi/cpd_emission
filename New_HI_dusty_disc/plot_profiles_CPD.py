# Plot profiles CPD
from profiles_cpd import profiles_ff
from units_astro import *
import numpy as np
from numpy import log10
import matplotlib.pyplot as plt
# now plot everything together
plt.style.use('tableau-colorblind10')
# now changing the stile not 10^n but 100, 1000, etc
from matplotlib.ticker import ScalarFormatter
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin
# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10,
# 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False 



def plot_profiles_cpd(Mpdot, alpha, filename='profiles_cpd.pdf'):
    profiles=profiles_ff(Mpdot, alpha)
    R_arr=profiles[0]
    f_arr=profiles[1]
    T_c_arr=profiles[2]
    ne_arr=profiles[3]
    sigma_arr=profiles[4]
    EM_arr=profiles[5]
    z_arr_mid_f=profiles[6]
    flux_anullus=profiles[7]

    fig, ax = plt.subplots(3, 2, figsize=(7.5, 4.5), sharex=True)
    # make space in the width between the subplots
    fig.subplots_adjust(wspace=0.3, hspace=0)

    # and put the x ticks in all the subplots to show the lines upward and downward

    for i in range(3):
        ax[i, 0].tick_params(axis='both', direction='in')
        ax[i, 1].tick_params(axis='both', direction='in')

    ax[0, 0].plot(R_arr/au, log10(f_arr), label=r'$f$')
    #ax[0, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 0].set_ylabel(r'$\log(f)$', fontsize=14)
    ax[0, 0].set_ylim(-4, 0.2)
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('linear')
    ax[0, 0].set_yticks([-3, -2, -1, 0])

    ax[0, 1].plot(R_arr/au, log10(flux_anullus), label=r'$n_e$')
    #ax[0, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 1].set_ylabel(r'$\log \left( \frac{F^{\rm annulus}_{671}}{\rm mJy} \right)$', fontsize=14)
    #ax[0, 1].set_ylim(np.max(log10(ne_arr))-4, np.max(log10(ne_arr))+0.2)
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('linear')
    ax[0, 1].set_yticks([-1, 0, 1])
    ax[0, 1].set_ylim(-1, np.max(log10(flux_anullus))+0.2)

    ax[1, 0].plot(R_arr/au, log10(sigma_arr), label=r'$\Sigma$')
    #ax[1, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[1, 0].set_ylabel(r'$\log \left( \frac{\Sigma}{\rm g ~ cm^{-2}} \right)$', fontsize=14)
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('linear')
    
    ax[1, 1].plot(R_arr/au, log10(T_c_arr), label=r'$T_c$')
    #ax[1, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[1, 1].set_ylabel(r'$\log \left( \frac{T_c}{\rm K} \right)$', fontsize=14)
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('linear')

    ax[2, 0].plot(R_arr/au, log10(EM_arr/pc), label=r'$EM$')
    ax[2, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 0].set_ylabel(r'$\log \left( \frac{\rm EM}{\rm cm^{-6} pc} \right)$', fontsize=14)
    ax[2, 0].set_ylim(np.max(log10(EM_arr/pc))-4, np.max(log10(EM_arr/pc))+0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('linear')
    

    ax[2, 1].plot(R_arr/au, log10(z_arr_mid_f/au), label=r'$z_{mid,f}/H$')
    ax[2, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 1].set_ylabel(r'$\log \left( \frac{H_{\rm hwhm}}{\rm au} \right)$', fontsize=14)
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_yscale('linear')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# for example for the best fits of constant alpha 
best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
dust_to_gas=10**(float(best_fit[3][1]))
alpha=0.01

plot_profiles_cpd(Mpdot, alpha, filename='profiles_cpd_constant_alpha.pdf')

# and contant mpdot

best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta_constant_Mpdot.txt', dtype=str)
alpha=10**(float(best_fit[0][1]))
dust_to_gas=10**(float(best_fit[3][1]))
Mpdot=10**(-6)*Mj/yr

plot_profiles_cpd(Mpdot, alpha, filename='profiles_cpd_constant_Mpdot.pdf')

