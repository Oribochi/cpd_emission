# Plot profiles CPD
from profiles_cpd import profiles_ff
from magneticCPD import R_truncation, Mdisk, H
from units_astro import *
import numpy as np
from numpy import log10, pi, sqrt
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



def plot_profiles_cpd(Mpdot, Bps, filename='profiles_cpd.pdf'):
    profiles=profiles_ff(Mpdot, Bps)
    R_arr=profiles[0]
    f_arr=profiles[1]
    T_c_arr=profiles[2]
    ne_arr=profiles[3]
    sigma_arr=profiles[4]
    EM_arr=profiles[5]
    z_arr_mid_f=profiles[6]
    alpha_arr=profiles[7]
    flux_anullus=profiles[8]
    # and the rho arr
    rho_arr=sigma_arr/(sqrt(2*pi)*H(R_arr, Mpdot, alpha_arr))
    RT=R_truncation(Mpdot, Bps)
    print('Truncation radius is', RT/Rj, 'Rj')

    Mass_disc=Mdisk(Mpdot, Bps)
    print('Mass of the disc is', Mass_disc/Mj, 'Mj')

    fig, ax = plt.subplots(4, 2, figsize=(7.5, 6), sharex=True)
    # make space in the width between the subplots
    fig.subplots_adjust(wspace=0.4, hspace=0.0)

    # and put the x ticks in all the subplots to show the lines upward and downward

    for i in range(4):
        ax[i, 0].tick_params(axis='both', direction='in')
        ax[i, 1].tick_params(axis='both', direction='in')

    ax[0, 0].plot(R_arr/au, log10(f_arr), label=r'$f$')
    #ax[0, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 0].set_ylabel(r'$\log(f)$', fontsize=14)
    ax[0, 0].set_ylim(-3+log10(np.max(f_arr)), 0.2+log10(np.max(f_arr)))
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('linear')
    ax[0, 0].set_yticks([-8, -7, -6])

    ax[0, 1].plot(R_arr/au, log10(flux_anullus), label=r'$n_e$')
    #ax[0, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 1].set_ylabel(r'$\log \left( \frac{F^{\rm annulus}_{671}}{\rm mJy} \right)$', fontsize=14)
    #ax[0, 1].set_ylim(np.max(log10(ne_arr))-4, np.max(log10(ne_arr))+0.2)
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('linear')
    ax[0, 1].set_yticks([-1, 0, 1])
    ax[0, 1].set_ylim(-1.5, 1)

    ax[1, 0].plot(R_arr/au, log10(sigma_arr), label=r'$\Sigma$')
    #ax[1, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[1, 0].set_ylabel(r'$\log \left( \frac{\Sigma}{\rm g ~ cm^{-2}} \right)$', fontsize=14)
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('linear')
    ax[1, 0].set_yticks([-4, -2, 0, 2, 4])
    
    ax[1, 1].plot(R_arr/au, log10(T_c_arr), label=r'$T_c$')
    #ax[1, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[1, 1].set_ylabel(r'$\log \left( \frac{T_c}{\rm K} \right)$', fontsize=14)
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('linear')
    ax[1, 1].set_yticks([2.0, 2.4, 2.8, 3.2])

    ax[2, 0].plot(R_arr/au, log10(EM_arr/pc), label=r'$EM$')
    ax[2, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 0].set_ylabel(r'$\log \left( \frac{\rm EM}{\rm cm^{-6} pc} \right)$', fontsize=14)
    ax[2, 0].set_ylim(np.max(log10(EM_arr/pc))-4, np.max(log10(EM_arr/pc))+0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('linear')
    ax[2, 0].set_yticks([5, 6, 7, 8])

    ax[2, 1].plot(R_arr/au, log10(z_arr_mid_f/au), label=r'$z_{mid,f}/H$')
    ax[2, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 1].set_ylabel(r'$\log \left( \frac{H_{\rm hwhm}}{\rm au} \right)$', fontsize=14)
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_yscale('linear')
    ax[2, 1].set_ylim(-5, np.max(log10(z_arr_mid_f/au))+0.4)
    ax[2, 1].set_yticks([-4, -3, -2])


    ax[3, 0].plot(R_arr/au, log10(alpha_arr), label=r'$\alpha$')
    #ax[0, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[3, 0].set_ylabel(r'$\log(\alpha)$', fontsize=14)
    #ax[0, 1].set_ylim(np.max(log10(alpha_arr)), np.max(log10(alpha_arr)))
    ax[3, 0].set_xscale('log')
    ax[3, 0].set_yscale('linear')
    ax[3, 0].set_yticks([-5, -2, 1, 4])

    ax[3, 1].plot(R_arr/au, log10(rho_arr), label=r'$\rho$')
    ax[3, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[3, 1].set_ylabel(r'$\log \left( \frac{\rho}{\rm g ~ cm^{-3}} \right)$', fontsize=14)
    ax[3, 1].set_xscale('log')
    ax[3, 1].set_yscale('linear')
    ax[3, 1].set_yticks([-14,-12,-10,-8])

    
    # for all plots we make a vertical line at the truncation radius 
    for i in range(4):
        ax[i, 0].axvline(RT/au, color='C1', linestyle='--', label=r'$R_{\rm T}$')
        ax[i, 1].axvline(RT/au, color='C1', linestyle='--', label=r'$R_{\rm T}$')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


    # make a plot of only alpha magnetic
    fig, ax = plt.subplots(figsize=(4.5,3.5))
    ax.plot(R_arr/au, log10(alpha_arr), label=r'$\alpha_{\rm m}$')
    ax.set_xlabel(r'$R$ [au]', fontsize=14)
    ax.set_ylabel(r'$\log(\alpha_{\rm m})$', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_yticks([-5, -2, 1, 4])
    ax.axvline(RT/au, color='C1', linestyle='--', label=r'$R_{\rm T}$')
    ax.legend(loc='best', fontsize=12)
    plt.savefig('alpha_profile_cpd.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# for example for the best fits of constant alpha 
best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
Bps=10**(float(best_fit[3][1]))


print('Best fit parameters: Mpdot =', Mpdot/(Mj/yr), 'Bps =', Bps)

plot_profiles_cpd(Mpdot, Bps, filename='profiles_cpd_mag_molecular.pdf')
