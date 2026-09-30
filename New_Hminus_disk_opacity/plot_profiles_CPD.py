# Plot profiles CPD
from profiles_cpd import profiles_ff
from magneticCPD import R_truncation, Mdisk
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
    RT=R_truncation(Mpdot, Bps)
    print('Truncation radius is', RT/Rj, 'Rj')

    Mass_disc=Mdisk(Mpdot, Bps)
    print('Mass of the disc is', Mass_disc/Mj, 'Mj')

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
    ax[0, 0].set_ylim(-3+log10(np.max(f_arr)), 0.2+log10(np.max(f_arr)))
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('linear')
    ax[0, 0].set_yticks([-8, -7, -6, -5])

    ax[0, 1].plot(R_arr/au, log10(alpha_arr), label=r'$\alpha$')
    #ax[0, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 1].set_ylabel(r'$\log(\alpha)$', fontsize=14)
    #ax[0, 1].set_ylim(np.max(log10(alpha_arr)), np.max(log10(alpha_arr)))
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('linear')

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
    ax[1, 1].set_yticks([2.2, 2.6, 3, 3.4])

    ax[2, 0].plot(R_arr/au, log10(EM_arr/pc), label=r'$EM$')
    ax[2, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 0].set_ylabel(r'$\log \left( \frac{\rm EM}{\rm cm^{-6} pc} \right)$', fontsize=14)
    ax[2, 0].set_ylim(np.max(log10(EM_arr/pc))-4, np.max(log10(EM_arr/pc))+0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('linear')
    ax[2, 0].set_yticks([8, 9, 10, 11])

    ax[2, 1].plot(R_arr/au, log10(z_arr_mid_f/au), label=r'$z_{mid,f}/H$')
    ax[2, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 1].set_ylabel(r'$\log \left( \frac{H_{\rm hwhm}}{\rm au} \right)$', fontsize=14)
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_yscale('linear')
    ax[2, 1].set_ylim(-4.6, np.max(log10(z_arr_mid_f/au))+0.4)
    
    # for all plots we make a vertical line at the truncation radius 
    for i in range(3):
        ax[i, 0].axvline(RT/au, color='C1', linestyle='--', label=r'$R_{\rm T}$')
        ax[i, 1].axvline(RT/au, color='C1', linestyle='--', label=r'$R_{\rm T}$')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# for example for the best fits of constant alpha 
best_fit=np.loadtxt('best_fit_params_magnetic_disk.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
Bps=10**(float(best_fit[3][1]))


print('Best fit parameters: Mpdot =', Mpdot/(Mj/yr), 'Bps =', Bps)

plot_profiles_cpd(Mpdot, Bps, filename='profiles_cpd_mag.pdf')
