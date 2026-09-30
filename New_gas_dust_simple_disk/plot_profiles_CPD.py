# Plot profiles CPD
from profiles_cpd import profiles_ff
from dustyCPD import Rin, Rout, Sigma
from magneticCPD import H
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


# Disk mass
def Mdisk(Mpdot, alpha, TISM=27):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    Sigma_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        #alpha=alpha_calculator(R_arr[i], Mpdot, Bps, TISM)
        Sigma_arr[i]=Sigma(R_arr[i], Mpdot, alpha, TISM)
    return np.trapezoid(Sigma_arr*R_arr, R_arr)*2*pi # to get the mass in g

def plot_profiles_cpd(Mpdot, alpha, dust_to_gas, filename='profiles_cpd.pdf'):
    profiles=profiles_ff(Mpdot, alpha, dust_to_gas)
    R_arr=profiles[0]
    f_arr=profiles[1]
    T_c_arr=profiles[2]
    ne_arr=profiles[3]
    sigma_arr=profiles[4]
    EM_arr=profiles[5]
    z_arr_mid_f=profiles[6]
    flux_anullus=profiles[7]
    # and the rho arr
    rho_arr=sigma_arr/(sqrt(2*pi)*H(R_arr, Mpdot, alpha))
    #RT=R_truncation(Mpdot, Bps)
    #print('Truncation radius is', RT/Rj, 'Rj')

    Mass_disc=Mdisk(Mpdot, alpha)
    print('Mass of the disc is', Mass_disc/Mj, 'Mj')

    fig, ax = plt.subplots(3, 2, figsize=(7.5, 4.5), sharex=True)
    # make space in the width between the subplots
    fig.subplots_adjust(wspace=0.4, hspace=0.0)

    # and put the x ticks in all the subplots to show the lines upward and downward

    for i in range(3):
        ax[i, 0].tick_params(axis='both', direction='in')
        ax[i, 1].tick_params(axis='both', direction='in')

    ax[0, 0].plot(R_arr/au, log10(f_arr), label=r'$f$')
    #ax[0, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 0].set_ylabel(r'$\log(f)$', fontsize=14)
    ax[0, 0].set_ylim(-7+log10(np.max(f_arr)), 0.2+log10(np.max(f_arr)))
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('linear')
    ax[0, 0].set_yticks([-6, -4, -2, 0])

    ax[0, 1].plot(R_arr/au, log10(flux_anullus), label=r'$n_e$')
    #ax[0, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[0, 1].set_ylabel(r'$\log \left( \frac{F^{\rm annulus}_{671}}{\rm mJy} \right)$', fontsize=14)
    #ax[0, 1].set_ylim(np.max(log10(ne_arr))-4, np.max(log10(ne_arr))+0.2)
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('linear')
    ax[0, 1].set_yticks([-1, 0, 1])
    ax[0, 1].set_ylim(-1, 1.5)

    ax[1, 0].plot(R_arr/au, log10(sigma_arr), label=r'$\Sigma$')
    #ax[1, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[1, 0].set_ylabel(r'$\log \left( \frac{\Sigma}{\rm g ~ cm^{-2}} \right)$', fontsize=14)
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('linear')
    #ax[1, 0].set_yticks([1, 2, 3])
    
    ax[1, 1].plot(R_arr/au, log10(T_c_arr), label=r'$T_c$')
    #ax[1, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[1, 1].set_ylabel(r'$\log \left( \frac{T_c}{\rm K} \right)$', fontsize=14)
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('linear')
    #ax[1, 1].set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])

    ax[2, 0].plot(R_arr/au, log10(EM_arr/pc), label=r'$EM$')
    ax[2, 0].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 0].set_ylabel(r'$\log \left( \frac{\rm EM}{\rm cm^{-6} pc} \right)$', fontsize=14)
    ax[2, 0].set_ylim(np.max(log10(EM_arr/pc))-4, np.max(log10(EM_arr/pc))+0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('linear')
    ax[2, 0].set_yticks([22, 24, 26])

    ax[2, 1].plot(R_arr/au, log10(z_arr_mid_f/au), label=r'$z_{mid,f}/H$')
    ax[2, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    ax[2, 1].set_ylabel(r'$\log \left( \frac{H_{\rm hwhm}}{\rm au} \right)$', fontsize=14)
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_yscale('linear')
    ax[2, 1].set_ylim(-5, np.max(log10(z_arr_mid_f/au))+0.4)
    #ax[2, 1].set_yticks([-4, -3, -2])


    # ax[3, 0].plot(R_arr/au, log10(alpha_arr), label=r'$\alpha$')
    # #ax[0, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    # ax[3, 0].set_ylabel(r'$\log(\alpha)$', fontsize=14)
    # #ax[0, 1].set_ylim(np.max(log10(alpha_arr)), np.max(log10(alpha_arr)))
    # ax[3, 0].set_xscale('log')
    # ax[3, 0].set_yscale('linear')
    # ax[3, 0].set_yticks([-5, -2, 1, 4])

    # ax[3, 1].plot(R_arr/au, log10(rho_arr), label=r'$\rho$')
    # ax[3, 1].set_xlabel(r'$R$ [au]', fontsize=14)
    # ax[3, 1].set_ylabel(r'$\log \left( \frac{\rho}{\rm g ~ cm^{-3}} \right)$', fontsize=14)
    # ax[3, 1].set_xscale('log')
    # ax[3, 1].set_yscale('linear')
    # ax[3, 1].set_yticks([-14,-12,-10,-8])

    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# for example for the best fits of constant alpha 
best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
dust_to_gas=10**(float(best_fit[3][1]))
alpha=0.01

print('Best fit parameters: Mpdot =', Mpdot/(Mj/yr), 'dust_to_gas =', dust_to_gas, 'alpha =', alpha)

plot_profiles_cpd(Mpdot, alpha, dust_to_gas, filename='profiles_cpd_mag_molecular.pdf')
