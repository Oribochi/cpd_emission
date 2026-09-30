# plot H_H2 profile of the best fit
from profiles_cpd import profiles_ff
from magneticCPD import H
from dustyCPD import mu
from H_H2_ratio import H_H2_ratio
from H_H2_ratio_2017 import  molecular_fraction_H2
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



def plot_profiles_H_H2(Mpdot, Bps):
    profiles=profiles_ff(Mpdot, Bps)
    R_arr=profiles[0]
    T_c_arr=profiles[2]
    sigma_arr=profiles[4]
    alpha_arr=profiles[7]
    rho_arr=sigma_arr/(sqrt(2*pi)*H(R_arr, Mpdot, alpha_arr))
    n_arr=rho_arr/(mu*mH)
    # obtain the H_H2 ratio in the midplane
    Pressure_arr=n_arr*kb*T_c_arr
    n_H_rel, n_H2_rel=H_H2_ratio(T_c_arr, Pressure_arr).T # relative abundances

    return R_arr, n_H_rel, n_H2_rel

def plot_profiles_H_H2(Mpdot, Bps):
    profiles=profiles_ff(Mpdot, Bps)
    R_arr=profiles[0]
    T_c_arr=profiles[2]
    sigma_arr=profiles[4]
    alpha_arr=profiles[7]
    rho_arr=sigma_arr/(sqrt(2*pi)*H(R_arr, Mpdot, alpha_arr))
    #n_arr=rho_arr/(mu*mH)
    # obtain the H_H2 ratio in the midplane
    #Pressure_arr=n_arr*kb*T_c_arr
    #n_H_rel, n_H2_rel=H_H2_ratio(T_c_arr, Pressure_arr).T # relative abundances

    H_rel, H2_rel=np.array([molecular_fraction_H2(rho_arr[i], T_c_arr[i], 0.25) for i in range(len(R_arr))]).T

    return R_arr, H_rel, H2_rel

# for example for the best fits of constant alpha 
best_fit=np.loadtxt('best_fit_params_magnetic_disk.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
Bps=10**(float(best_fit[3][1]))
R_array, n_H_rel_array, n_H2_rel_array=plot_profiles_H_H2(Mpdot, Bps)

# with the other code

R_array, n_H_rel_array2, n_H2_rel_array2=plot_profiles_H_H2(Mpdot, Bps)

plt.figure(figsize=(4.8,3.8))
plt.plot(R_array/au, n_H_rel_array, label=r'$\tilde{n}_{\rm H}$')
plt.plot(R_array/au, n_H2_rel_array, label=r'$\tilde{n}_{\rm H_2}$')
#plt.plot(R_array/au, n_H_rel_array2, label=r'$\tilde{n}_{\rm H}$ (2017)', linestyle='--')
#plt.plot(R_array/au, n_H2_rel_array2, label=r'$\tilde{n}_{\rm H_2}$ (2017)', linestyle='--')
plt.xlabel("$R$ [au]", fontsize=14)   
plt.ylabel(r'$\tilde{n}$ [${\rm cm}^{-3}/n_{\rm tot}$]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_xaxis()
plt.xscale('log')
#plt.title("Relative Number Densities of H and H$_2$ vs Temperature at P=1 bar")
plt.legend(fontsize=12)
plt.savefig('H_H2_ratio_profile_best_fit.pdf', dpi=300, bbox_inches='tight')
#plt.grid()
plt.show()