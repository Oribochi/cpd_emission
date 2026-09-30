# Plot the H-H2 profile and the flux profile with the opacity contribution
from H_H2_profile_plot import plot_profiles_H_H2
from flux_and_opacity_plot import I_nu_tau, alpha_calculator
from dustyCPD import Rin, Rout, dis
from units_astro import *
import numpy as np
from numpy import log10, sqrt, pi
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')


# for example for the best fits of constant alpha 
best_fit=np.loadtxt('best_fit_params_magnetic_disk_with_zeta.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
Bps=10**(float(best_fit[3][1]))
dust_to_gas=10**(float(best_fit[8][1]))
R_array, n_H_rel_array, n_H2_rel_array=plot_profiles_H_H2(Mpdot, Bps)

# example with our frequencies
nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9

R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 500)*au

I_nu_arr=np.zeros(len(R_arr))
# save an binary array of 4 spaces where the 1 is located in the respective tau that is the largest (tau_Hminus, tau_H2, tau_metals, tau_dust)
binary_array=np.zeros((len(R_arr), 4))

for i in range(len(R_arr)):
    alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
    I_nu_arr[i], tau_metals, tau_Hminus_arr, tau_H2_arr, tau_dust, tau_total = I_nu_tau(R_arr[i], 671E9, Mpdot, alpha, dust_to_gas=dust_to_gas)
    j=np.argmax([tau_Hminus_arr, tau_H2_arr, tau_metals, tau_dust])
    # store an array of 4 spaces with one 1 in the index i
    binary_array[i] = np.zeros(4)
    binary_array[i][j] = 1


# now make both plots in one figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(5.5,5), sharex=True)
plt.subplots_adjust(hspace=0)
axs[0].plot(R_array/au, n_H_rel_array, label=r'$\tilde{n}_{\rm H}$')
axs[0].plot(R_array/au, n_H2_rel_array, label=r'$\tilde{n}_{\rm H_2}$')
axs[0].set_ylabel(r'$\tilde{n}$ [${\rm cm}^{-3}/n_{\rm tot}$]', fontsize=12)
axs[0].legend(fontsize=12)
axs[0].set_xscale('log')
line, = axs[1].plot(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, label=r'$F_\nu^{\rm annulus}$')
axs[1].fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,0]==1, color='C0', alpha=0.5, label='H$^-$')
axs[1].fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,1]==1, color='C1', alpha=0.5, label=r'H$_2^-$')
axs[1].fill_between(R_arr/au, I_nu_arr*2*pi*R_arr*au/dis**2, where=binary_array[:,3]==1, color='C2', alpha=0.5, label='Dust')
# Create handles for the shaded regions
opacity_handles = [
    Patch(color='C0', alpha=0.5, label=r'H$^-$'),
    Patch(color='C1', alpha=0.5, label=r'H$_2^-$'),
    Patch(color='C2', alpha=0.5, label='Dust')
]

# First legend
legend1 = axs[1].legend(
    handles=[line],
    loc='upper right'
)

# Keep the first legend
axs[1].add_artist(legend1)

# Second legend
axs[1].legend(
    handles=opacity_handles,
    title='Dominant opacity',
    loc='right',
)
axs[1].set_xlabel("$R$ [au]", fontsize=12)
axs[1].set_ylabel(r'$I_{\nu}\times \frac{2 \pi R}{d^2} \times 1 \rm au ~ [\rm mJy]$', fontsize=12)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_ylim(1e-10, 1e1)
axs[1].set_yticks([1e-9,1e-6,1e-3,1], [r'$10^{-9}$', r'$10^{-6}$', '0.001', '1'])
axs[0].tick_params(axis='x', which='both', bottom=True, top=False)
axs[1].tick_params(axis='x', which='both', bottom=True, top=True)
plt.savefig('H_H2_and_flux_profile_best_fit.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Calculate the flux emited per region 
flux_Hminus = np.trapz(I_nu_arr*2*pi*R_arr*au/dis**2 * (binary_array[:,0]==1), R_arr/au)
flux_H2 = np.trapz(I_nu_arr*2*pi*R_arr*au/dis**2 * (binary_array[:,1]==1), R_arr/au)
flux_metals = np.trapz(I_nu_arr*2*pi*R_arr*au/dis**2 * (binary_array[:,2]==1), R_arr/au)
flux_dust = np.trapz(I_nu_arr*2*pi*R_arr*au/dis**2 * (binary_array[:,3]==1), R_arr/au)

print(f"Flux contribution from H-: {flux_Hminus:.3e} mJy")
print(f"Flux contribution from H2-: {flux_H2:.3e} mJy")
print(f"Flux contribution from metals: {flux_metals:.3e} mJy")
print(f"Flux contribution from dust: {flux_dust:.3e} mJy")

# and the percentaje with respect to the total flux
total_flux = flux_Hminus + flux_H2 + flux_metals + flux_dust
print ('total flux: ', total_flux, 'mJy')
print(f"Percentage contribution from H-: {flux_Hminus/total_flux*100:.8f}%")
print(f"Percentage contribution from H2-: {flux_H2/total_flux*100:.8f}%")
print(f"Percentage contribution from metals: {flux_metals/total_flux*100:.8f}%")
print(f"Percentage contribution from dust: {flux_dust/total_flux*100:.8f}%")
