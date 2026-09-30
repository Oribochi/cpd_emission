from units_astro import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# the following should avoid having the silly-looking "10^0, 10^1, 10^2",
#  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
matplotlib.rcParams['axes.formatter.min_exponent'] = 5
matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
plt.style.use('tableau-colorblind10')

# obtain the flux and nu
_1Mj_5au_nu, _1Mj_5au_flux1, _1Mj_5au_flux2, _1Mj_5au_flux3, _1Mj_5au_flux4 = np.loadtxt('fluxes_1Mj_5au.txt', unpack=True)
_1Mj_30au_nu, _1Mj_30au_flux1, _1Mj_30au_flux2, _1Mj_30au_flux3, _1Mj_30au_flux4 = np.loadtxt('fluxes_1Mj_30au.txt', unpack=True)
_1Mj_80au_nu, _1Mj_80au_flux1, _1Mj_80au_flux2, _1Mj_80au_flux3, _1Mj_80au_flux4 = np.loadtxt('fluxes_1Mj_80au.txt', unpack=True)
_10Mj_5au_nu, _10Mj_5au_flux1, _10Mj_5au_flux2, _10Mj_5au_flux3, _10Mj_5au_flux4 = np.loadtxt('fluxes_10Mj_5au.txt', unpack=True)
_10Mj_30au_nu, _10Mj_30au_flux1, _10Mj_30au_flux2, _10Mj_30au_flux3, _10Mj_30au_flux4 = np.loadtxt('fluxes_10Mj_30au.txt', unpack=True)
_10Mj_80au_nu, _10Mj_80au_flux1, _10Mj_80au_flux2, _10Mj_80au_flux3, _10Mj_80au_flux4 = np.loadtxt('fluxes_10Mj_80au.txt', unpack=True)

# now plot the fluxes 4 (no dust) with nu using log-log scale with 10 Mj solid lines and 1 Mj dashed lines
plt.figure(figsize=(4.5,3.8))
plt.plot(_1Mj_5au_nu, _1Mj_5au_flux3, linestyle='dashed', label='1 Mj, 5 au', color='C0', alpha=0.7)
plt.plot(_1Mj_30au_nu, _1Mj_30au_flux3, linestyle='dashed', label='1 Mj, 30 au', color='C1', alpha=0.7)
plt.plot(_1Mj_80au_nu, _1Mj_80au_flux3, linestyle='dashed', label='1 Mj, 80 au', color='C2', alpha=0.7)
plt.plot(_10Mj_5au_nu, _10Mj_5au_flux3, label='10 Mj, 5 au', color='C0', alpha=0.7)
plt.plot(_10Mj_30au_nu, _10Mj_30au_flux3, label='10 Mj, 30 au', color='C1', alpha=0.7)
plt.plot(_10Mj_80au_nu, _10Mj_80au_flux3, label='10 Mj, 80 au', color='C2', alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ (GHz)', fontsize=14)
plt.ylabel(r'$F_\nu$ (mJy)', fontsize=14)
plt.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('fluxes_zeta_1e-8.pdf', dpi=300, bbox_inches='tight')