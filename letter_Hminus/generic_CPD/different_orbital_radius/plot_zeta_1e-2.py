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

# With the upper limit at bands 3, 4, 7 and 9
sigma4=4.1 /1000 #mJy
sigma7=13 /1000 #mJy
sigma9=115 /1000 #mJy

band4=145 # GHz
band7=343.5 # GHz
band9= 671 # GHz


# obtain the flux and nu
_1Mj_5au_nu, _1Mj_5au_flux1, _1Mj_5au_flux2, _1Mj_5au_flux3, _1Mj_5au_flux4 = np.loadtxt('fluxes_1Mj_5au.txt', unpack=True)
_1Mj_30au_nu, _1Mj_30au_flux1, _1Mj_30au_flux2, _1Mj_30au_flux3, _1Mj_30au_flux4 = np.loadtxt('fluxes_1Mj_30au.txt', unpack=True)
_1Mj_80au_nu, _1Mj_80au_flux1, _1Mj_80au_flux2, _1Mj_80au_flux3, _1Mj_80au_flux4 = np.loadtxt('fluxes_1Mj_80au.txt', unpack=True)
_10Mj_5au_nu, _10Mj_5au_flux1, _10Mj_5au_flux2, _10Mj_5au_flux3, _10Mj_5au_flux4 = np.loadtxt('fluxes_10Mj_5au.txt', unpack=True)
_10Mj_30au_nu, _10Mj_30au_flux1, _10Mj_30au_flux2, _10Mj_30au_flux3, _10Mj_30au_flux4 = np.loadtxt('fluxes_10Mj_30au.txt', unpack=True)
_10Mj_80au_nu, _10Mj_80au_flux1, _10Mj_80au_flux2, _10Mj_80au_flux3, _10Mj_80au_flux4 = np.loadtxt('fluxes_10Mj_80au.txt', unpack=True)

# now plot the fluxes 4 (no dust) with nu using log-log scale with 10 Mj solid lines and 1 Mj dashed lines
plt.figure(figsize=(5,3))
plt.plot(_1Mj_5au_nu, _1Mj_5au_flux1, linestyle='dashed', label='1 Mj, 5 au', color='C0', alpha=0.7)
plt.plot(_1Mj_30au_nu, _1Mj_30au_flux1, linestyle='dashed', label='1 Mj, 30 au', color='C1', alpha=0.7)
plt.plot(_1Mj_80au_nu, _1Mj_80au_flux1, linestyle='dashed', label='1 Mj, 80 au', color='C2', alpha=0.7)
plt.plot(_10Mj_5au_nu, _10Mj_5au_flux1, label='10 Mj, 5 au', color='C0', alpha=0.7)
plt.plot(_10Mj_30au_nu, _10Mj_30au_flux1, label='10 Mj, 30 au', color='C1', alpha=0.7)
plt.plot(_10Mj_80au_nu, _10Mj_80au_flux1, label='10 Mj, 80 au', color='C2', alpha=0.7)
plt.scatter([band4, band7, band9], [3*sigma4, 3*sigma7, 3*sigma9], marker='v', color='k', s=20, label=r'3$\sigma$ upper limits', zorder=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu$ (GHz)', fontsize=12)
plt.ylabel(r'$F_\nu$ (mJy)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('fluxes_zeta_1e-2.pdf', dpi=300, bbox_inches='tight')