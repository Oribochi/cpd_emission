# The opacity free-free of the H2- atom using Someville 1964
from units_astro import *
from numpy import pi
# from astropy import constants as const
# from astropy import units as u
# from numpy import exp, log10, sqrt, pi, arctan
# import numpy as np
# import matplotlib.pyplot as plt
# from simple_ionization_fraction import saha, simple_ionization_fraction_H_minus, mu, H_minus_abundance, ionization_fraction_H_minus
# from k_ff_bf_Hminus import k_ff_Hminus
# from H_H2_ratio import H_H2_ratio


# The abundance of five key contributing elements (Lide 2004) hydrogen, helium, sodium, magnesium and potassium
a1=9.21e-1 # H
a4=7.84e-2 # He
a11=1.6e-6 # Na
a12=3.67e-5 # Mg
a19=9.87e-8 # K

# The ionization potential for each element
fi1=13.6 # H
fi4=24.59 # He
fi11=5.14 # Na
fi12=7.65 # Mg
fi19=4.34 # K

# The mass of each element
m1=1.67e-24 # H
m4=6.65e-24 # He
m11=3.82e-23 # Na
m12=2.82e-23 # Mg
m19=1.2e-22 # K


# In the atmospheres of cool stars, most of the hydrogen is in molecular form and continuous absorption 
# by H2- is of importance in determining opacities, similarly to the absorption by H- in hotter stars. (See Someville 1964)

# useful constant a0 the Bohr radius
hbar = h/(2*pi)
a0 = hbar**2/(me*qe**2)

# They use the wave number diference
def wave_number_sq(nu):
    wav=c/nu*1.0e8 # Convert frequency to wavelength (this is in angstroms)
    return 911.27/wav # this is the definition in Someville 1964 eq 19 where delta_k^2=k_f^2-k_i^2

# For practical applications, it is useful to have a simple analytic fit to the numerical values (cf. Gingerich 1961). It is found that the formula
# K = [{0.093190 + 2.857 - 0.9316/0}/(A^2)2
# -{ 2.6000+ 6.831 -
# + {35.290 -
# 4.993/0}/A&2
# (20)
# 9.804 - 10.62/0}
# -{74.520 - 62.48 + 0.4679/0} A^2] 
# cm4/dyne this opacity is per per H2 molecule and per Unit Electron Pressure
def k_ff_H2_minus(T, nu):
    delta_k_sq=wave_number_sq(nu)
    theta=5040/T
    A=delta_k_sq
    c0=0.09319*theta + 2.857 - 0.9316/theta
    c1=2.600*theta + 6.731 - 4.993/theta
    c2= 35.29*theta- 9.804 - 10.62/theta
    c3=74.52*theta - 62.48 + 0.4679/theta
    return (c0/(A**2) - c1/A + c2 - c3*A)*1e-29 # in cm4/dyne

# Contributions to the opacity per unit volume and per unit electron pressure from free-free absorption
# by H- and by H2- (present work), at a constant density p = 10⁻⁶ gm/cm3 and at wavelength 18225 Angstroms.

# theta_arr=np.linspace(0.5, 3.6, 100)
# rho=1e-6 # g/cm3
# T_arr=5040/theta_arr
# nu=c/18225e-8 # frequency in Hz
# Presure=rho*kb*T_arr/(mu*mH)    # in dyn/cm²

# # the number density of H atoms vs the H2 molecule is given by the dissociation equilibrium
# # H2 ⇌ 2H
# n_H_H2_ratios = np.array([H_H2_ratio(T, P) for T, P in zip(T_arr, Presure)])
# n_H_arr = n_H_H2_ratios[:,0]*rho/(mu*mH)*a1 # number density of H atoms
# n_H2_arr = n_H_H2_ratios[:,1]*rho/(mu*mH)*a1 # number density of H2 molecules

# #
# kappa_H_minus_arr = k_ff_Hminus(T_arr, nu)*n_H_arr
# kappa_H2_minus_arr = k_ff_H2_minus(T_arr, nu)*n_H2_arr

# plt.plot(theta_arr, kappa_H_minus_arr, label='H-')
# plt.plot(theta_arr, kappa_H2_minus_arr, label='H2-')
# plt.xlabel('Theta (5040/T)')
# plt.ylabel('Opacity (per unit volume and per unit electron pressure)')
# plt.legend()
# plt.title('Free-free opacity at 18225 Angstroms and rho=1e-6 g/cm3')
# plt.tick_params(axis='both', tickdir='in')
# plt.show()
