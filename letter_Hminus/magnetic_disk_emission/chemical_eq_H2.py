# Funtion to calculate the ratio of H vs H_2 
# From Heng 2015 we have that in chemical equilibrium N_H+2*N_H2 = N_tot (total number of hydrogen atoms in all forms)
# and also that Keq=N_H2/(N_H)^2
# From these two equations we can derive a quadratic equation for N_H
import numpy as np
import matplotlib.pyplot as plt
from units_astro import Rgas, kb, mH

# Function to calculate the number densities of H and H2 given Keq' and N_tot
def calc_H_H2_ratio(Keq, N_tot):
    a = 2*Keq
    b = 1
    c = -N_tot

    N_H = (-b + (b**2 - 4*a*c)**0.5)/(2*a)
    N_H2 = Keq*N_H**2

    return N_H, N_H2, N_H/N_H2

# plot the ratio as a function of Keq for a given N_tot
N_tot=1e10 # total number density in cm-3
Keq = np.logspace(-4, 4, 100)/N_tot
N_H, N_H2, ratio = calc_H_H2_ratio(Keq, N_tot)
plt.figure()
plt.plot(Keq*N_tot, N_H/N_tot, label='N_H')
plt.plot(Keq*N_tot, N_H2/N_tot, label='N_H2')
plt.xscale('log')
plt.xlabel('Keq*N_tot')
plt.ylabel('Number Density (cm$^{-3}$)/n_tot')
plt.title('Number Densities of H and H$_2$ vs Keq (N_tot={})'.format(N_tot))
plt.legend()
plt.grid()
plt.show()

# Now taking into acount the Gibbs energy of formation of H2 from H given at 1 bar in Appendix D of Heng 2015 from 0 to 6000 K
# P0 = 1 bar: 216.035, 212.450, 208.004, 203.186, 198.150, 192.957, 187.640, 182.220,
# 176.713, 171.132, 165.485, 159.782, 154.028, 148.230, 142.394, 136.522, 130.620,
# 124.689, 118.734, 112.757, 106.760, 100.744, 94.712, 88.664, 82.603, 76.530,
# 70.444, 64.349, 58.243, 52.129, 46.007, 39.877, 33.741, 27.598, 21.449, 15.295,
# 9.136, 2.973, -3.195, -9.366, -15.541, -21.718, -27.899, -34.082, -40.267, -46.454,
# -52.643, -58.834, -65.025, -71.218, -77.412, -83.606, -89.801, -95.997, -102.192,
# -108.389, -114.584, -120.780, -126.976, -133.172, -139.368 

T_data = np.linspace(0, 6000, 61)
G_data = np.array([216.035, 212.450, 208.004, 203.186, 198.150, 192.957, 187.640, 182.220,
 176.713, 171.132, 165.485, 159.782, 154.028, 148.230, 142.394, 136.522, 130.620,
 124.689, 118.734, 112.757, 106.760, 100.744, 94.712, 88.664, 82.603, 76.530,
 70.444, 64.349, 58.243, 52.129, 46.007, 39.877, 33.741, 27.598, 21.449 , 15.295,
 9.136, 2.973, -3.195, -9.366, -15.541, -21.718, -27.899, -34.082, -40.267, -46.454,
 -52.643, -58.834, -65.025, -71.218, -77.412, -83.606, -89.801, -95.997, -102.192,
 -108.389, -114.584, -120.780, -126.976, -133.172, -139.368])

print(T_data)

P0=1 # in cgs units 1 bar = 1e6 dyn/cm2

Ntot=10**10 # total number density in cm-3
# We can obtain the Keq' from the Gibbs free energy of formation at 1 bar as
# Keq' = exp(-G0/(Rgas_H2*T))*n0
# where G0 is the Gibbs free energy of formation of H2 from H at 1 bar in erg/mol
# Rgas_H2 is the specific gas constant for H2 = Rgas/(2*mH) in erg/K
# n0 is the number density at 1 bar and temperature T, n0 = P0/(kb*T)

def Keq_G0(T, G0):
    G0_erg = -2*G0*1e10 # convert from kJ/mol to erg/mol and from -2*GH = G0
    # specific gas constant for H2
    Rgas_H2 = Rgas/(2) # specific gas constant for H2 in erg/mol/K
    if T==0:
        T=2.73 # avoid division by zero
    n0 = P0/(kb*T) # number density at 1 bar and temperature T
    return np.exp(-G0_erg/(Rgas_H2*T))/n0 # de Keq -> Keq'

#now we can plot the ratio of H to H2 as a function of temperature for a given total number density



Keq_arr = np.array([Keq_G0(Ti, G_data) for Ti, G_data in zip(T_data, G_data)])

print(Keq_arr)


N_H, N_H2, ratio = calc_H_H2_ratio(Keq_arr, N_tot)
plt.figure()
plt.plot(T_data, N_H/N_tot, label='N_H')
plt.plot(T_data, N_H2/N_tot, label='N_H2')
plt.xscale('linear')
plt.yscale('linear')
# invert x axis
plt.gca().invert_xaxis()
plt.xlabel('Temperature (K)')
plt.ylabel('Number Density (cm$^{-3}$)/n_tot')
plt.title('Number Densities of H and H$_2$ vs Temperature (N_tot={})'.format(N_tot))
plt.legend()
plt.grid()
plt.show()