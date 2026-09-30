# Funtion to calculate the ratio of H vs H_2 
# From Heng 2015 we have that in chemical equilibrium N_H+2*N_H2 = N_tot (total number of hydrogen atoms in all forms)
# and also that Keq=N_H2/(N_H)^2
# From these two equations we can derive a quadratic equation for N_H
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from units_astro import Rgas, kb, mH

# The abundance of five key contributing elements (Lide 2004) hydrogen, helium, sodium, magnesium and potassium
a1=9.21e-1 # H
a4=7.84e-2 # He
a11=1.6e-6 # Na
a12=3.67e-5 # Mg
a19=9.87e-8 # K

# Function to calculate the number densities of H and H2 relative to the total number density od hydrogen ntot, given K' = Keq'*n_tot
def calc_H_H2_ratio(K):
    # Final result of the quadratic formula
    n_H_rel=(-1+sqrt(1+8*K))/(4*K)
    n_H2_rel=1/2*(1-n_H_rel)
    return n_H_rel, n_H2_rel

# # Example of this function

# K_arr = np.logspace(-4, 4, 100)
# n_H_rel, n_H2_rel = calc_H_H2_ratio(K_arr)
# plt.figure()
# plt.plot(K_arr, n_H_rel, label='n_H_rel')
# plt.plot(K_arr, n_H2_rel, label='n_H2_rel')
# plt.xscale('log')
# plt.xlabel("K'")
# plt.ylabel('Number Density (cm$^{-3}$)/n_tot')
# plt.title("Relative Number Densities of H and H$_2$ vs K'")
# plt.legend()
# plt.grid()
# plt.show()

# Now taking into acount the Gibbs energy of formation of H2 from H given at 1 bar in Appendix D of Heng 2015 from 0 to 6000 K
# P0 = 1 bar: 216.035, 212.450, 208.004, 203.186, 198.150, 192.957, 187.640, 182.220,
# 176.713, 171.132, 165.485, 159.782, 154.028, 148.230, 142.394, 136.522, 130.620,
# 124.689, 118.734, 112.757, 106.760, 100.744, 94.712, 88.664, 82.603, 76.530,
# 70.444, 64.349, 58.243, 52.129, 46.007, 39.877, 33.741, 27.598, 21.449, 15.295,
# 9.136, 2.973, -3.195, -9.366, -15.541, -21.718, -27.899, -34.082, -40.267, -46.454,
# -52.643, -58.834, -65.025, -71.218, -77.412, -83.606we can calculate the optical depth from the free electrons interacting with the ionized metals using the equation of \citet[][including stimulated emission]{WrightBarlow1975MNRAS.170...41W,Allen1973asqu.book.....A}, 
# -108.389, -114.584, -120.780, -126.976, -133.172, -139.368 

T_data = np.linspace(0, 6000, 61)
G_data = np.array([216.035, 212.450, 208.004, 203.186, 198.150, 192.957, 187.640, 182.220,
 176.713, 171.132, 165.485, 159.782, 154.028, 148.230, 142.394, 136.522, 130.620,
 124.689, 118.734, 112.757, 106.760, 100.744, 94.712, 88.664, 82.603, 76.530,
 70.444, 64.349, 58.243, 52.129, 46.007, 39.877, 33.741, 27.598, 21.449 , 15.295,
 9.136, 2.973, -3.195, -9.366, -15.541, -21.718, -27.899, -34.082, -40.267, -46.454,
 -52.643, -58.834, -65.025, -71.218, -77.412, -83.606, -89.801, -95.997, -102.192,
 -108.389, -114.584, -120.780, -126.976, -133.172, -139.368])

# Now for each temperature we can calculate the K' for a given presure this was for P0 = 1 bar
def K_prime_calc(T, P):
    # presure is in cgs to transform to bar
    Presure_bar=P/1e6  # convert dyn/cm² to bar
    Rgas_H2=Rgas/2.0 # Rgas for H2 in
    # making a simple interpolation for the G data using the slope between the two closest points
    G_interp = np.interp(T, T_data, G_data)
    G0=-G_interp*1e10 # in erg/mol
    K_eq=np.exp(-G0/(Rgas_H2*T))
    return K_eq*Presure_bar/1*a1# the referencie presure is 1 bar

# # example
# P1=1e6 # 1 bar in cgs units where P = ntot*kb*T
# P2=1e7 # 10 bar in cgs units
# K_arr_calc=np.array([K_prime_calc(T, P1) for T in T_data])
# K_arr_calc2=np.array([K_prime_calc(T, P2) for T in T_data])
# n_H_rel_calc, n_H2_rel_calc = calc_H_H2_ratio(K_arr_calc)
# n_H_rel_calc2, n_H2_rel_calc2 = calc_H_H2_ratio(K_arr_calc2)
# plt.figure()
# plt.plot(T_data, n_H_rel_calc, label='n_H_rel')
# plt.plot(T_data, n_H2_rel_calc, label='n_H2_rel')
# plt.plot(T_data, n_H_rel_calc2, label='n_H_rel (10 bar)', linestyle='--')
# plt.plot(T_data, n_H2_rel_calc2, label='n_H2_rel (10 bar)', linestyle='--')
# plt.xlabel("Temperature (K)")
# plt.ylabel('Number Density (cm$^{-3}$)/n_tot')
# plt.gca().invert_xaxis()
# plt.title("Relative Number Densities of H and H$_2$ vs Temperature at P=1 and 10 bar")
# plt.legend()
# plt.grid()
# plt.show()

# now the ratio is given by a presure and a temperature
def H_H2_ratio(T, P):
    if T>6000:
        return 1.0, 0.0
    else:
        K=K_prime_calc(T, P)
        # if K is too big or too small we return the limits of everything in H2 or everything in H respectively
        if K>1e6:
            return 0.0, 0.5
        elif K<1e-4:
            return 1.0, 0.0
        else:
            return calc_H_H2_ratio(K)
        
# # Example of this function
# P_example=1e6 # 1 bar in cgs units where P = ntot*kb*T
# P_example2=1e7 # 10 bar in cgs units
# # now with our physical values
# rho1=1e-8 # g/cm³
# n0_tot=rho1/mH # total number density of hydrogen in cm^-3 assuming pure hydrogen gas
# T_example=np.linspace(100, 6000, 100)
# P_example3=n0_tot*kb*T_example  # ideal gas law

# n_H_rel_example, n_H2_rel_example = np.array([H_H2_ratio(T, P_example) for T in T_example]).T
# n_H_rel_example2, n_H2_rel_example2 = np.array([H_H2_ratio(T, P_example2) for T in T_example]).T
# n_H_rel_example3, n_H2_rel_example3 = np.array([H_H2_ratio(T, P) for T, P in zip(T_example, P_example3)]).T
# plt.figure()
# plt.plot(T_example, n_H_rel_example, label='ñ_H (1 bar)')
# plt.plot(T_example, n_H2_rel_example, label='ñ_H2 (1 bar)')
# plt.plot(T_example, n_H_rel_example2, label='ñ_H (10 bar)', linestyle='--')
# plt.plot(T_example, n_H2_rel_example2, label='ñ_H2 (10 bar)', linestyle='--')
# plt.plot(T_example, n_H_rel_example3, label='ñ_H (P from rho=1e-8 g/cm³)', linestyle=':')
# plt.plot(T_example, n_H2_rel_example3, label='ñ_H2 (P from rho=1e-8 g/cm³)', linestyle=':')
# plt.xlabel("Temperature (K)")       
# plt.ylabel('Number Density (cm$^{-3}$)/n_tot')
# plt.gca().invert_xaxis()
# plt.title("Relative Number Densities of H and H$_2$ vs Temperature at P=1 bar")
# plt.legend()
# plt.grid()
# plt.show()