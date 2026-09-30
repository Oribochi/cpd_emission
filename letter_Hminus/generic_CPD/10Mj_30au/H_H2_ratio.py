# Funtion to calculate the ratio of H vs H_2 
# From Heng 2015 we have that in chemical equilibrium N_H+2*N_H2 = N_tot (total number of hydrogen atoms in all forms)
# and also that Keq=N_H2/(N_H)^2
# From these two equations we can derive a quadratic equation for N_H
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from units_astro import Rgas, kb, mH
from dustyCPD import mu

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
    return np.array([n_H_rel, n_H2_rel])

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


# Now for each temperature we can calculate the K' for a given presure this was for P0 = 1 bar
def K_prime_calc(T, P):
    T_data = np.linspace(0, 6000, 61)
    G_data = np.array([216.035, 212.450, 208.004, 203.186, 198.150, 192.957, 187.640, 182.220,
    176.713, 171.132, 165.485, 159.782, 154.028, 148.230, 142.394, 136.522, 130.620,
    124.689, 118.734, 112.757, 106.760, 100.744, 94.712, 88.664, 82.603, 76.530,
    70.444, 64.349, 58.243, 52.129, 46.007, 39.877, 33.741, 27.598, 21.449 , 15.295,
    9.136, 2.973, -3.195, -9.366, -15.541, -21.718, -27.899, -34.082, -40.267, -46.454,
    -52.643, -58.834, -65.025, -71.218, -77.412, -83.606, -89.801, -95.997, -102.192,
    -108.389, -114.584, -120.780, -126.976, -133.172, -139.368])

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
    try:
        mask1=T>6000
        mask1=np.tile(mask1, [2,1]).transpose()
        K=K_prime_calc(T, P)
        mask2=K<1e-4
        mask2=np.tile(mask2, [2,1]).transpose()
        mask3=K>1e6
        mask3=np.tile(mask3, [2,1]).transpose()
        length_T=len(T)
        matrix1=np.full((length_T,2), [1.0, 0.0])
        matrix2=np.full((length_T,2), [0.0, 0.5])

        ratio=np.where((~mask1) & (mask3), matrix2, matrix1)
        ratio=np.where((~mask1) & (mask2), matrix1, ratio)
        #ratio=np.where((mask1) | mask2, matrix1, ratio)[0]
        #mask4=(~mask3)&((~mask1) & (~mask2)) # in the case no condition is fulfilled
        #mask5=mask4[:,0:,0].transpose()[0]
        #ratio[mask5]=calc_H_H2_ratio(K[mask5]).transpose()
        ratio=np.where((mask1), matrix1, ratio)
        ratio=np.where((~mask1) & (~mask2) & (~mask3), calc_H_H2_ratio(K).transpose(), ratio)
    except:
        if T>6000:
            ratio=np.array([1.0, 0.0])
        else:
            K=K_prime_calc(T, P)
            if K<1e-4:
                ratio=np.array([1.0, 0.0])
            elif K>1e6:
                ratio=np.array([0.0, 0.5])
            else:
                ratio=calc_H_H2_ratio(K)
    return ratio
        

# import matplotlib
# matplotlib.rcParams['axes.linewidth'] = 1. # the default is too thin

# # the following should avoid having the silly-looking "10^0, 10^1, 10^2",
# #  the unnecessarily complicated and "noisy" versions of "1, 10, 100"!
# matplotlib.rcParams['axes.formatter.min_exponent'] = 5
# matplotlib.rcParams['axes.formatter.limits'] = (-4,4)
# matplotlib.rcParams['axes.formatter.useoffset'] = False  # also usually better
# plt.style.use('tableau-colorblind10')

# # Example of this function
# P_example=1e6 # 1 bar in cgs units where P = ntot*kb*T
# P_example2=1e7 # 10 bar in cgs units
# # now with our physical values
# rho1=1e-8 # g/cm³
# rho2=1e-11 # g/cm³
# n0_tot=rho1/(mu*mH) # total number density of hydrogen in cm^-3 assuming a given mu
# n0_tot2=rho2/(mu*mH) # total number density of hydrogen in cm^-3 assuming a given mu
# T_example=np.linspace(100, 6000, 100)
# P_example3=n0_tot*kb*T_example  # ideal gas law
# P_example4=n0_tot2*kb*T_example  # ideal gas law with 10% of the pressure

# n_H_rel_example, n_H2_rel_example = H_H2_ratio(T_example, P_example).T
# n_H_rel_example2, n_H2_rel_example2 = H_H2_ratio(T_example, P_example2).T
# n_H_rel_example3, n_H2_rel_example3 = H_H2_ratio(T_example, P_example3).T
# n_H_rel_example4, n_H2_rel_example4 = H_H2_ratio(T_example, P_example4).T
# plt.figure(figsize=(9,5))
# plt.plot(T_example, n_H_rel_example, label=r'$\tilde{n}_{\rm H}$ (1 bar)')
# plt.plot(T_example, n_H2_rel_example, label=r'$\tilde{n}_{\rm H_2}$ (1 bar)')
# plt.plot(T_example, n_H_rel_example2, label=r'$\tilde{n}_{\rm H}$ (10 bar)', linestyle='--')
# plt.plot(T_example, n_H2_rel_example2, label=r'$\tilde{n}_{\rm H_2}$ (10 bar)', linestyle='--')
# plt.plot(T_example, n_H_rel_example3, label=r'$\tilde{n}_{\rm H}$ (P from $\rho=10^{-8} \rm ~g ~ cm^{-3}$)', linestyle=':')
# plt.plot(T_example, n_H2_rel_example3, label=r'$\tilde{n}_{\rm H_2}$ (P from $\rho=10^{-8} \rm ~g ~ cm^{-3}$)', linestyle=':')
# plt.plot(T_example, n_H_rel_example4, label=r'$\tilde{n}_{\rm H}$ (P from $\rho=10^{-10} \rm ~g ~ cm^{-3}$)', linestyle=':')
# plt.plot(T_example, n_H2_rel_example4, label=r'$\tilde{n}_{\rm H_2}$ (P from $\rho=10^{-10} \rm ~g ~ cm^{-3}$)', linestyle=':')
# plt.xlabel("$T$ [K]", fontsize=14)   
# plt.ylabel(r'Relative number density [${\rm cm}^{-3}/n_{\rm tot}$]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.gca().invert_xaxis()
# #plt.title("Relative Number Densities of H and H$_2$ vs Temperature at P=1 bar")
# plt.legend(fontsize=12)
# plt.savefig('H_H2_ratio_example.pdf', dpi=300, bbox_inches='tight')
# #plt.grid()
# plt.show()