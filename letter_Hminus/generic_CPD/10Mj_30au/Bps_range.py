# generating the range of magnetic field strengths for a given accretion rate
from dustyCPD import *
from numpy import exp
from magneticCPD import R_truncation, R_T1000, alpha_calculator
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from k_ff_metals import k_ff_metals
from k_ff_bf_Hminus import k_ff_Hminus, k_bf_Hminus
#from k_bf_H_2021 import k_bf_H
from k_ff_H2_2021_minus import k_ff_H2_minus
from H_H2_ratio import H_H2_ratio
from k_ff_He_minus import k_ff_He_minus
from k_dust import kappa_mm
import matplotlib.pyplot as plt

# for a given accretion rate
Mpdot=1e-7*Mj/yr # 1 Jupiter mass per Myr 
Bps1=100

# For various magnetic field strengths, calculate the truncation radius and the radius where T>1000K
Bps_arr=np.logspace(0, 3, 50) # Gauss
R_trunc_arr=np.zeros(len(Bps_arr))
R_T1000_arr=np.zeros(len(Bps_arr))

def T_visc_simple(R,Mpdot,Bps):
    alpha=alpha_calculator(R,Mpdot,Bps)
    T_v=T_visc(R, Sigma_visc(R,Mpdot,alpha), Mpdot)
    return T_v

# Find the first R that has 1000K temperature or more
def R_T1000_new(Mpdot,Bps):
    R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 500)*au
    R_T1000=0
    for i in range(len(R_arr)):
        Tc=T_visc_simple(R_arr[i],Mpdot,Bps)
        if Tc>1000:
            return R_arr[i]
        else:
            R_T1000=R_arr[i]
    return R_T1000


for i in range(len(Bps_arr)):
    Bps=Bps_arr[i]
    R_trunc_arr[i]=R_truncation(Mpdot,Bps)
    R_T1000_arr[i]=R_T1000_new(Mpdot,Bps)

# I want a plot with the magnetic field strength in the y-axis and the distance from the planet in the x-axis
# 
plt.figure(figsize=(8,6))
plt.plot(R_trunc_arr/Rj, Bps_arr, label='R_trunc', color='b')
plt.plot(R_T1000_arr/Rj, Bps_arr, label='R_T1000', color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distance from planet [Rj]')
plt.ylabel('Magnetic field strength [G]')
plt.legend()
plt.show()

# plot temperature and the truncation radius
R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au
T_arr=np.zeros(len(R_arr))
for i in range(len(R_arr)):
    T_arr[i]=T_visc_simple(R_arr[i],Mpdot,Bps1)

R_trunc_1=R_truncation(Mpdot,Bps1)
R_T1000_1=R_T1000_new(Mpdot,Bps1)
plt.plot(R_arr/au, T_arr, label='Temperature', color='g')
plt.axvline(R_trunc_1/au, color='b', label='R_trunc')
plt.axvline(R_T1000_1/au, color='r', linestyle='--', label='R_T1000')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distance from planet [AU]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()