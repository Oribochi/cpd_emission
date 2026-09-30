# this is to see the temperature profile for a given Bps and Mpdot
# generating the range of magnetic field strengths for a given accretion rate
from dustyCPD import *
from numpy import exp
from magneticCPD import R_truncation, R_T1000, alpha_calculator
from Mag_accretion_condition import magnetospheric_accretion_range
import matplotlib.pyplot as plt

# for a given accretion rate
Mpdot=10**(-7.1)*Mj/yr # 1 Jupiter mass per Myr 
Bps1=40

# print("For Mpdot=", Mpdot/Mj*yr, "the range of Bps for magnetospheric accretion is:", magnetospheric_accretion_range(Mpdot,0,100))
#R_T
R_trunc=R_truncation(Mpdot,Bps1)

R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au
alpha_arr=alpha_calculator(R_arr,Mpdot,Bps1)
T_c_arr=T_c_approx(R_arr, Mpdot, alpha_arr)
# the ext
T_ext_arr=T_ext(R_arr, Mpdot)
# the viscous
T_visc_arr=T_visc(R_arr, Sigma_visc(R_arr,Mpdot,alpha_arr), Mpdot)

# plot
plt.plot(R_arr/au, T_c_arr, label='T_c', color='b')
plt.plot(R_arr/au, T_ext_arr, label='T_ext', color='r')
plt.plot(R_arr/au, T_visc_arr, label='T_visc', color='g')
plt.axvline(x=R_trunc/au, color='k', linestyle='--', label='R_trunc')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distance from planet [au]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.savefig('temperature_profile_Mdot_1e-7_Bps_100.pdf', dpi=300, bbox_inches='tight')
plt.show()
