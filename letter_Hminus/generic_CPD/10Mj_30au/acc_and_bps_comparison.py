# generating the flux using the range of magnetic fields and accretion rates for the magnetospheric accretion model
from flux_gas_dust_mag_disk import F_nu
from dustyCPD import *
from units_astro import *
import numpy as np
import matplotlib.pyplot as plt
from magneticCPD import R_truncation, R_T1000, alpha_calculator

print('Planet mass =', Mp/Mj, 'Mjup')
print('Orbital distance =', a/au, 'AU')

Mpdot_arr=np.array([6.7e-08, 1e-7, 10**(-6.5), 10**(-6), 10**(-5.5), 10**(-5)])*Mj/yr
Bps_range_arr=np.array([[73.5,73.5], [48, 222], [85,600], [150,600], [265,600], [472,600]]) # Gauss
zeta=0

# we calculate profiles for the different accretion rates and magnetic fields
# example in 10^-6.5
Mpdot_example=Mpdot_arr[2]
Bps_example_min=Bps_range_arr[2,0]
Bps_example_max=Bps_range_arr[2,1]

R_trunc_min=R_truncation(Mpdot=Mpdot_example, Bps=Bps_example_min)
R_trunc_max=R_truncation(Mpdot=Mpdot_example, Bps=Bps_example_max)

R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au

alpha_arr_min=alpha_calculator(R_arr,Mpdot_example,Bps_example_min)
alpha_arr_max=alpha_calculator(R_arr,Mpdot_example,Bps_example_max)

T_c_arr_min=T_c_approx(R_arr, Mpdot_example, alpha_arr_min)
T_c_arr_max=T_c_approx(R_arr, Mpdot_example, alpha_arr_max)

T_ext_arr_min=T_ext(R_arr, Mpdot_example)
T_ext_arr_max=T_ext(R_arr, Mpdot_example)

T_visc_arr_min=T_visc(R_arr, Sigma_visc(R_arr,Mpdot_example,alpha_arr_min), Mpdot_example)
T_visc_arr_max=T_visc(R_arr, Sigma_visc(R_arr,Mpdot_example,alpha_arr_max), Mpdot_example)


# plot
plt.plot(R_arr/au, T_c_arr_min, label='T_c_min', color='b')
plt.plot(R_arr/au, T_c_arr_max, label='T_c_max', color='c', linestyle='--')
plt.plot(R_arr/au, T_ext_arr_min, label='T_ext_min', color='r')
plt.plot(R_arr/au, T_ext_arr_max, label='T_ext_max', color='m', linestyle='--')
plt.plot(R_arr/au, T_visc_arr_min, label='T_visc_min', color='g')
plt.plot(R_arr/au, T_visc_arr_max, label='T_visc_max', color='y', linestyle='--')
plt.axvline(x=R_trunc_min/au, color='k', linestyle='--', label='R_trunc_min')
plt.axvline(x=R_trunc_max/au, color='b', linestyle='--', label='R_trunc_max')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distance from planet [au]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()