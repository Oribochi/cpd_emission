# to see how arrays and cumulative function work
import numpy as np
z_arr=np.array([0.1,0.2,0.3,0.4,0.5])
m=len(z_arr)
dz=z_arr[1]-z_arr[0]

d_kappa_arr=np.array([1,0.8,0.6,0.4,0.2])

tau_arr_new=np.cumsum(d_kappa_arr*dz)

tau_arr_old=np.zeros(len(d_kappa_arr))
for i in range(len(d_kappa_arr)):
    tau_arr_old[i]=np.trapezoid(d_kappa_arr[:i+1], z_arr[:i+1])

print(tau_arr_new)
print(tau_arr_old)

# how to make the old and new the same
tau_arr_new_fixed=np.zeros(len(d_kappa_arr))
tau_arr_new_fixed[1:m+1]=tau_arr_new[1:m+1]-d_kappa_arr[1:m+1]/2*dz - d_kappa_arr[0]/2*dz
print(tau_arr_new_fixed)

from units_astro import *

R=1000*au
M=1*Msun
rho0=M/(4/3*np.pi*R**3) 
t_free_fall=np.sqrt(3*np.pi/(32*G*rho0))

print(t_free_fall/yr)