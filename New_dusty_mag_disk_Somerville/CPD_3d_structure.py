# this has the structure of the 3d disk with all the parameters needed for a constant alpha case and for a variable alpha case
import numpy as np
from numpy import log10, sqrt, pi, exp
from dustyCPD import Rin, Rout, mu, Sigma
from units_astro import *
from magneticCPD import H, T_z, alpha_calculator
from H_H2_ratio import H_H2_ratio
import matplotlib

# This returns 2d arrays of R and z with the structure of the disk for a given Mpdot and Bps


def structure_disk(Mpdot, Bps):
    #R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(0.05*au/au), 100)*au
    alpha_arr=alpha_calculator(R_arr, Mpdot, Bps)
    H_arr=H(R_arr, Mpdot, alpha_arr)
    sigma_arr=Sigma(R_arr, Mpdot, alpha_arr)
    z_matrix=np.array([np.linspace(-3*H_arr[i], 3*H_arr[i], 100) for i in range(len(R_arr))])
    R_matrix=R_arr[:, np.newaxis].repeat(100, axis=1)
    alpha_matrix=alpha_arr[:, np.newaxis].repeat(100, axis=1)
    T_matrix=T_z(z_matrix, R_matrix, Mpdot, alpha_matrix)
    sigma_matrix=sigma_arr[:, np.newaxis].repeat(100, axis=1)
    rho_matrix=sigma_matrix/(sqrt(2*pi)*H(R_matrix, Mpdot, alpha_matrix))*exp(-z_matrix**2/(2*H(R_matrix, Mpdot, alpha_matrix)**2))
    n_matrix=rho_matrix/(mu*mH)
    # obtain the H_H2 ratio in the midplane
    Pressure_arr=n_matrix*kb*T_matrix

    n_H_rel_matrix=np.zeros_like(n_matrix)
    n_H2_rel_matrix=np.zeros_like(n_matrix)
    for i in range(len(R_arr)):
        
        rel_ab=H_H2_ratio(T_matrix[:,i], Pressure_arr[:,i])# relative abundances
        n_H_rel=rel_ab[:,0]
        n_H2_rel=rel_ab[:,1]

        n_H_rel_matrix[:,i]=n_H_rel
        n_H2_rel_matrix[:,i]=n_H2_rel

    return R_matrix, z_matrix, n_H_rel_matrix
    
# Example of structure for Mpdot=1E-6 Msun/yr and Bps=100 G
best_fit=np.loadtxt('best_fit_params_magnetic_disk.txt', dtype=str)
Mpdot=10**(float(best_fit[0][1]))*Mj/yr
Bps=10**(float(best_fit[3][1]))

R_example, z_example, n_H_rel_example=structure_disk(Mpdot, Bps)
# making a new array with the log of z (in the case z is negative we take -log(-z))
print(n_H_rel_example)

# replace less than 10^-4 for 10^-4 to avoid problems with the log scale
n_H_rel_example[n_H_rel_example<1e-5]=1e-5
# plot a 2d color map of the temperature structure
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.pcolormesh(R_example/au, z_example/au, n_H_rel_example, cmap='GnBu', norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1))
plt.colorbar(label=r'$\tilde{n}_{\rm H}$')
plt.xlabel('Radius (au)')
plt.ylabel('Height (au)')
plt.show()


# # This returns 2d arrays of R and z with the structure of the disk for a given Mpdot and constant alpha
# def structure_disk_simple(Mpdot, alpha):
#     #R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
#     R_arr=np.linspace(1.0001*Rin, Rout, 100)
#     H_arr=H(R_arr, Mpdot, alpha)
#     z_matrix=np.array([np.linspace(-3*H_arr[i], 3*H_arr[i], 100) for i in range(len(R_arr))])
#     R_matrix=R_arr[:, np.newaxis].repeat(100, axis=1)
#     T_matrix=T_z(z_matrix, R_matrix, Mpdot, alpha)
#     return R_matrix, z_matrix, T_matrix
    
# # Example of structure for Mpdot=1E-6 Msun/yr and alpha=0.01

# R_example, z_example, T_example=structure_disk_simple(1E-6*Mj/yr, 0.001)

# # plot a 2d color map of the temperature structure
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,6))
# plt.pcolormesh(R_example/au, z_example/au, np.log10(T_example), shading='auto', cmap='inferno')
# plt.colorbar(label=r'$\log_{10}(T/K)$')
# plt.xlabel('Radius (au)')
# plt.ylabel('Height (au)')
# plt.title('Temperature Structure of the CPD (Mdot=1E-6 Msun/yr, alpha=0.001)')
# plt.show()