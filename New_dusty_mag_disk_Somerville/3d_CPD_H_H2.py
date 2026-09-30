# this has the structure of the 3d disk with all the parameters needed for a constant alpha case and for a variable alpha case
import numpy as np
from numpy import log10, sqrt
from dustyCPD import Rin, Rout, mu
from units_astro import *
from magneticCPD import H, T_z, alpha_calculator

# This returns 2d arrays of R and z with the structure of the disk for a given Mpdot and Bps


def structure_disk(Mpdot, Bps):
    #R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    R_arr=np.linspace(1.0001*Rin, Rout, 100)
    alpha_arr=alpha_calculator(R_arr, Mpdot, Bps)
    H_arr=H(R_arr, Mpdot, alpha_arr)
    z_matrix=np.array([np.linspace(-3*H_arr[i], 3*H_arr[i], 100) for i in range(len(R_arr))])
    R_matrix=R_arr[:, np.newaxis].repeat(100, axis=1)
    alpha_matrix=alpha_arr[:, np.newaxis].repeat(100, axis=1)
    T_matrix=T_z(z_matrix, R_matrix, Mpdot, alpha_matrix)
    return R_matrix, z_matrix, T_matrix
    
# Example of structure for Mpdot=1E-6 Msun/yr and Bps=100 G

R_example, z_example, T_example=structure_disk(1E-6*Mj/yr, 300)
# making a new array with the log of z (in the case z is negative we take -log(-z))

# plot a 2d color map of the temperature structure
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.pcolormesh(R_example/au, z_example/au, np.log10(T_example), shading='auto', cmap='inferno')
plt.colorbar(label=r'$\log_{10}(T/K)$')
plt.xlabel('Radius (au)')
plt.ylabel('Height (H(R))')
plt.title('Temperature Structure of the CPD (Mdot=1E-6 Msun/yr, Bps=300 G)')
plt.show()


# This returns 2d arrays of R and z with the structure of the disk for a given Mpdot and constant alpha
def structure_disk_simple(Mpdot, alpha):
    #R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    R_arr=np.linspace(1.0001*Rin, Rout, 100)
    H_arr=H(R_arr, Mpdot, alpha)
    z_matrix=np.array([np.linspace(-3*H_arr[i], 3*H_arr[i], 100) for i in range(len(R_arr))])
    R_matrix=R_arr[:, np.newaxis].repeat(100, axis=1)
    T_matrix=T_z(z_matrix, R_matrix, Mpdot, alpha)
    return R_matrix, z_matrix, T_matrix
    
# Example of structure for Mpdot=1E-6 Msun/yr and alpha=0.01

R_example, z_example, T_example=structure_disk_simple(1E-6*Mj/yr, 0.001)

# plot a 2d color map of the temperature structure
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.pcolormesh(R_example/au, z_example/au, np.log10(T_example), shading='auto', cmap='inferno')
plt.colorbar(label=r'$\log_{10}(T/K)$')
plt.xlabel('Radius (au)')
plt.ylabel('Height (au)')
plt.title('Temperature Structure of the CPD (Mdot=1E-6 Msun/yr, alpha=0.001)')
plt.show()