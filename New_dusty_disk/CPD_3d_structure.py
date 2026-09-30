# this has the structure of the 3d disk with all the parameters needed for a constant alpha case and for a variable alpha case
import numpy as np
from numpy import log10, sqrt, pi, exp
from dustyCPD import Rin, Rout, mu, Sigma
from units_astro import *
from magneticCPD import H, T_z, alpha_calculator


# This returns 2d arrays of R and z with the temperature and density structure of the disk for a given Mpdot and Bps
def structure_disk(Mpdot, Bps):
    #R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    R_arr=np.linspace(1.0001*Rin, Rout, 100)
    alpha_arr=alpha_calculator(R_arr, Mpdot, Bps)
    H_arr=H(R_arr, Mpdot, alpha_arr)
    z_matrix=np.array([np.linspace(-3*H_arr[i], 3*H_arr[i], 100) for i in range(len(R_arr))])
    R_matrix=R_arr[:, np.newaxis].repeat(100, axis=1)
    alpha_matrix=alpha_arr[:, np.newaxis].repeat(100, axis=1)
    rho_matrix=Sigma(R_matrix, Mpdot, alpha_matrix)/(sqrt(2*pi)*H_arr[:, np.newaxis].repeat(100, axis=1))*exp(-z_matrix**2/(2*H_arr[:, np.newaxis].repeat(100, axis=1)**2))
    T_matrix=T_z(z_matrix, R_matrix, Mpdot, alpha_matrix)
    return R_matrix, z_matrix, rho_matrix, T_matrix
    
# Example of structure for Mpdot=1E-6 Msun/yr and Bps=300 G

R_example, z_example, rho_example, T_example = structure_disk(1E-6*Mj/yr, 300)


# plot a 2d color map of the temperature structure
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.figure(figsize=(4.8,3.8))
plt.pcolormesh(R_example/au, z_example/au, np.log10(T_example), shading='gouraud', cmap='inferno', vmin=1.6, vmax=3.5)
plt.colorbar(label=r'$\log_{10}(T/\rm K)$')
plt.xlabel('Radius [au]', fontsize=14)
plt.ylabel('Height [au]', fontsize=14)
#plt.title('Temperature Structure of the CPD (Mdot=1E-6 Msun/yr, Bps=300 G)')
plt.savefig('T_structure_mag_disc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# now the density
plt.figure(figsize=(4.8,3.8))
plt.pcolormesh(R_example/au, z_example/au, np.log10(rho_example), shading='gouraud', cmap='inferno', vmin=-12, vmax=-8)
plt.colorbar(label=r'$\log_{10}(\rho/\rm g~cm^{-3})$')
plt.xlabel('Radius [au]', fontsize=14)
plt.ylabel('Height [au]', fontsize=14)
#plt.title('Density Structure of the CPD (Mdot=1E-6 Msun/yr, Bps=300 G)')
plt.savefig('rho_structure_mag_disc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# This returns 2d arrays of R and z with the structure of the disk for a given Mpdot and constant alpha
def structure_disk_simple(Mpdot, alpha):
    #R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    R_arr=np.linspace(1.0001*Rin, Rout, 100)
    H_arr=H(R_arr, Mpdot, alpha)
    z_matrix=np.array([np.linspace(-3*H_arr[i], 3*H_arr[i], 100) for i in range(len(R_arr))])
    R_matrix=R_arr[:, np.newaxis].repeat(100, axis=1)
    rho_matrix=Sigma(R_matrix, Mpdot, alpha)/(sqrt(2*pi)*H_arr[:, np.newaxis].repeat(100, axis=1))*exp(-z_matrix**2/(2*H_arr[:, np.newaxis].repeat(100, axis=1)**2))
    T_matrix=T_z(z_matrix, R_matrix, Mpdot, alpha)
    return R_matrix, z_matrix, rho_matrix, T_matrix
    
# Example of structure for Mpdot=1E-6 Msun/yr and alpha=0.01

R_example, z_example, rho_example, T_example=structure_disk_simple(1E-6*Mj/yr, 10**(-5))

# now for the alpha disc
plt.figure(figsize=(4.8,3.8))
plt.pcolormesh(R_example/au, z_example/au, np.log10(T_example), shading='gouraud', cmap='inferno', vmin=1.6, vmax=3.5)
plt.colorbar(label=r'$\log_{10}(T/\rm K)$')
plt.xlabel('Radius [au]', fontsize=14)
plt.ylabel('Height [au]', fontsize=14)
plt.savefig('T_structure_alpha_disc.pdf', dpi=300, bbox_inches='tight')
# plt.title('Temperature Structure of the CPD (Mdot=1E-6 Msun/yr, alpha=0.001)')
plt.show()

# now the density
plt.figure(figsize=(4.8,3.8))
plt.pcolormesh(R_example/au, z_example/au, np.log10(rho_example), shading='gouraud', cmap='inferno', vmin=-12, vmax=-8)
plt.colorbar(label=r'$\log_{10}(\rho/\rm g~cm^{-3})$')
plt.xlabel('Radius [au]', fontsize=14)
plt.ylabel('Height [au]', fontsize=14)
plt.savefig('rho_structure_alpha_disc.pdf', dpi=300, bbox_inches='tight')
# plt.title('Density Structure of the CPD (Mdot=1E-6 Msun/yr, alpha=0.001)')
plt.show()  

# Make a plot of the z temperature and the rho profile for an specific case Mpdot=1e-6 Mjup/yr and alpha=0.01

# at R=50 Rjup, Mpdot=1e-6 Mjup/yr and alpha=0.01
Mpdot=1E-6*Mj/yr
alpha=10**(-2)
R_example_2d=0.02*au
z_example_2d=np.linspace(-3*H(R_example_2d, Mpdot, alpha), 3*H(R_example_2d, Mpdot, alpha), 200)
T_z_example=T_z(z_example_2d, R_example_2d, Mpdot, alpha)
h=H(R_example_2d, Mpdot, alpha)
Sigma_ex=Sigma(R_example_2d, Mpdot, alpha)
rho_z_example=Sigma_ex/(sqrt(2*pi)*h)*exp(-z_example_2d**2/(2*h**2))

plt.figure(figsize=(4.8,3.8))
plt.plot(z_example_2d/au, T_z_example, label=r'$T(z)$', color='C1')
plt.xlabel('Height [au]', fontsize=14)
plt.ylabel(r'$T(z)$ [K]', fontsize=14)

plt.savefig('vertical_temperature_alpha_disc.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(4.8,3.8))
plt.plot(z_example_2d/au, rho_z_example/1e-10, label=r'$\rho(z)$', color='C1')
# Add two vertical lines at -h and h and label them with h and -h
plt.axvline(-h/au, color='C0', linestyle='--')
plt.axvline(h/au, color='C0', linestyle='--')
# make a horizontal line at 1 between -h and h
plt.hlines(1, -h/au, h/au, color='C0')  
# add a text label at the top of the plot with h and -h
plt.text(0, 1.2, r'$2~H(R)$', color='C0', fontsize=20, ha='center')
plt.ylabel(r'$\rho \left[ \frac{\rm g~cm^{-3}}{10^{-10}} \right]$', fontsize=14)
plt.xlabel('Height [au]', fontsize=14)
plt.savefig('vertical_density_alpha_disc.pdf', dpi=300, bbox_inches='tight')
plt.show()