# The opacity free-free of the H2- atom using John 1994
from numpy import log, sqrt, pi, log
from units_astro import c

# first we define the A(T) coefficient in units of 10^-16 cm^2
def A(T):
    const=11605/T
    gamma3=2
    gamma3_5=15*sqrt(pi)/8
    gamma_4=6

    euler_num=0.5772156649

    C1=5.706
    C2=18.619
    C3=6.6235

    return 140*T**(-7/2)*const**(-4)*(C1*gamma3 + C2*gamma3_5 + C3*gamma_4*(-euler_num + 11/6 -log(const)))*10**(-16) # in cm^2

# Now the absorption coefficients in the IR spectrum for the free-free transitions
# of the negative ion of molecular hydrogen, at temperature T (K) and wavelength X (um)
# in units of cm^4^dyn^-1 (per neutral hydrogen molecule and per unit electron pressure)
def k_ff_H2_minus_john(T, X):
    return A(T)*X**2

# compare with  K_ff_H2_minus(T, nu) from k_ff_H2_2021_minus.py
from k_ff_H2_2021_minus import k_ff_H2_minus
import numpy as np
import matplotlib.pyplot as plt

lambda_um = np.linspace(0.3, 1000, 100) # wavelength in microns


theta=3.6 # theta = 5040/T, so T = 5040/theta
Temp=5040/theta

k_ff_H2_minus_john_vals = 10**26*k_ff_H2_minus_john(Temp, lambda_um)
k_ff_H2_minus_vals = 10**26*k_ff_H2_minus(Temp, c/lambda_um*1e4) # convert wavelength to frequency in Hz

# compare with Bell table: 

wav_bell_arr=np.array([151883, 113913, 91130, 60753, 45565, 36452, 30377, 22783, 18226, 15188, 11391, 9113, 7594, 6509, 5696, 5063, 4142, 3505])*1e-4 # convert to microns

k_theta_05=np.array([7.16, 4.03, 2.58, 1.15, 0.647, 0.415, 0.289, 0.163, 0.105, 0.0736, 0.042, 0.0273, 0.0192, 0.0143, 0.0110, 0.0087, 0.00584, 0.00417])*10
k_theta_2=np.array([1.26, 0.708, 0.454, 0.204, 0.116, 0.075, 0.0528, 0.0307, 0.0204, 0.0137, 0.00909, 0.00633, 0.00476, 0.00375, 0.00305, 0.00253, 0.00182, 0.00137])*100
k_theta_36=np.array([1.47, 0.83, 0.533, 0.240, 0.138, 0.0902, 0.0644, 0.0390, 0.0268, 0.0192, 0.0132, 0.00963, 0.00751, 0.00609, 0.00507, 0.00427, 0.00316, 0.00240])*100


plt.figure(figsize=(8, 6))
plt.loglog(lambda_um, k_ff_H2_minus_john_vals, label='John 1994', color='blue')
plt.loglog(lambda_um, k_ff_H2_minus_vals, label='Someville 1964', color='orange', linestyle='--')
if theta==0.5:
    plt.loglog(wav_bell_arr, k_theta_05, 'o', label='Bell table (theta=0.5)', color='green')
elif theta==2:
    plt.loglog(wav_bell_arr, k_theta_2, 'o', label='Bell table (theta=2)', color='green')

elif theta==3.6:
    plt.loglog(wav_bell_arr, k_theta_36, 'o', label='Bell table (theta=36)', color='green')
else:
    print("No Bell table data available for this theta value.")
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Opacity ($\rm 10^{26} ~ cm^4$/dyne)')
plt.title(f'H2- free-free absorption coefficien theta={theta}')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.show()

# print for example at 151883 angstroms
lambda_um_example = 151883 * 1e-4 # convert angstroms to microns
k_ff_H2_minus_john_example = 10**26*k_ff_H2_minus_john(Temp, lambda_um_example)
k_ff_H2_minus_example = 10**26*k_ff_H2_minus(Temp, c/lambda_um_example*1e4) # convert wavelength to frequency in Hz

print(f"At wavelength {lambda_um_example:.6f} um (151883 angstroms) and T={Temp:.2f} K:")
print(f"John 1994: {k_ff_H2_minus_john_example:.2e}")
print(f"Someville 1964: {k_ff_H2_minus_example:.2e}")

lambda2_um_example = 11391 * 1e-4 # angstroms
k_ff_H2_minus_john_example2 = 10**26*k_ff_H2_minus_john(Temp, lambda2_um_example)
k_ff_H2_minus_example2 = 10**26*k_ff_H2_minus(Temp, c/lambda2_um_example*1e4) # convert wavelength to frequency in Hz

print(f"At wavelength {lambda2_um_example:.6f} um (11391 angstroms) and T={Temp:.2f} K:")
print(f"John 1994: {k_ff_H2_minus_john_example2:.2e}")
print(f"Someville 1964: {k_ff_H2_minus_example2:.2e}")

