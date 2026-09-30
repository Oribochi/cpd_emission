# The opacity free-free of the H2- atom using John 1994 from and Bell table to extrapolate A(T)
from units_astro import c
import numpy as np

# first we define the table with opacities


# k_john at 1 um
T_arr = np.array([1400, 1800, 2520, 3150, 4200, 5040, 6300, 10080]) # in K
k_john = np.array([0.637, 0.598, 0.546, 0.511, 0.468, 0.438, 0.4, 0.31]) # in cm^4 dyne^-1 * 10^-26
A_T_john = k_john*1e-26 # in cm^2

def A_T_interp(T):
    # Interpolate A(T) from the John 1994 values
    A_interp = np.interp(T, T_arr, A_T_john)
    return A_interp

def k_ff_H2_minus_john_interp(T, nu):
    X=c/nu*1e4 # convert frequency to wavelength in microns
    # Interpolate A(T) from the John 1994 values
    A_interp = A_T_interp(T)
    return A_interp*X**2

# # compare this with the Bell table values and Somerville 1964 values
# import numpy as np
# import matplotlib.pyplot as plt
# from k_ff_H2_2021_minus import k_ff_H2_minus


# theta=2 # theta = 5040/T, so T = 5040/theta
# Temp=5040/theta

# lambda_um = np.linspace(0.3, 1000, 100) # wavelength in microns

# k_ff_H2_minus_john_vals = 10**26*k_ff_H2_minus_john_interp(Temp, c/lambda_um*1e4) # convert wavelength to frequency in Hz
# k_ff_H2_minus_vals = 10**26*k_ff_H2_minus(Temp, c/lambda_um*1e4) 

# # compare with Bell table: 

# theta_arr = np.array([0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 2.8, 3.6])

# wavelength = np.array([
#     151883, 113913, 91130, 60753, 45565, 36452,
#     30377, 22783, 18226, 15188, 11391, 9113,
#     7594, 6509, 5696, 5063, 4142, 3505
# ])

# opacity = np.array([
#     [7.16e1, 9.23e1, 1.01e2, 1.08e2, 1.18e2, 1.26e2, 1.38e2, 1.47e2],
#     [4.03e1, 5.20e1, 5.70e1, 6.08e1, 6.65e1, 7.08e1, 7.76e1, 8.30e1],
#     [2.58e1, 3.33e1, 3.65e1, 3.90e1, 4.27e1, 4.54e1, 4.98e1, 5.33e1],
#     [1.15e1, 1.48e1, 1.63e1, 1.74e1, 1.91e1, 2.04e1, 2.24e1, 2.40e1],
#     [6.47,   8.37,   9.20,   9.84,   1.08e1, 1.16e1, 1.28e1, 1.38e1],
#     [4.15,   5.38,   5.92,   6.35,   6.99,   7.50,   8.32,   9.02],
#     [2.89,   3.76,   4.14,   4.44,   4.91,   5.28,   5.90,   6.44],
#     [1.63,   2.14,   2.36,   2.55,   2.84,   3.07,   3.49,   3.90],
#     [1.05,   1.39,   1.54,   1.66,   1.87,   2.04,   2.36,   2.68],
#     [7.36e-1, 9.75e-1, 1.09, 1.18, 1.34, 1.48, 1.74, 2.01],
#     [4.20e-1, 5.64e-1, 6.35e-1, 6.97e-1, 8.06e-1, 9.09e-1, 1.11, 1.32],
#     [2.73e-1, 3.71e-1, 4.22e-1, 4.67e-1, 5.52e-1, 6.33e-1, 7.97e-1, 9.63e-1],
#     [1.92e-1, 2.64e-1, 3.03e-1, 3.39e-1, 4.08e-1, 4.76e-1, 6.13e-1, 7.51e-1],
#     [1.43e-1, 1.98e-1, 2.30e-1, 2.59e-1, 3.17e-1, 3.75e-1, 4.92e-1, 6.09e-1],
#     [1.10e-1, 1.54e-1, 1.80e-1, 2.06e-1, 2.55e-1, 3.05e-1, 4.06e-1, 5.07e-1],
#     [8.70e-2, 1.24e-1, 1.46e-1, 1.67e-1, 2.10e-1, 2.53e-1, 3.39e-1, 4.27e-1],
#     [5.84e-2, 8.43e-2, 1.01e-1, 1.17e-1, 1.49e-1, 1.82e-1, 2.49e-1, 3.16e-1],
#     [4.17e-2, 6.10e-2, 7.34e-2, 8.59e-2, 1.11e-1, 1.37e-1, 1.87e-1, 2.40e-1]
# ])

# plt.figure(figsize=(8, 6))
# plt.loglog(lambda_um, k_ff_H2_minus_john_vals, label='John 1994', color='blue')
# plt.loglog(lambda_um, k_ff_H2_minus_vals, label='Someville 1964', color='orange', linestyle='--')
# # if theta is some value from the Bell table, plot the Bell table values
# binary=theta_arr==theta
# if np.any(binary):
#     index = np.where(binary)[0][0]
#     plt.loglog(wavelength/1e4, opacity[:, index], 'o', label=f'Bell table (theta={theta})', color='green')
# else:
#     print("No Bell table data available for this theta value.")
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'Opacity ($\rm 10^{26} ~ cm^4$/dyne)')
# plt.title(f'H2- free-free absorption coefficien theta={theta}')
# plt.legend()
# plt.grid(True, which='both', ls='--', lw=0.5)
# plt.show()