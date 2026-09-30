# Profiles of the CPD
import numpy as np
from numpy import log10, exp, sqrt, pi
from units_astro import *
from dustyCPD import Rin, Rout, Sigma, T_c_approx, mu, dis
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance, a1, a4
from flux_gas_dust_simple_disk import I_nu
from magneticCPD import H, T_z, alpha_calculator

def rho_calc(R, z, Mpdot, alpha):
    # density at a given R and z
    H_z=H(R, Mpdot, alpha)
    Sigma_R=Sigma(R, Mpdot, alpha)
    rho=Sigma_R/(sqrt(2*pi)*H_z)*exp(-z**2/(2*H_z**2))
    return rho

def emission_measure(R, Mpdot, alpha):
    # This is the emission measure of the CPD at a given R
    z_arr=np.linspace(0,3*H(R,Mpdot,alpha),100)
    ntot_arr=rho_calc(R, z_arr, Mpdot, alpha)/(mu*mH) # define the total number density
    Temp_z_arr=T_z(z_arr, R, Mpdot, alpha)
    rho_arr=rho_calc(R, z_arr, Mpdot, alpha)
    f_arr=simple_ionization_fraction_H_minus(Temp_z_arr, rho_arr)
    ne_arr=f_arr*ntot_arr
    EM=np.trapezoid(ne_arr**2, z_arr)*2 # we multiply by 2 to account for the contribution from the other side of the disk (z<0)
    return EM

def EM(R,Mpdot,alpha):
    H_z=H(R,Mpdot,alpha)
    z_arr=np.linspace(0,3*H_z,50)
    n_e_arr=np.zeros(len(z_arr))
    T_z_arr=T_z(z_arr, R, Mpdot, alpha)
    rho_arr=rho_calc(R, z_arr, Mpdot, alpha)
    f_arr=simple_ionization_fraction_H_minus(T_z_arr, rho_arr)
    #print(f_arr)
    for i in range(len(z_arr)):
        rho=rho_arr[i]
        f=f_arr[i]
        
        n_e_arr[i]=f*rho/(mu*mH)
    em=np.trapezoid(n_e_arr**2, z_arr)*2
    return em

def profiles_ff(Mpdot, alpha, dust_to_gas):
    # for each R return the ionization fraction, the temperature, the density number of electrons and the surface density Sigma
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    #alpha_arr=alpha_calculator(R_arr, Mpdot, Bps)  

    sigma_arr=Sigma(R_arr, Mpdot, alpha)
    T_c_arr=T_c_approx(R_arr, Mpdot, alpha)
    EM_arr=np.zeros(len(R_arr))
    flux_anullus=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        EM_arr[i]=EM(R_arr[i], Mpdot, alpha)

    ntot_arr=rho_calc(R_arr, 0, Mpdot, alpha)/(mu*mH)
    f_arr=simple_ionization_fraction_H_minus(T_c_arr, ntot_arr*mu*mH)
    ne_arr=ntot_arr*f_arr
    H_arr=H(R_arr, Mpdot, alpha)

    z_arr_mid_f=np.zeros(len(R_arr))

    for i in range(len(R_arr)):
        flux_anullus[i]=I_nu(R_arr[i], 671E9, Mpdot, alpha, dust_to_gas=dust_to_gas)*2*pi*R_arr[i]/dis**2*au
        z_arr=np.logspace(log10(0.001), log10(3), 500)*H_arr[i]
        Tz_arr=T_z(z_arr, R_arr[i], Mpdot, alpha)
        rho_z_arr=rho_calc(R_arr[i], z_arr, Mpdot, alpha)
        ne_arr_z=np.zeros(len(z_arr))

        for j in range(len(z_arr)):
            ne_arr_z[j]=simple_ionization_fraction_H_minus(Tz_arr[j], rho_z_arr[j])*rho_z_arr[j]/(mu*mH)
            # we find where f_arr_z is less than 0.5*f_arr_z[0] and we take the z value
            #print(n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha))

            # print(ne_arr_z[j], ne_arr[i])
            if  ne_arr_z[j]<0.5*ne_arr[i]:
                z_arr_mid_f[i]=z_arr[j] # we normalize to the scale height
                break
        
        if z_arr_mid_f[i]==0:
            z_arr_mid_f[i]=10**(-10)*au # we put a very small value to avoid problems with log10 when plotting

    return [R_arr, f_arr, T_c_arr, ne_arr, sigma_arr, EM_arr, z_arr_mid_f, flux_anullus]


# # Example

# best_fit=np.loadtxt('best_fit_params_magnetic_disk.txt', dtype=str)
# Mpdot=10**(float(best_fit[0][1]))*Mj/yr
# Bps=10**(float(best_fit[3][1]))

# profiles=profiles_ff(Mpdot, Bps)
# R_arr=profiles[0]
# f_arr=profiles[1]
# T_c_arr=profiles[2]
# ne_arr=profiles[3]
# sigma_arr=profiles[4]
# EM_arr=profiles[5]
# z_arr_mid_f=profiles[6]
# alpha_arr=profiles[7]

# print(z_arr_mid_f)