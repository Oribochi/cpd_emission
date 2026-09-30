# Profiles of the CPD
import numpy as np
from numpy import log10, exp, sqrt, pi
from units_astro import *
from dustyCPD import Rin, Rout, Sigma, T_c_approx, mu, dis
from flux_gas_dust_simple_disk import rho, n_H, ionization_fraction_HI, I_nu
from magneticCPD import H, T_z

def emission_measure(R, Mpdot, alpha):
    # This is the emission measure of the CPD at a given R
    z_arr=np.linspace(0,3*H(R,Mpdot,alpha),100)
    nH_arr=n_H(rho(R, z_arr, Mpdot, alpha))
    Temp_z_arr=T_z(z_arr, R, Mpdot, alpha)
    f_arr=ionization_fraction_HI(Temp_z_arr, R, z_arr, Mpdot, alpha)
    ne_arr=f_arr*nH_arr
    EM=np.trapezoid(ne_arr**2, z_arr)*2 # we multiply by 2 to account for the contribution from the other side of the disk (z<0)
    return EM

def EM(R,Mpdot,alpha):
    H_z=H(R,Mpdot,alpha)
    z_arr=np.linspace(0,3*H_z,50)
    n_e_arr=np.zeros(len(z_arr))
    for i in range(len(z_arr)):
        Tz=T_z(z_arr[i], R, Mpdot, alpha)
        rho=Sigma(R,Mpdot,alpha)/(sqrt(2*pi)*H_z)*exp(-z_arr[i]**2/(2*H_z**2))
        f=ionization_fraction_HI(Tz, R, z_arr[i], Mpdot, alpha)
        n_e_arr[i]=f*rho/(mu*mH)
    em=np.trapezoid(n_e_arr**2, z_arr)*2
    return em

def profiles_ff(Mpdot, alpha):
    # for each R return the ionization fraction, the temperature, the density number of electrons and the surface density Sigma
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    
    sigma_arr=Sigma(R_arr, Mpdot, alpha)
    T_c_arr=T_c_approx(R_arr, Mpdot, alpha)
    EM_arr=np.zeros(len(R_arr))
    flux_anullus=np.zeros(len(R_arr))

    for i in range(len(R_arr)):
        EM_arr[i]=EM(R_arr[i], Mpdot, alpha)

    nH_arr=n_H(rho(R_arr,0,Mpdot,alpha))
    f_arr=ionization_fraction_HI(T_c_arr, R_arr, 0, Mpdot, alpha)
    ne_arr=nH_arr*f_arr
    H_arr=H(R_arr, Mpdot, alpha)

    z_arr_mid_f=np.zeros(len(R_arr))

    for i in range(len(R_arr)):
        flux_anullus[i]=I_nu(R_arr[i], 671E9, Mpdot, alpha, dust_to_gas=10**(-10))*2*pi*R_arr[i]/dis**2*au
        z_arr=np.logspace(log10(0.001), log10(3), 500)*H_arr[i]
        Tz_arr=T_z(z_arr, R_arr[i], Mpdot, alpha)
        ne_arr_z=ionization_fraction_HI(Tz_arr, R_arr[i], z_arr, Mpdot, alpha)*n_H(rho(R_arr[i], z_arr, Mpdot, alpha))

        for j in range(len(z_arr)):
            ne_arr_z[j]=ionization_fraction_HI(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha)*n_H(rho(R_arr[i], z_arr[j], Mpdot, alpha))
            # we find where f_arr_z is less than 0.5*f_arr_z[0] and we take the z value
            #print(n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha))
            if  ne_arr_z[j]<0.5*ne_arr[i]:
                z_arr_mid_f[i]=z_arr[j] # we normalize to the scale height
                break

    return [R_arr, f_arr, T_c_arr, ne_arr, sigma_arr, EM_arr, z_arr_mid_f, flux_anullus]
