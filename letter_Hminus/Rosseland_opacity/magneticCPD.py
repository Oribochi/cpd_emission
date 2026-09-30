# Replicating the Hasegawa model for a magnetic disk 2021
import numpy as np
from numpy import pi, sqrt, log10, exp
from units_astro import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from simple_ionization_fraction import simple_ionization_fraction_H_minus, total_ion_abundance
from dustyCPD import T_c_approx, Sigma, T_eff, T_ext, Mp, Rp, Rout, Rin, kappa_r, mu
from scipy.special import erf

def alpha_mag(R,Mpdot,alpha,Bps):
    # This is the magnetic alpha parameter
    Sigma_mag = Sigma(R,Mpdot,alpha)
    Omega = sqrt(G*Mp/R**3)
    c_s = sqrt(Rgas*T_c_approx(R,Mpdot,alpha)/(mu))
    Bp=Bps*(Rp/R)**3
    beta_mag=Sigma_mag*Omega*c_s/sqrt(2*pi)/(Bp**2/(8*pi))
    return 11*beta_mag**(-0.53)


# now we try to fing an alpha that makes alpha_mag=alpha
def alpha_finder(log_10_alpha,R,Mpdot,Bps):
    alpha=10.0**log_10_alpha
    return alpha-alpha_mag(R,Mpdot,alpha,Bps)

# We calculate the 0 of the function alpha_finder
def alpha_calculator(R,Mpdot,Bps,min_log_alpha=-5):
    logalpha=fsolve(alpha_finder, 5, args=(R, Mpdot, Bps))
    return 10**np.max([logalpha[0], min_log_alpha])


# Radius where the magnetic field dominates the ram presure
def R_truncation(Mpdot,Bps):
    Rt=(Bps*Rp**3)**(4/7)/(2*G*Mp*Mpdot**2)**(1/7)
    return Rt

# Find the first R that has 1000K temperature or more
def R_T1000(Mpdot,Bps):
    R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 500)*au
    alpha_arr=np.zeros(len(R_arr))
    R_T1000=0
    for i in range(len(R_arr)):
        alpha_arr[i]=alpha_calculator(R_arr[i],Mpdot,Bps)
        Tc=T_c_approx(R_arr[i],Mpdot,alpha_arr[i])
        if Tc>1000:
            return R_arr[i]
        else:
            R_T1000=R_arr[i]
    return R_T1000

def H(R,Mpdot,alpha):
    T=T_c_approx(R,Mpdot,alpha)
    c_s=sqrt(Rgas*T/(mu))
    Omega=sqrt(G*Mp/R**3)
    H=c_s/Omega
    return H

# Asuming a vertical hydrostatic equilibrium we can calculate the temperature
def T_z(z, R, Mpdot, alpha):
    h=H(R,Mpdot,alpha)
    tau_r=kappa_r*Sigma(R,Mpdot,alpha)*(1-erf(abs(z)/(sqrt(2)*h)))/2
    # we can use that rho is a gaussian
    ########## Cambiar
    Teff_z4=3/8*T_eff(R,Mpdot)**4*tau_r
    Text=T_ext(R,Mpdot)
    return (Teff_z4+Text**4)**(1/4)
    #  return T_c_approx(R,Mpdot,alpha)

def EM(R,Mpdot,alpha):
    H_z=H(R,Mpdot,alpha)
    z_arr=np.linspace(0,3*H_z,50)
    n_e_arr=np.zeros(len(z_arr))
    n_ion_arr=np.zeros(len(z_arr))
    for i in range(len(z_arr)):
        Tz=T_z(z_arr[i], R, Mpdot, alpha)
        rho=Sigma(R,Mpdot,alpha)/(sqrt(2*pi)*H_z)*exp(-z_arr[i]**2/(2*H_z**2))
        f=simple_ionization_fraction_H_minus(Tz, rho)
        n_e_arr[i]=f*rho/(mu*mH)
        n_ion_arr[i]=total_ion_abundance(n_e_arr[i], Tz)*rho/(mu*mH)
    em=np.trapezoid(n_e_arr*n_ion_arr, z_arr)*2
    return em

def profiles(Mpdot,Bps):
    R_array=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    alpha_array=np.zeros(len(R_array))
    Sigma_array=np.zeros(len(R_array))
    T_c_array=np.zeros(len(R_array))
    EM_array=np.zeros(len(R_array))
    f_array=np.zeros(len(R_array))
    rho_array=np.zeros(len(R_array))
    ne_array_z=np.zeros(len(R_array))
    ne_array=np.zeros(len(R_array))
    z_array_mid_f=np.zeros(len(R_array))
    for i in range(len(R_array)):
        alpha=alpha_calculator(R_array[i],Mpdot,Bps)
        alpha_array[i]=alpha
        Sigma_array[i]=Sigma(R_array[i],Mpdot,alpha)
        rho_array[i]=Sigma_array[i]/(sqrt(2*pi)*H(R_array[i],Mpdot,alpha))
        T_c_array[i]=T_c_approx(R_array[i],Mpdot,alpha)
        EM_array[i]=EM(R_array[i],Mpdot,alpha)
        f_array[i]=simple_ionization_fraction_H_minus(T_c_array[i], Sigma_array[i]/(sqrt(2*pi)*H(R_array[i],Mpdot,alpha)))
        
        # calculo la fraccion de ionización en el R dado para cada z
        z_arr=np.linspace(0,3*H(R_array[i],Mpdot,alpha),len(R_array))
        ntot=rho_array[i]/(mu*mH)
        ne_array[i]=ntot*f_array[i]
        EM_array[i]=EM(R_array[i], Mpdot, alpha)

        for j in range(len(z_arr)):
            Temp_z=T_z(z_arr[j],R_array[i],Mpdot,alpha)
            n_totz=ntot*exp(-z_arr[j]**2/(2*H(R_array[i],Mpdot,alpha)**2))
            f_ionization=simple_ionization_fraction_H_minus(Temp_z, n_totz*mu*mH)
            ne_array_z[j]=f_ionization*n_totz           
            # we find where f_arr_z is less than 0.5*f_arr_z[0] and we take the z value
            #print(n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha))
            if  ne_array_z[j]<0.5*ne_array[i]:
                z_array_mid_f[i]=z_arr[j] # we normalize to the scale height
                break
    return R_array, alpha_array, Sigma_array, T_c_array, EM_array, f_array, rho_array, z_array_mid_f

    #return R_arr, f_arr, T_c_arr, ne_arr, sigma_arr, EM_arr, z_arr_mid_f


# # We can plot the profiles
# Bps=600
# Mpdot=10**(-5.49)*Mj/yr


# R_array, alpha_array, Sigma_array, T_c_array, EM_array, f_array, rho_array = profiles(Mpdot,Bps)
# Rt=R_truncation(Mpdot,Bps)
# print('The truncation radius is:',Rt/Rj, 'Rj')
# R_T1000=R_T1000(Mpdot,Bps)
# print('The first R with T>1000K is:',R_T1000/Rj, 'Rj')

# # all the three profiles together in a plot
# fig, ax1 = plt.subplots(2,3, figsize=(15,10))
# ax1[0,0].plot(R_array/au, alpha_array, label='alpha')
# ax1[0,0].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[0,0].set_xscale('log')
# ax1[0,0].set_yscale('log')
# ax1[0,0].set_xlabel('R (au)')
# ax1[0,0].set_ylabel('alpha')

# ax1[1,0].plot(R_array/au, Sigma_array, label='Sigma')
# ax1[1,0].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[1,0].set_xscale('log')
# ax1[1,0].set_yscale('log')
# ax1[1,0].set_xlabel('R (au)')
# ax1[1,0].set_ylabel('Sigma (g/cm³)')

# ax1[0,1].plot(R_array/au, T_c_array, label='T_c')
# ax1[0,1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[0,1].set_xscale('log')
# ax1[0,1].set_yscale('log')
# ax1[0,1].set_xlabel('R (au)')
# ax1[0,1].set_ylabel('T_c')

# ax1[1,1].plot(R_array/au, EM_array/pc, label='EM (cm⁻⁶ pc)')
# ax1[1,1].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[1,1].set_xscale('log')
# ax1[1,1].set_yscale('log')
# ax1[1,1].set_xlabel('R (au)')
# ax1[1,1].set_ylabel('EM (cm⁻⁶ pc)')
# ax1[1,1].set_ylim(np.max(EM_array/pc)/1.0e10, np.max(EM_array/pc)*10)

# ax1[0,2].plot(R_array/au, f_array, label='f')
# ax1[0,2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[0,2].set_xscale('log')
# ax1[0,2].set_yscale('log')
# ax1[0,2].set_xlabel('R (au)')
# ax1[0,2].set_ylabel('f')
# ax1[0,2].set_ylim(np.max(f_array)/1.0e10, np.max(f_array)*10)

# ax1[1,2].plot(R_array/au, rho_array, label='rho')
# ax1[1,2].axvline(x=Rt/au, color='r', linestyle='--', label='Rt')
# ax1[1,2].set_xscale('log')
# ax1[1,2].set_yscale('log')
# ax1[1,2].set_xlabel('R (au)')
# ax1[1,2].set_ylabel('rho')

# plt.show()


# Disk mass
def Mdisk(Mpdot, Bps):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    Sigma_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
        Sigma_arr[i]=Sigma(R_arr[i], Mpdot, alpha)
    return np.trapezoid(Sigma_arr*R_arr, R_arr)*2*pi # to g

# print('The disk mass is:', Mdisk(Mpdot, Bps)/Mj, 'Mj')