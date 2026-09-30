# Replicating the Hasegawa model for a magnetic disk 2021
import numpy as np
from numpy import pi, sqrt, log10
from units_astro import *
from scipy.optimize import fsolve
from dustyCPD import T_c_approx, Sigma, T_eff, T_ext, Mp, Rp, Rout, Rin, kappa_r, mu
from scipy.special import erf
from scipy.optimize import fsolve
from numpy import sqrt, pi, log10
from units_astro import *
from dustyCPD import Sigma, T_c_approx, mu, Rp, Mp
import numpy as np

def alpha_mag(R,Mpdot,alpha,Bps, TISM=27):
    # This is the magnetic alpha parameter
    Sigma_mag = Sigma(R,Mpdot,alpha, TISM)
    Omega = sqrt(G*Mp/R**3)
    c_s = sqrt(Rgas*T_c_approx(R,Mpdot,alpha)/(mu))
    Bp=Bps*(Rp/R)**3
    beta_mag=Sigma_mag*Omega*c_s/sqrt(2*pi)/(Bp**2/(8*pi))
    return 11*beta_mag**(-0.53)



# now we try to fing an alpha that makes alpha_mag=alpha
def alpha_finder(log_10_alpha,R,Mpdot,Bps, TISM=27):
    alpha=10.0**log_10_alpha
    return log_10_alpha-log10(alpha_mag(R,Mpdot,alpha,Bps, TISM))


# We calculate the 0 of the function alpha_finder
def alpha_calculator(R,Mpdot,Bps,min_log_alpha=-5, TISM=27):
    # first check R is an array or a number
    if np.isscalar(R):
        R=np.array([R])
        logalpha=fsolve(alpha_finder, 5, args=(R, Mpdot, Bps, TISM))
        return 10**np.max(np.array([logalpha[0], min_log_alpha]))
    else:
        length_R=len(R)
        arr_initial_guess=np.full(length_R,5)
        arr_min_log_alpha=np.full(length_R,min_log_alpha)
        logalpha=fsolve(alpha_finder, arr_initial_guess, args=(R, Mpdot, Bps, TISM))
        #print('logalpha:', logalpha)
        return 10**np.max(np.array([logalpha, arr_min_log_alpha]), axis=0)


# Radius where the magnetic field dominates the ram presure
def R_truncation(Mpdot,Bps):
    Rt=(Bps*Rp**3)**(4/7)/(2*G*Mp*Mpdot**2)**(1/7)
    return Rt

# Find the first R that has 1000K temperature or more
def R_T1000(Mpdot,Bps, TISM=27):
    R_arr=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 500)*au
    alpha_arr=np.zeros(len(R_arr))
    R_T1000=0
    for i in range(len(R_arr)):
        alpha_arr[i]=alpha_calculator(R_arr[i],Mpdot,Bps, TISM)
        Tc=T_c_approx(R_arr[i],Mpdot,alpha_arr[i], TISM)
        if Tc>1000:
            return R_arr[i]
        else:
            R_T1000=R_arr[i]
    return R_T1000

def H(R,Mpdot,alpha, TISM=27):
    T=T_c_approx(R,Mpdot,alpha,TISM)
    c_s=sqrt(Rgas*T/(mu))
    Omega=sqrt(G*Mp/R**3)
    H=c_s/Omega
    return H

# Asuming a vertical hydrostatic equilibrium we can calculate the temperature
def T_z(z, R, Mpdot, alpha, TISM=27):
    h=H(R,Mpdot,alpha)
    tau_r=kappa_r*Sigma(R,Mpdot,alpha, TISM)*(1-erf(abs(z)/(sqrt(2)*h)))/2
    # we can use that rho is a gaussian
    ########## Cambiar
    Teff_z4=3/8*T_eff(R,Mpdot)**4*tau_r
    Text=T_ext(R,Mpdot, TISM)
    return (Teff_z4+Text**4)**(1/4)
    #  return T_c_approx(R,Mpdot,alpha, TISM)

# Disk mass
def Mdisk(Mpdot, Bps, TISM=27):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    Sigma_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps, TISM)
        Sigma_arr[i]=Sigma(R_arr[i], Mpdot, alpha, TISM)
    return np.trapezoid(Sigma_arr*R_arr, R_arr)*2*pi # to get the mass in g