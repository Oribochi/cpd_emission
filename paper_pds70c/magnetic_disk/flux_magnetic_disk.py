# This is the code with all the functions to calculate the flux and brightness temperature of the magnetic disk model
from units_astro import *
import numpy as np
from numpy import pi, sqrt, log, log10, exp
from dustyCPD import Sigma, Rin, Rout, dis, mu
from magneticCPD import H, alpha_calculator, T_z
from simple_ionization_fraction import simple_ionization_fraction
import matplotlib.pyplot as plt

def gaunt_factor(nu, T):
    value=log10(4.955e6*nu**(-1))+1.5*log10(T)
    # we use the gaunt factor of Oster 1961
    if value>1:
        return value   # for the low frequency limit
    else: 
        return 1  # for the high frequency limit

# Some constants that appear in the free free optical depth

k_ff_const=4*pi*qe**6/(3*sqrt(3)*me**2*h*c)

v_prom_const=sqrt(pi*kb/(2*me))

cte=k_ff_const/v_prom_const


def k_ff(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return (cte*x*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))
    elif x>1e3: # high frequency limit
        return (cte*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))
    else:
        return (cte*(1-exp(-x))*T**(-1/2)*nu**(-3)*gaunt_factor(nu,T))

def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-x)
    else:
        return 2*h*nu**3/c**2/(exp(x)-1)

###### Case with stratification ######

def find_1(arr):
    k=len(arr)-1
    j=0
    while j<k-1:
        p=(j+k)//2
        if arr[p]>1:
            k=p
        else:
            j=p
    # print(tau_menos, tau_mas)
    return j, k

def I_ff_metals(R, nu, Mpdot, alpha):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    ne_arr=np.zeros(len(z_arr))
    k_ff_arr=np.zeros(len(z_arr))
    tau_arr=np.zeros(len(z_arr))
    plank_arr=np.zeros(len(z_arr))
    k=0
    # we go from 3H to -3H
    for i in range(len(z_arr)):
        m=i
        k_ff_arr[m]=k_ff(T_z(z_arr[m], R, Mpdot, alpha), nu)
        # print('k_ff', k_ff_arr[m]) the kappa opacity is of the order of 10**(-27)
        rho=Sigma(R, Mpdot, alpha)/(sqrt(2*pi)*h)*exp(-z_arr[m]**2/(2*h**2))
        f=simple_ionization_fraction(T_z(z_arr[m], R, Mpdot, alpha), rho)
        # f=10**(-4)
        ne_arr[m]=f*rho/(mu*mH)
        # tau is the integral of ne**2*k_ff from z to 3H
        tau_arr[m]=np.trapezoid(ne_arr[:m]**2*k_ff_arr[:m], z_arr[:m])
        plank_arr[m]=plank(T_z(z_arr[m], R, Mpdot, alpha), nu)

        # in the case tau is too big we return to the non stratified case
        if tau_arr[m]>1000:
            k=m
            break
        
    tau_tot=tau_arr[-1]

    if k==0:
        integrate=np.array([plank_arr[i]*exp(-tau_tot+tau_arr[i]) for i in range(N)])*10**26
        return np.trapezoid(integrate, tau_arr)
        
    # for the case optically thinner
    #print(tau_arr[0], tau_arr[int(N/2)], tau_arr[-1])
    #print(ne_arr[0]**2*h, ne_arr[int(N/2)]**2*h, ne_arr[-1]**2*h)
    #print(k_ff_arr[0], k_ff_arr[int(N/2)], k_ff_arr[-1])
    # for the case optically thicker tau is too big and only the last value is important
    else:
        l, p =find_1(tau_arr[:k])
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)
        return plank_val*10**26
    
#plot the I_ff
# R_array=np.logspace(log10(1.001*Rin/au), log10(Rout/au), 50)*au
# I_ff_arr=np.zeros(len(R_array))
# for i in range(len(R_array)):
#     alpha=alpha_calculator(R_array[i], Mpdot, Bps)
#     I_ff_arr[i]=I_ff_metals(R_array[i], 100E9, Mpdot, alpha)
# plt.plot(R_array/au, I_ff_arr)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R (au)')
# plt.ylabel('I_ff (W/m²/Hz/sr)')
# plt.ylim(1e-10, 1e17)
# plt.show()
    
def F_ff_metals(nu, Mpdot, Bps):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        alpha=alpha_calculator(R_arr[i], Mpdot, Bps)
        F_ff_arr[i]=I_ff_metals(R_arr[i], nu, Mpdot, alpha)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy


# # ejemplo

# Bps=600
# Mpdot=10**(-5.49)*Mj/yr


# # # First our data with the error bars
# nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
# Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
# sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# # To mJy
# nu = np.array(nus)
# Fnus = np.array(Fnus)*1E3
# sFnus = np.array(sFnus)*1E3

# fluxes = np.zeros(len(nu))
# for i in range(len(nu)):
#     fluxes[i]=F_ff_metals(nu[i], Mpdot, Bps)
#     print('Flux at', round(nu[i]/1E9, 1), 'GHz =', fluxes[i], 'mJy')
#     if i>0:
#         print('Spectral index is ', (log10(fluxes[i])-log10(fluxes[i-1]))/(log10(nu[i])-log10(nu[i-1])))

# plt.errorbar(np.array(nus)/10**9, Fnus, yerr=sFnus, fmt='o', label='Data', color='blue')
# plt.plot(np.array(nus)/10**9, fluxes, label='model', alpha=0.5, color='red')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\nu$ [GHz]', fontsize=12)
# plt.ylabel(r'$F_{\nu}$ [mJy]', fontsize=12)
# plt.legend(loc='upper left')
# plt.savefig('best_fit_ff_metals.png')
# plt.show()