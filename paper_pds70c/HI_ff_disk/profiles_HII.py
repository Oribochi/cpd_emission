import numpy as np
from numpy import pi, sqrt, log10, exp
from units_astro import *
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from free_free_CPD import *

def profiles_ff(Mpdot, alpha):
    # for each R return the ionization fraction, the temperature, the density number of electrons and the surface density Sigma
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    f_arr=np.zeros(len(R_arr))
    ne_arr_z=np.zeros(len(R_arr))
    sigma_arr=np.zeros(len(R_arr))
    ne_arr=np.zeros(len(R_arr))
    T_c_arr=np.zeros(len(R_arr))
    EM_arr=np.zeros(len(R_arr))
    z_arr_mid_f=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        T_c_arr[i]=T_c_approx(R_arr[i],Mpdot,alpha)
        # calculo la fraccion de ionización en el R dado para cada z
        z_arr=np.linspace(0,3*H(R_arr[i],Mpdot,alpha),len(R_arr))
        sigma_arr[i]=Sigma(R_arr[i],Mpdot,alpha)
        ne_arr[i]=n_e(T_c_arr[i], R_arr[i], 0, Mpdot, alpha)
        f_arr[i]=ne_arr[i]/n_H(rho(R_arr[i],0,Mpdot,alpha))
        EM_arr[i]=emission_measure(R_arr[i], Mpdot, alpha)

        for j in range(len(z_arr)):
            ne_arr_z[j]=n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha)
            # we find where f_arr_z is less than 0.5*f_arr_z[0] and we take the z value
            #print(n_e(T_c_arr[i], R_arr[i], z_arr[j], Mpdot, alpha))
            if  ne_arr_z[j]<0.5*ne_arr[i]:
                z_arr_mid_f[i]=z_arr[j] # we normalize to the scale height
                break
    return R_arr, f_arr, T_c_arr, ne_arr, sigma_arr, EM_arr, z_arr_mid_f