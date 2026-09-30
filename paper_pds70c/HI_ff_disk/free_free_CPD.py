# Adding free free emission from the CPD
from units_astro import *
from dustyCPD import *
from numpy import exp, sqrt, pi, log
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.integrate import quad

# From saha equation we can calculate the ionization fraction f = n+/n for each radius in the disk
# we asume the temperature T is for a given radius in the disk
def partition_function(T):
    Z=2
    for m in range(2,4):
        zj=2*m**2*exp(-13.6*(1-1/m**2)*eV/(kb*T))
        Z+=zj
    return Z
print(partition_function(10000))

def saha(T, n_e):
    return 2/partition_function(T)*((2*pi*me*kb*T/h**2)**(3/2) * exp(-13.6*eV/(kb*T))/n_e)

# Where n_H is the number density of hydrogen atoms. Asuming the gas is only hydrogen then we define it with the mass density
def H(R, Mpdot, alpha):
    # the aspect ratio is the angular velocity divided by the sound speed
    omega=sqrt(G*Mp/(R**3))
    c_s=sqrt(kb*T_c_approx(R, Mpdot, alpha)/(mu*mH))
    H=(c_s/omega)
    return H # scale height of the disk (CPD are expected to be thick)

def rho(R, z, Mpdot, alpha):
    return 1/(sqrt(2*pi*H(R, Mpdot, alpha)**2))*Sigma(R,Mpdot,alpha)*exp(-z**2/(2*H(R, Mpdot, alpha)**2))

def n_H(rho):
    return (rho/(mH))

def n_e(T, R, z, Mpdot, alpha):
    n_h=n_H(rho(R,z,Mpdot,alpha))
    return (sqrt(saha(T,1)**2+4*saha(T,1)*n_h)-saha(T,1))/2

def T_z(z, R, Mpdot, alpha):
    h=H(R, Mpdot, alpha)
    tau_r=kappa_r*Sigma(R,Mpdot,alpha)*(1-erf(abs(z)/(sqrt(2)*h)))/2
    # we can use that rho is a gaussian
    ########## Cambiar
    Teff_z4=3/8*T_eff(R,Mpdot)**4*tau_r
    Text=T_ext(R,Mpdot)
    return (Teff_z4+Text**4)**(1/4)
    #  return T_c_approx(R,Mpdot,alpha)
                 
# with the ionization fraction we can calculate the emission measure of the disk for a given radius
def emission_measure(R, Mpdot, alpha):
    z_arr=np.linspace(0,3*H(R, Mpdot, alpha),50)
    n_e_arr=np.zeros(len(z_arr))
    for i in range(len(z_arr)):
        Tz=T_z(z_arr[i], R, Mpdot, alpha)
        n_e_arr[i]=n_e(Tz, R, z_arr[i], Mpdot, alpha)
    em=np.trapezoid(n_e_arr**2, z_arr)*2
    return em

def emission_measure_Tc(R, Mpdot, alpha):
    z_arr=np.linspace(0,3*H(R, Mpdot, alpha),50)
    n_e_arr=np.zeros(len(z_arr))
    Tc=T_c_approx(R, Mpdot, alpha)
    for i in range(len(z_arr)):
        n_e_arr[i]=n_e(Tc, R, z_arr[i], Mpdot, alpha)
    em=np.trapezoid(n_e_arr**2, z_arr)*2
    return em

def gaunt_factor(nu, T):
    gaunt_constant=49546092.4114831
    value=log(gaunt_constant*nu**(-1))+1.5*log(T)
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

#def tau(T, R, Mpdot, alpha, nu):
 #   return emission_measure(T, R, Mpdot, alpha)*k_ff(T, nu)

def tau(R, Mpdot, alpha, nu):
    # due symmetry we can integrate from 0 to 3H and multiply by 2
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h,100)
    n_e_arr=np.zeros(len(z_arr))
    k_ff_arr=np.zeros(len(z_arr))
    for i in range(len(z_arr)):
        Tz=T_z(z_arr[i], R, Mpdot, alpha)
        n_e_arr[i]=n_e(Tz, R, z_arr[i], Mpdot, alpha)
        k_ff_arr[i]=k_ff(Tz, nu)
    tau=np.trapezoid(n_e_arr**2*k_ff_arr, z_arr)
    return tau

def tau_Tc(R, Mpdot, alpha, nu):
    # due symmetry we can integrate from 0 to 3H and multiply by 2
    Tc=T_c_approx(R, Mpdot, alpha)
    EM=emission_measure_Tc(R, Mpdot, alpha)
    return EM*k_ff(Tc, nu)

def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-h*nu/(kb*T))
    else:
        return 2*h*nu**3/c**2/(exp(h*nu/(kb*T))-1)

###### Case without stratification ######

def I_ff_2(R, nu, Mpdot, alpha):
    tau_0=tau_Tc(R, Mpdot, alpha, nu)
    if tau_0<1e-4:
        return plank(T_c_approx(R, Mpdot, alpha), nu)*(tau_0)*10**26
    elif tau_0>1e4:
        return plank(T_c_approx(R, Mpdot, alpha), nu)*10**26
    else:
        return plank(T_c_approx(R, Mpdot, alpha), nu)*(1-exp(-tau_0))*10**26
    
def plank_tau_Tc(R, Mpdot, alpha, nu):
    tau_0=tau_Tc(R, Mpdot, alpha, nu)
    return plank(T_c_approx(R, Mpdot, alpha), nu), tau_0

def F_ff_2(nu, Mpdot, alpha):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_ff_2(R_arr[i], nu, Mpdot, alpha)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy

# def F_ff_2(nu, Mpdot, alpha):
#     T_mid=np.zeros(len(R))
#     I_nu_arr=np.zeros(len(R))
#     T_mid[0]=T_c_approx(R[0],Mpdot,alpha)
#     tau_i=tau(R[0], Mpdot, alpha, nu)
#     # ***** maybe we should use another temperature ******
#     I_nu_arr[0]=plank(T_mid[0], nu)*(1-exp(-tau_i))/(dis)**2
#     tau_0=tau_i

#     i=1

#     for i in range(1,len(R)):
#         T_mid[i]=T_c_approx(R[i],Mpdot,alpha)
#         tau_i=tau(R[i], Mpdot, alpha, nu)
#         I_nu_arr[i]=plank(T_mid[i], nu)*(1-exp(-tau_i))/(dis)**2
#         i+=1
#     S_nu_disk=2*pi*np.trapezoid(I_nu_arr*R, R) # this is in egs/s/cm^2/Hz
#     return S_nu_disk*10**26 # erg/cm² mJy


###### Case with stratification ######
def dtau_z(z, R, nu, Mpdot, alpha):
    h=H(R, Mpdot, alpha)
    return n_e(T_z(z, R, Mpdot, alpha), R, z, Mpdot, alpha)**2*k_ff(T_z(z, R, Mpdot, alpha), nu)

def tau_z(z, R, nu, Mpdot, alpha):
    h=H(R, Mpdot, alpha)
    tauz=quad(lambda z: dtau_z(z, R, nu, Mpdot, alpha), -10*h, z)[0]
    return tauz

#find tau_z=1 using the bissection method and return z 
def tau_z_1(a, b, R, nu, Mpdot, alpha):
    h=H(R, Mpdot, alpha)
    z=(a+b)/2
    while abs(tau_z(z, R, nu, Mpdot, alpha)-1)>0.01:
        if tau_z(z, R, nu, Mpdot, alpha)>1:
            a=z
        else:
            b=z
        z=(a+b)/2
        #print(z/h)
    return z

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


def dIntegrand(z, R, nu, Mpdot, alpha):
    h=H(R, Mpdot, alpha)
    tau_tot=tau_z(10*h, R, nu, Mpdot, alpha)
    tauz=tau_z(z, R, nu, Mpdot, alpha)
    ne=n_e(T_z(z, R, Mpdot, alpha), R, z, Mpdot, alpha)
    kff=k_ff(T_z(z, R, Mpdot, alpha), nu)
    plank_val=plank(T_z(z, R, Mpdot, alpha), nu)
    return plank_val*exp(-tau_tot+tauz)*ne**2*kff

def I_ff_quad(R, nu, Mpdot, alpha):
    h=H(R, Mpdot, alpha)
    integral=quad(lambda z: dIntegrand(z, R, nu, Mpdot, alpha), -10*h, 10*h)[0]
    return integral*10**26

def I_ff(R, nu, Mpdot, alpha):
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
        ne_arr[m]=n_e(T_z(z_arr[m], R, Mpdot, alpha), R, z_arr[m], Mpdot, alpha)
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

def tau_plank_arr(R, Mpdot, alpha, nu):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    ne_arr=np.zeros(len(z_arr))
    k_ff_arr=np.zeros(len(z_arr))
    tau_arr=np.zeros(len(z_arr))
    plank_arr=np.zeros(len(z_arr))
    # we go from 3H to -3H
    for i in range(len(z_arr)):
        m=i
        k_ff_arr[m]=k_ff(T_z(z_arr[m], R, Mpdot, alpha), nu)
        ne_arr[m]=n_e(T_z(z_arr[m], R, Mpdot, alpha), R, z_arr[m], Mpdot, alpha)
        # tau is the integral of ne**2*k_ff from z to 3H
        tau_arr[m]=np.trapezoid(ne_arr[:m]**2*k_ff_arr[:m], z_arr[:m])
        plank_arr[m]=plank(T_z(z_arr[m], R, Mpdot, alpha), nu)
        #print(tau_arr[m])
    
    return tau_arr, plank_arr, k_ff_arr, z_arr

def tau_strat(R, Mpdot, alpha, nu):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    ne_arr=np.zeros(len(z_arr))
    k_ff_arr=np.zeros(len(z_arr))
    # we go from 3H to -3H
    for i in range(len(z_arr)):
        m=i
        k_ff_arr[m]=k_ff(T_z(z_arr[m], R, Mpdot, alpha), nu)
        ne_arr[m]=n_e(T_z(z_arr[m], R, Mpdot, alpha), R, z_arr[m], Mpdot, alpha)
        # tau is the integral of ne**2*k_ff from z to 3H
    tau_0=np.trapezoid(ne_arr**2*k_ff_arr, z_arr)
    return tau_0

def F_ff(nu, Mpdot, alpha):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 50)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_ff(R_arr[i], nu, Mpdot, alpha)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # in mJy

# print("I_ff_quad", I_ff_quad(2*Rin, 1e10, 1e-5*Mj/yr, 1e-2))

# R_try=np.logspace(log10(2*Rin/au), log10(Rout/au), 50)*au
# # label put r in au
# plt.plot(R_try/au, [I_ff(r, 1e10, 1e-5*Mj/yr, 1e-2) for r in R_try], label='I_ff', alpha=0.5)
# plt.plot(R_try/au, [I_ff_2(r, 1e10, 1e-5*Mj/yr, 1e-2) for r in R_try], label='I_ff_2', linestyle='--')
# # plt.plot(R_try/au, [I_ff_quad(r, 1e10, 1e-5*Mj/yr, 1e-2) for r in R_try], label='I_ff_quad', linestyle='-.')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R (au)', fontsize=12)
# plt.ylabel('I_ff mJy', fontsize=12)
# plt.show()

# plt.plot(R_try/au, [I_ff(r, 1e10, 1e-5*Mj/yr, 1e-2)*r**2 for r in R_try], label='I_ff*r²', alpha=0.5)
# plt.plot(R_try/au, [I_ff_2(r, 1e10, 1e-5*Mj/yr, 1e-2)*r**2 for r in R_try], label='I_ff_2*r²', linestyle='--')
# # plt.plot(R_try/au, [I_ff_quad(r, 1e10, 1e-5*Mj/yr, 1e-2)*r**2 for r in R_try], label='I_ff_quad*r²', linestyle='-.')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R (au)', fontsize=12)
# plt.ylabel('I_ff mJy*cm²)', fontsize=12)
# plt.show()


def ionization_fraction(T, R, Mpdot, alpha):
    z_arr=np.linspace(0, 3*H(R, Mpdot, alpha),100)
    f_arr=np.zeros(len(z_arr))
    for i in range(len(z_arr)):
        f_arr[i]=n_e(T, R, z_arr[i], Mpdot, alpha)/n_H(rho(R,z_arr[i],Mpdot,alpha))
    return f_arr


# # # First our data with the error bars
# nus = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
# Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
# sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# # To mJy
# nu = np.array(nus)
# Fnus = np.array(Fnus)*1E3
# sFnus = np.array(sFnus)*1E3

# Mpdot=10**(-3.65)*Mj/yr
# alpha=10**(-2)
# fluxes = np.zeros(len(nu))
# for i in range(len(nu)):
#     fluxes[i]=F_ff(nu[i], Mpdot, alpha)
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
# plt.show()