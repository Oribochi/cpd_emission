# The flux of the dusty disk
from dustyCPD import *
from numpy import exp
from magneticCPD import H, T_z

# With this we can calculate the temperature of brighness depending on the wavelength (if optically thick or thin)
# this depends on the opacity, we We adopt the mass opacity from Andrews et al. (2012), κ mm = zeta x 3.4 × (0.87mm/λ) cm2 g−1
# with zeta the dust-to-gas ratio (we assume Mj0.01)
def kappa_mm(wavelength,zeta=0.01):
    # we should receive the wavelength in mm
    return zeta*3.4*(0.87/wavelength)

# With the optical depth 
def tau_mm(R,wavelength,Mpdot,alpha,TISM=27,zeta=0.01):
    return 1/2*kappa_mm(wavelength,zeta)*Sigma(R,Mpdot,alpha,TISM)

# And the brightness temperature then depending on tau_mm > 0.5 or < 0.5 is optically thick or thin, the wavelength must be in mm
def T_b(wavelength,R,Mpdot,alpha,TISM=27,zeta=0.01):
    tau = tau_mm(R,wavelength,Mpdot,alpha,TISM,zeta)
    if tau > 0.5:
        # opticall thick we choose the temperature at tau_mm=1
        Tb = (3/8*kappa_r/kappa_mm(wavelength,zeta)*T_eff(R,Mpdot)**4+T_ext(R,Mpdot,TISM)**4)**(1/4)
    else:
        # optically thin we choose the temperature at the midplane T_c weighted by the optical depth
        Tb = Sigma(R,Mpdot,alpha,TISM)*kappa_mm(wavelength,zeta)*T_c_approx(R,Mpdot,alpha,TISM)
    return Tb

# the plank function
def plank(T, nu):
    x=h*nu/(kb*T)
    if x<1e-3: # low frequency limit
        return 2*nu**2/c**2*kb*T
    elif x>1e1: # high frequency limit
        return 2*h*nu**3/c**2*exp(-x)
    else:
        return 2*h*nu**3/c**2/(exp(x)-1)
    
# Now the S is the flux density in erg/s/cm^2/Hz/ster this receives the wavelength in mm
def S(wavelength,R,Mpdot,alpha,TISM=27,zeta=0.01):
    # T_b receives the wavelength in mm
    Temp_b=T_b(wavelength,R,Mpdot,alpha,TISM,zeta)
    wav=wavelength*1e-1 # we need to convert to cm (wavelength in mm)    
    #return 2*kb*Temp_b/((wav)**2)*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    return plank(Temp_b, c/wav)*10**(26)


# Now we can calculate the total flux as the integral of the flux density over the disc in function of Mpdot and alpha
# The flux is in erg/s/cm^2/Hz
def F(wavelength,Mpdot,alpha,TISM=27,zeta=0.01):
    flux=[]
    # if it's a list of wavelengths we calculate the flux for each en
    try:
        for wav in wavelength:
            # Importante calcular para un r unico pues la funcion S depende de T_b que tiene un if i funciona de a un r
            flux.append(np.trapezoid([S(wav, r, Mpdot, alpha, TISM, zeta)*2*pi*r/dis**2 for r in R], R))
        return flux
            
    # if it's a single wavelength we calculate the flux for that wavelength
    except:
        return np.trapezoid([S(wavelength, r, Mpdot, alpha, TISM, zeta)*2*pi*r/dis**2 for r in R], R)

# Now similar to flux_simple_disk but using the stratified disk 
####################################################################################3

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

def I_nu_dust(R, nu, Mpdot, alpha, dust_to_gas=0.01):
    N=101
    h=H(R, Mpdot, alpha)
    z_arr=np.linspace(-3*h, 3*h, N)
    tau_arr=np.zeros(len(z_arr))
    Temp_z_arr=np.zeros(len(z_arr))
    rho_z_arr=np.zeros(len(z_arr))
    plank_arr=np.zeros(len(z_arr))
    k=0
    # we go fromals_max=0
    # tau_Hminus_max=0
    # for r in R_arr:
    #     alpha=alpha_calculator(r, Mpdot, Bps)
    #     I_nu, tau_metals, tau_Hminus = I_metals_Hminus_and_taus(r, nu, Mpdot, alpha)
    #     if 3H to -3H
    for i in range(len(z_arr)):
        m=i
        Temp_z=T_z(z_arr[m], R, Mpdot, alpha)
        Temp_z_arr[m]=Temp_z
        # now the density at this height
        rho=Sigma(R, Mpdot, alpha)/(sqrt(2*pi)*h)*exp(-z_arr[m]**2/(2*h**2))
        rho_z_arr[m]=rho
        tau_arr[m]=kappa_mm(c/nu*1e1, dust_to_gas)*np.trapezoid(rho_z_arr[:m], z_arr[:m]) # we convert nu to wavelength in mm
        plank_arr[m]=plank(Temp_z, nu)

        # in the case tau is too big we return to the non stratified case
        if tau_arr[m]>1000:
            k=m
            print("Large tau, optically thick case at R=", R/au, "au, tau=", tau_arr[m])
            break
        
    tau_tot=tau_arr[-1]
    #print(tau_tot)

    if k==0:
        integrate=np.array([plank_arr[i]*exp(-tau_tot+tau_arr[i]) for i in range(N)])*10**26
        return np.trapezoid(integrate, tau_arr)

    # for the case optically thinner
    #print(tau_arr[0], tau_arr[int(N/2)], tau_arr[-1])
    #print(ne_arr[0]**2*h, ne_arr[int(N/2)]**2*h, ne_arr[-1]**2*h)
    #print(k_ff_arr[0], k_ff_arr[int(N/2)], k_ff_arr[-1])
    # for the case optically thicker tau is too big and only the last value is important
    else:
        l, p = find_1(tau_arr[:k])
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)

        return plank_val*10**26


# In the simple disk, alpha is constant with R
def F_nu_dust(nu, Mpdot, alpha):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_nu_dust(R_arr[i], nu, Mpdot, alpha)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy

# plot both fluxes F and F_dust for comparison
####################################################################################3
nus_arr=np.logspace(9, 12, 50)  # from 1 GHz to 1000 GHz
F_dusty_arr=[]
F_simple_arr=[]

alpha=1e-2
Mpdot=1e-5*Mj/yr

for nu in nus_arr:
    F_dusty_arr.append(F_nu_dust(nu, Mpdot, alpha))
    wav=c/nu*1e1  # in mm
    F_simple_arr.append(F(wav, Mpdot, alpha))

import matplotlib.pyplot as plt
plt.loglog(nus_arr/1e9, F_dusty_arr, label='Dusty Stratified Disk', color='blue')
plt.loglog(nus_arr/1e9, F_simple_arr, label='Dusty Simple Disk', color='red', linestyle='dashed')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux (mJy)')
plt.legend()
plt.show()