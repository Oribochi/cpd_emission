# The flux of the dusty disk from Zhu 2018
from dustyCPD import *
from numpy import exp
from magneticCPD import H, T_z
from k_dust import kappa_mm, tau_mm


# The brightness temperature then depending on tau_mm > 0.5 or < 0.5 is optically thick or thin, the wavelength must be in mm
def T_b(wavelength,R,Mpdot,alpha,TISM=27,zeta=0.01):
    tau = tau_mm(R,wavelength,Mpdot,alpha,TISM,zeta)
    if tau > 0.5:
        # opticall thick we choose the temperature at tau_mm=1
        Tb = (3/8*kappa_r/kappa_mm(wavelength,zeta)*T_eff(R,Mpdot)**4+T_ext(R,Mpdot,TISM)**4)**(1/4)
    else:
        # optically thin we choose the temperature at the midplane T_c weighted by the optical depth
        Tb = 2*tau*T_c_approx(R,Mpdot,alpha,TISM)
        #print("Tb =", Tb)
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
    return 2*kb*Temp_b/((wav)**2)*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    #return plank(Temp_b, c/wav)*10**(26)


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

# this will receive an array of Temperatures and return the plank function for each temperature
def plank(T, nu):    
    x=h*nu/(kb*T)
    mask1 = x < 1e-3  # low frequency limit
    mask2 = x > 1e1   # high frequency limit
    result = np.zeros_like(x) #
    result[mask1] = 2*nu**2/c**2*kb*T[mask1]
    result[mask2] = 2*h*nu**3/c**2*exp(-x[mask2])
    mask3 = ~(mask1 | mask2)  # intermediate values
    result[mask3] = 2*h*nu**3/c**2/(exp(x[mask3])-1)
    return result

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
    dz=z_arr[1]-z_arr[0] # step in z
    Temp_z_arr=T_z(z_arr, R, Mpdot, alpha)
    Sigma_R=Sigma(R, Mpdot, alpha) # surface density at R
    rho_z_arr=Sigma_R/(sqrt(2*pi)*h)*exp(-z_arr**2/(2*h**2))
    plank_arr=plank(Temp_z_arr, nu)
    d_tau_arr=kappa_mm(c/nu*1e1, dust_to_gas)*rho_z_arr*(dz)  # we convert nu to wavelength in mm
    tau_arr=np.cumsum(d_tau_arr) # cumulative sum 
    # Now the total tau
    tau_tot=tau_arr[-1]
    # in the case we dont have massive tau we can just integrate directly
    if tau_tot<1000:
        integrate=np.trapezoid(plank_arr*exp(-tau_tot+tau_arr), tau_arr)*10**26 # to mJy
        return integrate
    # in the case tau is too big we return to the non stratified case
    else:
        l, p = find_1(tau_arr)
        z1=(z_arr[l]+z_arr[p])/2
        # z1=3*h
        plank_val=plank(T_z(z1, R, Mpdot, alpha), nu)
        return plank_val*10**26


# In the simple disk, alpha is constant with R
def F_nu_dust(nu, Mpdot, alpha, zeta=0.01):
    R_arr=np.logspace(log10(1.0001*Rin/au), log10(Rout/au), 100)*au
    F_ff_arr=np.zeros(len(R_arr))
    for i in range(len(R_arr)):
        F_ff_arr[i]=I_nu_dust(R_arr[i], nu, Mpdot, alpha, zeta)*2*pi*R_arr[i]/dis**2
    return np.trapezoid(F_ff_arr, R_arr)  # to mJy

# #plot both fluxes F and F_dust for comparison
# ####################################################################################

# nus_arr=np.logspace(9, 12, 50)  # from 1 GHz to 1000 GHz
# F_dusty_arr=[]
# F_simple_arr=[]

# alpha=1e-2
# Mpdot=1e-8*Mj/yr

# for nu in nus_arr:
#     F_dusty_arr.append(F_nu_dust(nu, Mpdot, alpha))
#     wav=c/nu*1e1  # in mm
#     F_simple_arr.append(F(wav, Mpdot, alpha))

# import matplotlib.pyplot as plt
# plt.title('Flux for 1e-8 Mj/yr at 5 au')
# plt.loglog(nus_arr/1e9, F_dusty_arr, label='Dusty Stratified Disk', color='blue')
# plt.loglog(nus_arr/1e9, F_simple_arr, label='Dusty Simple Disk', color='red', linestyle='dashed')
# plt.xlabel('Frequency (GHz)')
# plt.ylabel('Flux (mJy)')
# plt.legend()
# plt.show()

# # print flux for 1 mm and 2 mm
# wav_list=[1.0, 2.0]  # in mm
# for wav in wav_list:
#     nu=c/wav*1e1  # in Hz
#     F_dusty=F_nu_dust(nu, Mpdot, alpha)
#     F_simple=F(wav, Mpdot, alpha)
#     print(f'Flux at {wav} mm for Mpdot={Mpdot/(Mj/yr)} Mj/yr:')
#     print(f'  Dusty Stratified Disk: {F_dusty} mJy')
#     print(f'  Dusty Simple Disk: {F_simple} mJy')   