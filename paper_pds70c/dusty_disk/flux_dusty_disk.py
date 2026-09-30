# The flux of the dusty disk
from dustyCPD import *
from numpy import exp

# With this we can calculate the temperature of brighness depending on the wavelength (if optically thick or thin)
# this depends on the opacity, we We adopt the mass opacity from Andrews et al. (2012), κ mm = zeta x 3.4 × (0.87mm/λ) cm2 g−1
# with zeta the dust-to-gas ratio (we assume 0.01)
def kappa_mm(wavelength,zeta=0.01):
    # we should receive the wavelength in mm
    return zeta*3.4*(0.87/wavelength)

# With the optical depth 
def tau_mm(R,wavelength,Mpdot,alpha,TISM=20,zeta=0.01):
    return 1/2*kappa_mm(wavelength,zeta)*Sigma(R,Mpdot,alpha,TISM)

# And the brightness temperature then depending on tau_mm > 0.5 or < 0.5 is optically thick or thin, the wavelength must be in mm
def T_b(wavelength,R,Mpdot,alpha,TISM=20,zeta=0.01):
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
def S(wavelength,R,Mpdot,alpha,TISM=20,zeta=0.01):
    # T_b receives the wavelength in mm
    Temp_b=T_b(wavelength,R,Mpdot,alpha,TISM,zeta)
    wav=wavelength*1e-1 # we need to convert to cm (wavelength in mm)    
    #return 2*kb*Temp_b/((wav)**2)*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    return plank(Temp_b, c/wav)*10**(26)


# Now we can calculate the total flux as the integral of the flux density over the disc in function of Mpdot and alpha
# The flux is in erg/s/cm^2/Hz
def F(wavelength,Mpdot,alpha,TISM=20,zeta=0.01):
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

# the simpler model for wavelength in mm
def S_thin(wavelength,R,gamma,zeta=0.01):
    # gamma is the ratio between Mpdot and alpha
    T_b_thin=kappa_mm(wavelength,zeta)*sqrt(G*Mp/(pi**2*R**3))*gamma*(Mj/yr)*(mu/Rgas)**(5/4)*(1-sqrt(Rin/R))**(1/2)
    #S_thin=2*kb*T_b_thin/(wavelength*1e-1)**2*1e26 # this was in erg/s/cm^2/Hz/ster we pass to mJy
    S_thin=plank(T_b_thin, c/(wavelength*10**(-1)))*10**(26)
    return S_thin

def F_thin(wavelength,gamma,zeta=0.01):
    # gamma is the dimenssionless ratio between Mpdot and alpha
    flux=[]
    # if it's a list of wavelengths we calculate the flux for each en
    try:
        for wav in wavelength:
            # Importante calcular para un r unico pues la funcion S depende de T_b que tiene un if i funciona de a un r
            flux.append(np.trapezoid([S_thin(wav, r, gamma,zeta)*2*pi*r/dis**2 for r in R], R))
        return flux
            
    # if it's a single wavelength we calculate the flux for that wavelength
    except:
        return np.trapezoid([S_thin(wavelength, r, gamma, zeta)*2*pi*r/dis**2 for r in R], R)

######################################################################################################################
######################################################################################################################

# Adding the jet emission

def jet_1cm(Mpdot):
    Lbol=Lacc(Mpdot)
    flux_mJy=0.008*(Lbol/Lsun)**(0.6)*(1000*pc/dis)**2
    return flux_mJy # The flux in mJy

# Now a function that calculates the flux in other wavelenght using the spectral index of the jet
def jet_flux_lambda(spec_ind, wavelength, Mpdot):
    # we have the flux in cm and the wavelength is in mm
    flux_1cm=jet_1cm(Mpdot)
    flux_wavelength=(10/wavelength)**(spec_ind)*flux_1cm
    return flux_wavelength