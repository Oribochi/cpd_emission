# We will calculate the range where magnetospheric accretion occurs using teh truncation radius and the radius where Tc>1000
# we ask that R_T>R_1000 and that additionally that the high temperature maintains T_c(R_T)>1000K
from dustyCPD import *
from magneticCPD import R_truncation, R_T1000, alpha_calculator
import matplotlib.pyplot as plt



# For a given Mpdot, Calculate the range of Bps where magnetospheric accretion is possible
def magnetospheric_accretion_range(Mpdot, Bps_min=1, Bps_max=1000):
    Bps_arr=np.linspace(Bps_min, Bps_max, 1000) # Gauss
    R_trunc_arr=np.zeros(len(Bps_arr))
    R_T1000_arr=np.zeros(len(Bps_arr))
    T_c_R_T_arr=np.zeros(len(Bps_arr))
    for i in range(len(Bps_arr)):
        Bps=Bps_arr[i]
        R_trunc_arr[i]=R_truncation(Mpdot,Bps)
        R_T1000_arr[i]=R_T1000(Mpdot,Bps)
        T_c_R_T_arr[i]=T_c_approx(R_trunc_arr[i], Mpdot, alpha_calculator(R_trunc_arr[i],Mpdot,Bps))
    
    # Now we check the conditions for magnetospheric accretion
    mask1=R_trunc_arr>R_T1000_arr
    mask2=T_c_R_T_arr>1000
    mask3=R_trunc_arr>Rin # we also want to avoid cases where the truncation radius is smaller than the inner radius of the disk
    mask_final=mask1 & mask2 & mask3
    
    Bps_range=Bps_arr[mask_final]
    if len(Bps_range)==0:
        return []
    else:
        return Bps_range[0], Bps_range[-1]

# examples 
print(magnetospheric_accretion_range(1e-6*Mj/yr, 1,400))
print(magnetospheric_accretion_range(10**(-5.5)*Mj/yr, 100,1000))
print(magnetospheric_accretion_range(10**(-5)*Mj/yr, 200,1000))
# print(magnetospheric_accretion_range(10**(-7.5)*Mj/yr, 1,1000))
# print(magnetospheric_accretion_range(10**(-8)*Mj/yr, 1,1000))

# finding the minimun Mpdot for which magnetospheric accretion is possible
def min_Mpdot_for_magnetospheric_accretion(Mpdot_min=10**(-6.5)*Mj/yr, Mpdot_max=1e-6*Mj/yr):
    Mpdot_arr=np.logspace(log10(Mpdot_min), log10(Mpdot_max), 20)
    for Mpdot in Mpdot_arr:
        Bps_range=magnetospheric_accretion_range(Mpdot,0,100)
        if len(Bps_range)>0:
            return Mpdot
    return None

# Min_Mpdot=min_Mpdot_for_magnetospheric_accretion()

# Bps_range=magnetospheric_accretion_range(Min_Mpdot,0,100)

# print("Minimum Mpdot for magnetospheric accretion:", Min_Mpdot/Mj*yr, "Bps range:", Bps_range)


# 6.951927961775606e-07 Bps range: (np.float64(69.96996996996998), np.float64(70.87087087087087))
print(magnetospheric_accretion_range(6.95e-07*Mj/yr, 65,75))