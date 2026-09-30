# This is to generate the extrapolation using diferent power law functions.

# Has the temperature at the reference radius, the reference radius, the power law index and the radius to extrapolate to.
def extrapolation_T(T0, r0, gamma, r): 
    return T0 * (r / r0) ** (gamma)

# for pds70c from the dust diagnostics:
T_1 = 21 # K
r_1 = 0.5 # arcsec
gamma1 = -0.5 # power law index
T_2 = 15.5 # K
r_2 = 0.6 # arcsec
gamma2 = -1.7

r_pds70c = 0.3 # arcsec

# Extrapolating the temperature at the reference radius to the radius of pds70c using the first power law index.
T_pds70c_1 = extrapolation_T(T_1, r_1, gamma1, r_pds70c)

# Extrapolating the temperature at the reference radius to the radius of pds70c using the second power law index.
T_pds70c_2 = extrapolation_T(T_2, r_2, gamma2, r_pds70c)

# Printing the results
print(f"Extrapolated temperature at {r_pds70c} arcsec using gamma1 ({gamma1}): {T_pds70c_1:.2f} K")

print(f"Extrapolated temperature at {r_pds70c} arcsec using gamma2 ({gamma2}): {T_pds70c_2:.2f} K")

# diferences in the Wein displacement law:
# The Wein displacement law 

def wein_displacement(T):
    # Wien's displacement constant in meters Kelvin
    b = 2.897771955e-3
    # Calculate the peak wavelength in meters
    lambda_max = b / T
    return lambda_max

# comparing with the 22 K using the most conservative approach (Shibaike et al. 2026)

wein_22=wein_displacement(22)
wein_27=wein_displacement(27)
wein_50=wein_displacement(50)

print(f'Wein 22 K: {wein_22*1000:.2e} mm', f'Wein 27 K: {wein_27*1000:.2e} mm', f'Wein 50 K: {wein_50*1000:.2e} mm')

wein_7=wein_displacement(7)
print(f'Wein 7 K: {wein_7*1000:.2e} mm')