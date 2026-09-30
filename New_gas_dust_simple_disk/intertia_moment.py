from units_astro import *

# the moment of inertia of a uniform sphere is 2/5 M R^2
def moment_of_inertia(M, R):
    return 2/5 * M * R**2

# for a protostelar could with 1 Msun and 1 AU radius, the moment of inertia is
M = Msun
R = 10000*au
I = moment_of_inertia(M, R)
print(f"The moment of inertia of a protostellar cloud with 1 Msun and 1 AU radius is: {I:.2e} kg m^2")

# for a start is 1Rsun
R_star = Rsun
I_star = moment_of_inertia(M, R_star)
print(f"The moment of inertia of a star with 1 Msun and 1 Rsun radius is: {I_star:.2e} kg m^2")

# the ratio
ratio = I / I_star
print(f"The ratio of the moment of inertia of the protostellar cloud to the star is: {ratio:.2e}")