# This is to test the degeneracy of the model of Zhu for syntetic data Ṁp = 10−9 MJup yr−1 and α = 10−3. and  Ṁp = 10−5 MJup yr−1 and α = 10−3
import numpy as np
from units_astro import *
from flux_dusty_disk import F as F_zhu
from nautilus import Prior, Sampler
import corner
import matplotlib.pyplot as plt
from errors import error_param, upper_limit, strict_upper_limit

Mpdot1=1e-9*Mj/yr
alpha1=1e-3

nus1=np.array([97.5E9, 145005402197.7, 343.5E9, 671E9])
Fnus1=np.array([F_zhu(10*c/nu, Mpdot1, alpha1) for nu in nus1])
sFnus1=0.1*Fnus1 # 10% error

print(Fnus1)

# now we use the nautilus sampler to try to fit the model to the syntetic data

prior = Prior()
prior.add_parameter('log_Mpdot', dist=(-15, 0))
prior.add_parameter('log_alpha', dist=(-10, 0))

def log_likelihood_F(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus1))
    Xis = np.zeros(len(Fnus1))
    for i in range(len(Fnus1)):
        ymodels[i] = F_zhu(10*c/nus1[i], Mpdot*Mj/yr, alpha,zeta=0.01)
        Xis[i] = -0.5*((Fnus1[i]-ymodels[i])/sFnus1[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

sampler_zhu = Sampler(prior, log_likelihood_F, n_live=1000, pool = 16)
sampler_zhu.run(verbose=True)

points, log_w, log_l = sampler_zhu.posterior()
corner.corner(points, weights=np.exp(log_w), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\alpha)$'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# Save the log Z and the corner plot
np.save('logZ1_zhu.npy', log_l)
np.save('logW1_zhu.npy', log_w)
np.save('logP1_zhu.npy', points)

plt.show()

# To see equal weights
points_equalw, log_w_equalw, log_l_equalw = sampler_zhu.posterior(equal_weight=True)

# now making our errors
Mpdot_maxl, alpha_maxl = points[np.argmax(log_l)]
Mpdot_min, Mpdot_max = error_param(points_equalw[:,0], Mpdot_maxl, bin_inf=-15, bin_sup=0)
alpha_min, alpha_max = error_param(points_equalw[:,1], alpha_maxl, bin_inf=-10, bin_sup=0)

alpha_upper = upper_limit(points_equalw[:,1], -9)
alpha_upper_strict = strict_upper_limit(points_equalw[:,1])

print('Mpdot =', Mpdot_maxl, Mpdot_min, Mpdot_max)
print('alpha =',  alpha_maxl, alpha_min, alpha_max)
print('alpha upper limit =', alpha_upper)
print('alpha = ', alpha_maxl, alpha_upper-alpha_maxl)
print('alpha upper strict limit =', alpha_upper_strict)
print('log_likelihood max =', np.max(log_l))

# save flux and nus for the best fit
Mpdot_best1=10**points[np.argmax(log_l)][0]*Mj/yr
alpha_best1=10**points[np.argmax(log_l)][1]
fluxes1 = np.zeros(len(nus1))
for i in range(len(nus1)):
    fluxes1[i]=F_zhu(10*c/nus1[i], Mpdot_best1, alpha_best1, zeta=0.01)

fluxes1=np.array(fluxes1)

np.savetxt('fluxes_zhu.txt', np.column_stack((nus1, fluxes1)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')

# Now the same but with Ṁp = 10−5 MJup yr−1 and α = 10−3
Mpdot2=1e-5*Mj/yr
alpha2=1e-3

nus2=np.array([97.5E9, 145005402197.7, 343.5E9, 671E9])
Fnus2=np.array([F_zhu(10*c/nu, Mpdot2, alpha2) for nu in nus2])
sFnus2=0.1*Fnus2 # 10% error

prior = Prior()
prior.add_parameter('log_Mpdot', dist=(-15, 0))
prior.add_parameter('log_alpha', dist=(-10, 0))

def log_likelihood_F2(param_dict):
    Mpdot = 10**param_dict['log_Mpdot']
    alpha = 10**param_dict['log_alpha']

    ymodels = np.zeros(len(Fnus2))
    Xis = np.zeros(len(Fnus2))
    for i in range(len(Fnus2)):
        ymodels[i] = F_zhu(10*c/nus2[i], Mpdot*Mj/yr, alpha,zeta=0.01)
        Xis[i] = -0.5*((Fnus2[i]-ymodels[i])/sFnus2[i])**2

    return np.sum(Xis)

sampler_zhu2 = Sampler(prior, log_likelihood_F2, n_live=1000, pool = 16)
sampler_zhu2.run(verbose=True)
points2, log_w2, log_l2 = sampler_zhu2.posterior()
corner.corner(points2, weights=np.exp(log_w2), bins=20, labels=[r'$\log\left( \frac{dM_p/dt}{M_j/yr} \right)$', r'$\log(\alpha)$'], color='C3',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fontsize=12)

# save log Z and corner plot for the second case
np.save('logZ2_zhu.npy', log_l2)
np.save('logW2_zhu.npy', log_w2)
np.save('logP2_zhu.npy', points2)

plt.show()

# To see equal weights
points2_equalw, log_w2_equalw, log_l2_equalw = sampler_zhu2.posterior(equal_weight=True)

# now making our errors for the second case
Mpdot2_maxl, alpha2_maxl = points2[np.argmax(log_l2)]
Mpdot2_min, Mpdot2_max = error_param(points2_equalw[:,0], Mpdot2_maxl, bin_inf=-15, bin_sup=0)
alpha2_min, alpha2_max = error_param(points2_equalw[:,1], alpha2_maxl, bin_inf=-10, bin_sup=0)

alpha2_upper = upper_limit(points2_equalw[:,1], -9)
alpha2_upper_strict = strict_upper_limit(points2_equalw[:,1])

print('Mpdot =', Mpdot2_maxl, Mpdot2_min, Mpdot2_max)
print('alpha =',  alpha2_maxl, alpha2_min, alpha2_max)
print('alpha upper limit =', alpha2_upper)
print('alpha = ', alpha2_maxl, alpha2_upper-alpha2_maxl)
print('alpha upper strict limit =', alpha2_upper_strict)
print('log_likelihood max =', np.max(log_l2))

# Best fit for the second case:

Mpdot_best2=10**points2[np.argmax(log_l2)][0]*Mj/yr
alpha_best2=10**points2[np.argmax(log_l2)][1]
fluxes2 = np.zeros(len(nus2))
for i in range(len(nus2)):
    fluxes2[i]=F_zhu(10*c/nus2[i], Mpdot_best2, alpha_best2, zeta=0.01)

fluxes2=np.array(fluxes2)

np.savetxt('fluxes2_zhu.txt', np.column_stack((nus2, fluxes2)), header='nu [GHz] F_nu [mJy]', fmt='%f %f')
