# best fit dusty disk for the constant alpha case
import numpy as np
from numpy import sqrt
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from units_astro import *
from flux_ring import F

# ---------------------------------------------------------
# 1. Define the log-likelihood function
# ---------------------------------------------------------
# Replace this with your actual model/data likelihood.
# Example: a Gaussian likelihood centered at mu_true with sigma_true.

# First our data with the error bars BAND 7 on 2019: 118.5 ± 16.6
nu = [97.5E9, 145005402197.7, 343.5E9, 671E9] # [nu1, nu2] are the x-axis values , 671E9
Fnus = [12e-06, 21.4e-06, 0.000121, 0.000143] # [Fnu1, Fnu2] are the y-axis values in Jy , 0.000122
sFnus = [4.7e-06, 4.1e-06, 0.000013, 0.000115] # [sigma_Fnu1, sigma_Fnu2] are the y-axis errors in Jy , 0.000090

# To mJy
nu = np.array(nu)
Fnus = np.array(Fnus)*1E3
sFnus = np.array(sFnus)*1E3

T_floor=50

def log_likelihood_F(log_width):
    width = 10**log_width*au
    ymodels = np.zeros(len(Fnus))
    Xis = np.zeros(len(Fnus))
    for i in range(len(Fnus)):
        ymodels[i] = F(nu[i], 1.264*au, width, T_floor, 112.3*pc)[0]
        Xis[i] = -0.5*((Fnus[i]-ymodels[i])/sFnus[i])**2

    # ejemplo
    # the likehood is the product of the likelihoods for each data point
    # we assume the errors are gaussian
    # L1 = norm.pdf(Fnus[0]-ymodel1, 0, sFnus[0])
    # 
    # 
    # L2 = norm.pdf(Fnus[1]-ymodel2, 0, sFnus[1])
    return np.sum(Xis)

# ---------------------------------------------------------
# 2. Define the prior transform (uniform prior in 1D)
# ---------------------------------------------------------
# dynesty samples from the unit cube [0,1]^ndim, so you map
# u -> parameter value using the inverse CDF of your prior.
# For a uniform prior U(a, b): theta = a + u * (b - a)

a, b = -2, 0 # bounds of the uniform prior for log_width

def prior_transform(u):
    theta = np.zeros_like(u)
    theta[0] = a + u[0] * (b - a)
    return theta

# ---------------------------------------------------------
# 3. Run the Dynamic Nested Sampler
# ---------------------------------------------------------
ndim = 1

dsampler = dynesty.DynamicNestedSampler(
    log_likelihood_F,
    prior_transform,
    ndim=ndim,
    bound='multi',      # bounding method
    sample='rwalk'      # sampling method (fine for low dims too)
)

dsampler.run_nested()

# ---------------------------------------------------------
# 4. Extract results / posterior samples
# ---------------------------------------------------------
results = dsampler.results
results.summary()

# Equal-weighted posterior samples (resampled according to importance weights)
from dynesty.utils import resample_equal

weights = np.exp(results.logwt - results.logz[-1])
posterior_samples = resample_equal(results.samples, weights)

print("Posterior mean:", posterior_samples.mean())
print("Posterior std:", posterior_samples.std())

# print width mean +- std in au
log_width_mean = posterior_samples.mean()
log_width_minus = posterior_samples.mean() - posterior_samples.std()
log_width_plus = posterior_samples.mean() + posterior_samples.std()
print("Posterior mean width (au):", 10**log_width_mean, 10**log_width_plus - 10**log_width_mean, 
      10**log_width_minus - 10**log_width_mean)


def H(R, T):
    mu=2.3
    Mp=4*Mj
    c_s=sqrt(Rgas*T/(mu))
    Omega=sqrt(G*Mp/R**3)
    H=c_s/Omega
    return H

print('Scale height at 1 au (in au):', H(1*au, T_floor)/au)


# ---------------------------------------------------------
# 5. Plot the posterior with the mean
# ---------------------------------------------------------
plt.figure()
plt.hist(posterior_samples[:, 0], bins=50, density=True, alpha=0.7)
plt.axvline(posterior_samples.mean(), color='k', linestyle='--', label='posterior mean')
plt.xlabel(r'$\log_{10}(\text{width})$')
plt.ylabel('Posterior density')
plt.legend()
plt.savefig(f'posterior_width_{T_floor}.pdf'.format(T_floor=T_floor))
plt.show()

# Optional: dynesty's built-in trace/corner plots
fig, axes = dyplot.traceplot(results, labels=[r'$\log_{10}(\text{width})$'])
plt.show()

# and the best fit model with the data
nu_arr=np.linspace(nu[0], 2*nu[-1], 100)
fluxes_ring=np.zeros(len(nu_arr))
for i in range(len(fluxes_ring)):
    fluxes_ring[i]=F(nu_arr[i], 1.264*au, 10**posterior_samples.mean()*au, T_floor, 112.3*pc)

plt.figure()
plt.plot(nu_arr/1e9, fluxes_ring, label='best fit model')
plt.errorbar(nu/1e9, Fnus, yerr=sFnus, fmt='o', label='data')
plt.xlabel(r'$\nu$ (GHz)')
plt.ylabel(r'$F_\nu$ (mJy)')
plt.legend()
plt.loglog()
plt.savefig(f'best_fit_model_{T_floor}.pdf'.format(T_floor=T_floor))
plt.show()