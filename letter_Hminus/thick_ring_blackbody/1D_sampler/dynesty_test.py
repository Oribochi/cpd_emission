# best fit dusty disk for the constant alpha case
import numpy as np
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define the log-likelihood function
# ---------------------------------------------------------
# Replace this with your actual model/data likelihood.
# Example: a Gaussian likelihood centered at mu_true with sigma_true.

mu_true = 2.5
sigma_true = 0.5

# Simulate some "observed" data for this example
np.random.seed(42)
data = np.random.normal(mu_true, sigma_true, size=50)

def loglikelihood(theta):
    """theta is a 1D array/tuple with a single parameter, e.g. the mean."""
    mu = theta[0]
    sigma = sigma_true  # assume known for simplicity
    logl = -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
    return logl

# ---------------------------------------------------------
# 2. Define the prior transform (uniform prior in 1D)
# ---------------------------------------------------------
# dynesty samples from the unit cube [0,1]^ndim, so you map
# u -> parameter value using the inverse CDF of your prior.
# For a uniform prior U(a, b): theta = a + u * (b - a)

a, b = -10.0, 10.0  # bounds of the uniform prior

def prior_transform(u):
    theta = np.zeros_like(u)
    theta[0] = a + u[0] * (b - a)
    return theta

# ---------------------------------------------------------
# 3. Run the Dynamic Nested Sampler
# ---------------------------------------------------------
ndim = 1

dsampler = dynesty.DynamicNestedSampler(
    loglikelihood,
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

# ---------------------------------------------------------
# 5. Plot the posterior
# ---------------------------------------------------------
plt.figure()
plt.hist(posterior_samples[:, 0], bins=50, density=True, alpha=0.7)
plt.axvline(mu_true, color='k', linestyle='--', label='true value')
plt.xlabel(r'$\mu$')
plt.ylabel('Posterior density')
plt.legend()
plt.show()

# Optional: dynesty's built-in trace/corner plots
fig, axes = dyplot.traceplot(results, labels=[r'$\mu$'])
plt.show()