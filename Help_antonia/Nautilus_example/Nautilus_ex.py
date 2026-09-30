# Nautilus example of usage

import numpy as np
import matplotlib.pyplot as plt
from nautilus import Prior
from nautilus import Sampler
import corner
from numpy import exp, sin, sqrt

# example of random data that we want to fit
x_data1=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_data1=[0.1, 3, 2.3, 2.1, 1.9, 3.1, 2.7, 2.5, 2.8, 3.2]
y_error1=[0.1, 0.05, 0.2, 0.15, 0.1, 0.05, 0.2, 0.15, 0.1, 0.05]

# my model function that I want to fit
def model(x, a , b, c):
    return exp(a)*x+sin(b*sqrt(x))+c

# Now we define the log-likelihood function that receives a dictionary of parameters (in this case a and b) and 
# returns the log-likelihood value
def log_likelihood(param_dict):
    # the parameters
    a = param_dict['a']
    b = param_dict['b']
    c = param_dict['c']

    # compute the model values
    ymodels = np.zeros(len(y_data1))

    # and the Xi-squared values (this is how much the model deviates from the data squared normalized by its errors
    # and we want to minimize this)
    Xis = np.zeros(len(y_data1))

    for i in range(len(y_data1)):
        ymodels[i] = model(x_data1[i], a, b, c) 
        Xis[i] = -0.5*((y_data1[i]-ymodels[i])/y_error1[i])**2

    # return the log-likelihood value
    return np.sum(Xis)

# now we define the prior ranges for each parameter, in this case uniform priors between given ranges because we 
# don't have any prior knowledge about them (this is the most conservative approach)
prior = Prior()
prior.add_parameter('a', dist=(-5, 5))
prior.add_parameter('b', dist=(0.01, 10))
prior.add_parameter('c', dist=(-5, 5))

# now we run the sampler with 1000 live points and using 4 parallel processes (you can adjust this pool number 
# depending on your CPU cores availability)
sampler = Sampler(prior, log_likelihood, n_live=1000, pool = 4)
# and we run it
sampler.run(verbose=True)

# We can now plot the corner plot with the results
points, log_w, log_l = sampler.posterior()

figure = plt.figure(figsize=(8, 8))

corner.corner(points, weights=np.exp(log_w), bins=20, labels=['a', 'b', 'c'], color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)), fig=figure, labelpad=0.05) #, label_kwargs={'fontsize': 20}

plt.show()

# And we can also plot the best-fit model against the data
best_index = np.argmax(log_l)
best_a = points[best_index][0]
best_b = points[best_index][1]
best_c = points[best_index][2]

x_for_plot = np.linspace(0, 10, 100)
y_bestmodel = np.zeros(len(x_for_plot))
for i in range(len(x_for_plot)):
    y_bestmodel[i] = model(x_for_plot[i], best_a, best_b, best_c) # compute the best-fit model values

plt.errorbar(x_data1, y_data1, yerr=y_error1, fmt='o', label='data')
plt.plot(x_for_plot, y_bestmodel, label='best-fit model', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Extra:

######### 1| Save/Load the results #########
# If you want to save the likelihood, weights and points directly:
# np.save('logZ1.npy', log_l)
# np.save('logW1.npy', log_w)
# np.save('logP.npy', points)

# points=np.load('logP.npy')
# log_w=np.load('logW1.npy')
# log_l=np.load('logZ1.npy')

######### 2| Compute reduced Chi-squared value #########
# If you want to see how good is your model you usually compute the reduced Chi-squared value:
# dof = len(y_data1) - len(prior.keys) # degrees of freedom
# best_Xi2 = -2*log_l[best_index] # best Xi-squared value
# reduced_Xi2 = best_Xi2/dof
# print("Reduced Xi-squared value of the best-fit model: ", reduced_Xi2) 

# should be around 1 for a good fit in this case is not
# if it is much larger than 1 it means that the model is not a good fit to the data, 
# if it is much smaller than 1 it means that we are overfitting the data

######### 3| when prior information you want to use is in log-space #########
# if you have parameters like the viscosity-alpha parameter in accretion disks that are
# known to be between 1e-4 and 1e-2 you can define the prior in log-space like this:
# prior.add_parameter('log_alpha', dist=(-4, -2))
# and then in the log_likelihood function you convert it back to linear space:
# alpha = 10**param_dict['log_alpha']