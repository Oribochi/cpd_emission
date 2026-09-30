# now for any parameter array we can calculate the error
import numpy as np

def error_param(data, data_max, bin_inf=-15, bin_sup=15, N=1000):
    array=np.zeros(N)
    bin=bin_inf
    delta=(bin_sup - bin_inf)/len(array)
    for i in range(len(array)):
        array[i]=len(data[(data>bin) & (data<bin+delta)])
        bin+=delta

    # where we have the maximun
    bin_max=int((data_max-bin_inf)/delta)
    # now i want to know where is the 68% of the data
    sum=0
    i=0
    normalization=float(np.sum(array[bin_max:]))
    while sum<0.68*normalization:
        sum+=array[i+bin_max]
        i+=1

    # to the other side
    sum=0
    j=0
    normalization=float(np.sum(array[:bin_max]))
    while sum<0.68*normalization:
        sum+=array[bin_max-j]
        j+=1
    
    # here we just return the erros: x - error1, x + error2, (we return -error1 and error2)
    return (bin_max-j)*delta+bin_inf-data_max, (i+bin_max)*delta+bin_inf-data_max

def upper_limit(data, under_bound=-15, bin_sup=15, N=1000):
    array=np.zeros(N)
    bin=float(under_bound)
    delta=(bin_sup-under_bound)/len(array)
    print(delta)
    for i in range(len(array)):
        array[i]=len(data[(data>bin) & (data<bin+delta)])
        bin+=delta
        
    # now i want to know where is the 99% of the data
    sum=0
    i=N-1
    normalization=float(np.sum(array))
    while sum<0.005*normalization:
        sum+=array[i]
        i-=1

    return (i)*delta+under_bound

def strict_upper_limit(data, bin_inf=-15, bin_sup=15, N=1000):
    array=np.zeros(N)
    delta=(bin_sup - bin_inf)/len(array)
    bin=bin_inf
    for i in range(len(array)):
        bin+=delta
        array[i]=len(data[(data>bin) & (data<bin+delta)])
        if array[i]==0:
            # print('Upper limit zeta =', (i)*30/len(array)-15)
            break
    return (i)*delta+bin_inf