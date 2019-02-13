import autograd.numpy as np

def standardization(data):
    data_avg = np.average(data, axis=0)
    data_std = np.std(data, axis=0)
    return (data - data_avg) / data_std, data_avg, data_std