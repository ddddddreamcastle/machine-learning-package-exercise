import autograd.numpy as np

def standardization(data):
    data_avg = np.average(data)
    data_std = np.std(data)
    return (data - data_avg) / data_std, data_avg, data_std

def onehot_decode_for_label(y):
    num_classes = np.max(y) + 1
    y = np.array(y).reshape(-1)
    return np.eye(num_classes)[y], num_classes
