import numpy as np


def calculate_distance(daxdata, h, j, k):

    # fixed parameters
    weight_vector = np.array([1/(k-i+1) for i in range(1, k+1)])
    last_k_obs = np.array(daxdata[f'LogRetLag{h}'].iloc[-k:])

    # get obs window
    starting_index = j-k+1
    obs_window = np.array(
        daxdata[starting_index:starting_index+k][f'LogRetLag{h}'])

    # calculate distance
    dist = np.sum(weight_vector*((last_k_obs-obs_window)**2))

    return dist
