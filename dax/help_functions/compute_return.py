import pandas as pd
import numpy as np


def compute_return_h(y, r_type="log", h=1):

    # exclude first h observations
    y2 = y[h:]
    # exclude last h observations
    y1 = y[:-h]

    if r_type == "log":
        ret = np.concatenate(([np.nan]*h, 100 * (np.log(y2) - np.log(y1))))
    else:
        ret = np.concatenate(([np.nan]*h, 100 * (y2-y1)/y1))

    return ret
