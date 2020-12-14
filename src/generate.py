import numpy as np
import statsmodels.api as sm

"""
Output must be a numpy array
"""

def arma(n_images, length, ar, ma):
    docs = []

    ar = np.r_[1, -np.array(ar)]
    ma = np.r_[1,  np.array(ma)]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    for _ in range(n_images):
        y = arma_process.generate_sample(nsample=length)
        docs.append(y.reshape(-1,1))
    docs = np.array(docs, dtype=np.float32).reshape(-1, 1, length)
    return docs
