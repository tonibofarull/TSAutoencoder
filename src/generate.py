import numpy as np
import pandas as pd
import statsmodels.api as sm

"""
Output must be a numpy array
"""

def sin_cos(n_images, length):
    docs = []
    for _ in range(n_images):

        ini = np.random.random()*30
        x = np.array(range(length)) + ini
        y = np.sin(x*2*np.pi/15) + np.sin(1.5*x*2*np.pi/15)
        y = y + 1/10 * np.random.randn(length)  # noise

        docs.append(y.reshape(-1,1))
    docs = np.array(docs, dtype=np.float32).reshape(-1, 1, length)
    return docs

def arma(n_images, length, ar=None, ma=None):
    docs = []
    if ar is None and ma is None:
        ar = [-0.1, 0,0,0,0,0,0, 0.5] # coef of y_{t-1}, y_{t-2}, ...
        ma = [ 0.7, 0,0,0,0,0,0, 0.3] # coef of e_{t-1}, e_{t-2}, ...

    ar = np.r_[1, -np.array(ar)]
    ma = np.r_[1,  np.array(ma)]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    for _ in range(n_images):
        y = arma_process.generate_sample(nsample=length)
        docs.append(y.reshape(-1,1))
    docs = np.array(docs, dtype=np.float32).reshape(-1, 1, length)
    return docs

def wind(col=9, num_elems=64):
    df = pd.read_csv("../data/110918.TXT", sep=",", header=None)
    x = df[col].to_numpy().astype(np.float32)
    num = x.shape[0]//num_elems
    return x[:num*num_elems].reshape((num,1,num_elems))
