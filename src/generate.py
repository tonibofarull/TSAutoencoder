import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

def salt_pepper(im, d, L, sp):
    for i in range(d):
        for j in range(L):
            if np.random.random() < sp:
                im[i, j] = 255-im[i, j]
    return im

"""
Each function produce a set of "n_images" images of size (d,L)
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

def arma(n_images, length):
    docs = []
    arparams = np.array([-0.1*0, 0,0,0,0,0,0, 0.5])
    maparams = np.array([ 0.7*0, 0,0,0,0,0,0, 0.3])
    ar = np.r_[1, -arparams]
    ma = np.r_[1,  maparams]
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
