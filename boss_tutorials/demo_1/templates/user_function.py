import numpy as np

def f(x):
    "Analytic function: f(x) = sin(x) + 1.5*exp(-(x-4.3)**2), 0<x<7"
    return np.sin(x[0][0]) + 1.5*np.exp(-(x[0][0]-4.3)**2.)

