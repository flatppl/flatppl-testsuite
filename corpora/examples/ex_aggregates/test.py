"""NumPy matrix products, reductions and unbiased sample variance."""

import numpy as np


def oracle(point):
    a = np.asarray(point["A"])
    b = np.array([[1,0],[0,1],[1,1]])
    c = a @ b
    d = np.sum((a[:,:,None]-b[None,:,:])**2 * np.array([1,2,1])[None,:,None], axis=1)
    v = np.var(a, axis=0, ddof=1)
    s = a[:,0]
    p = np.prod(a[:,:,None]+b[None,:,:], axis=1)
    return float(c.sum()+2*d.sum()+3*v.sum()+4*s.sum()+5*p.sum())
