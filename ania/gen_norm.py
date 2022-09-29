from scipy.special import gammaincinv, loggamma
from functools import lru_cache
import torch
import numpy as np
from ania.utils import device

import time

def sample_unnormalized(n, dim, fuzziness, corner_deflation):
    qnorm = np.random.random(n)

    norm = gammaincinv(fuzziness*dim, np.abs(qnorm)) ** (fuzziness)
    where = np.abs(norm)<=0.00
    norm[where] = (qnorm[where])**(1/dim)

    q = 2 * (np.random.random((n,dim)) - 0.5)
    x = np.sign(q) * gammaincinv(corner_deflation, np.abs(q)) ** (corner_deflation)
    where = np.abs(x)<=0.001
    x[where] = (q[where])

    div_by = np.linalg.norm(x,ord=1.0/corner_deflation,axis=1)
    where = div_by == 0.0
    div_by[where] = np.linalg.norm(x[where], ord=np.inf, axis=1)

    x_normed = x / (div_by[:,np.newaxis])

    return x_normed * norm[:,np.newaxis]


@lru_cache(maxsize=1024)
def get_std(dim, fuzziness, corner_deflation, n=100000):
    n_actual = int(n/dim)+1
    return sample_unnormalized(n_actual, dim, fuzziness, corner_deflation).std()

def _sample_gen_norm(n, dim, fuzziness, corner_deflation):
    if fuzziness == corner_deflation == 0.5:
        return torch.randn(size=(n,dim), device=device)
    if fuzziness == corner_deflation == 0.0:
        return (torch.rand((n,dim), device=device) -0.5)*2*np.sqrt(3) 
    if fuzziness == corner_deflation and dim > 1:
        return torch.Tensor(_sample_gen_norm(n*dim, 1, fuzziness, corner_deflation).reshape((n,dim)))
    return torch.Tensor(sample_unnormalized(n, dim, fuzziness, corner_deflation) / ( get_std(dim, fuzziness, corner_deflation)))

def sample_gen_norm(shape, fuzziness, corner_deflation):
    return _sample_gen_norm(1,np.prod(shape),fuzziness,corner_deflation).reshape(shape)

@lru_cache(maxsize=1024)
def make_log_partition(dim, fuzziness, corner_deflation):
    log_partition = (np.log(2)+loggamma(1 + corner_deflation))*dim - loggamma(1 + dim*corner_deflation)
    log_partition += loggamma(1+dim * fuzziness)
    return log_partition

def log_pdf_gen_norm(sample, fuzziness, corner_deflation):
    dim = sample.shape[-1]
    log_partition = make_log_partition(dim, fuzziness, corner_deflation)

    if corner_deflation == 0:
        p = np.inf
    else:
        p = 1/corner_deflation
    
    if fuzziness == 0:
        inv_fuzz = np.inf
    else:
        inv_fuzz = 1/corner_deflation
    
    std = get_std(dim, fuzziness, corner_deflation)
    return -torch.norm(sample * std,p=p)**inv_fuzz - log_partition + dim*np.log(std)

