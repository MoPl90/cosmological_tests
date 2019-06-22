#!/usr/bin/python

import numpy
from astropy.io import fits as pyfits
import glob

def mu_cov(alpha, beta):
    """ Assemble the full covariance matrix of distance modulus

    See Betoule et al. (2014), Eq. 11-13 for reference
    """
    Ceta = sum([pyfits.getdata(mat) for mat in glob.glob('covmat/C*.fits')])

    Cmu = numpy.zeros_like(Ceta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            Cmu += (coef1 * coef2) * Ceta[i::3,j::3]

    # Add diagonal term from Eq. 13
    sigma = numpy.loadtxt('covmat/sigma_mu.txt')
    sigma_pecvel = (5 * 150 / 3e5) / (numpy.log(10.) * sigma[:, 2])
    Cmu[numpy.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    
    return Cmu


if __name__ == "__main__":
    Cmu = get_mu_cov(0.13, 3.1)
