import numpy as np
from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from schwimmbad import MPIPool
from Cosmology import *
import sys

#Parameters for BAO -- CHECK THESE VALUES
z_d = 1089
h = .7
omega_baryon_preset = 0.022765/h**2
omega_gamma_preset = 2.469E-5/h**2

#Cosmological parameters
Omega_m_preset = 0.3089
Omega_r_preset = 2.469E-5/h**2
Omega_c_preset = 0.6911


#SN data
dataSN = np.loadtxt('data/jla_lcparams.txt', usecols=(2,4,6,8,10))
errSN = np.loadtxt('data/jla_lcparams.txt', usecols=(5,7,9))[np.argsort(dataSN.T[0])]
dataSN = dataSN[np.argsort(dataSN.T[0])]

#best fit values found in JLA analysis arXiv:1401.4064
a = 0.14
b = 3.14
MB = -19.04
delta_Mhost = -.06

SNdata = Supernova_data(dataSN, errSN, np.array([a,b,MB, 0]))


#Quasar data
dataQ = np.loadtxt('data/quasar_data_RL.txt', usecols=(0,1,2,3,4))
errQ = np.loadtxt('data/quasar_data_RL.txt', usecols=5)[np.argsort(dataQ.T[0])]
dataQ = dataQ[np.argsort(dataQ.T[0])]

#best fit values found in Risaliti & Lusso, Nature Astronomy, 2018
beta_prime, s = 7.4, 1.5

Qdata = Quasar_data(dataQ, errQ, np.array([beta_prime, s]))

# ## RC data
gamma00, kappa0 = 0.0093, 95
RCdata = RC_data([gamma00, kappa0])


#BAO data
# load all data points except for WiggleZ
dataBAO = np.loadtxt('data/BOSS.txt', usecols=(1,3))
# add WiggleZ
dataBAO = np.append(dataBAO, np.loadtxt('data/WiggleZ.txt', usecols=(1,3)), axis=0)

# the error for BAO is a cov mat, due to the addition of the WiggleZ data.
errBAO  = np.pad(np.diag(np.loadtxt('data/BOSS.txt', usecols=4)), [(0, 3), (0, 3)], mode='constant', constant_values=0)
# now add the WiggleZ cov mat. Note that this is the sqrt, as all the other errors are given in this format, too.
errBAO += np.pad(np.sqrt(np.loadtxt('data/WiggleZ_cov.txt', usecols=(3,4,5))), [(len(errBAO)-3, 0), (len(errBAO)-3, 0)], mode='constant', constant_values=0)

typeBAO = np.genfromtxt('data/BOSS.txt',dtype=str, usecols=2)
typeBAO = np.append(typeBAO, np.genfromtxt('data/WiggleZ.txt',dtype=str, usecols=2), axis=0)

BAOdata = BAO_data(dataBAO, errBAO, np.array([omega_baryon_preset, omega_gamma_preset]), typeBAO)


model = sys.argv[-1]

if not model in ['LCDM', 'wLCDM', 'conformal', 'bigravity']:
    raise(NameError('Specify a cosmological model as the last argument.'))

ranges_min = np.array([0, 0, -5, -10, -30, -.5, 0, 0]) #Omegam, Omegac, Omegab, H0, alpha, beta, MB, delda_M, beta_prime, s
ranges_max = np.array([1, 1.5, 5, 10, -10, .5, 10, 3]) #Omegam, Omegac, Omegab, H0, alpha, beta, MB, delda_M, beta_prime, s

#if not 'BAO' in sys.argv: #delte Omegab and H0 prior
#    ranges_min = np.delete(ranges_min, [2,3])
#    ranges_max = np.delete(ranges_max, [2,3])

if model == 'wLCDM': #insert w prior
    ranges_min = np.insert(ranges_min, -6, -2.5)
    ranges_max = np.insert(ranges_max, -6, -1/3.)

elif model == 'conformal': #replace Omegam, Omegac -> gamma0, kappa priors
    ranges_min[:2] = 0, 50
    ranges_max[:2] = 1., 300
    
elif model == 'bigravity': #remove Omegac prior and add Bigravity model priors
    ranges_min = np.append([-33., 0], np.delete(ranges_min, 1))
    ranges_max = np.append([-28., np.pi/2.], np.delete( ranges_max, 1))
    

data_sets = []
if 'SN' in sys.argv:
    data_sets.append(SNdata)
if 'Quasars' in sys.argv:
    data_sets.append(Qdata)
if 'BAO' in sys.argv:
    data_sets.append(BAOdata)
if 'RC' in sys.argv:
    data_sets.append(RCdata)
     
def Likelihood(theta): 
    l = likelihood(theta, data_sets, ranges_min, ranges_max, model = model)
    return l.logprobability_flat_prior()

ndim, nwalkers, nsteps = len(ranges_min), 512, 1000
pos0 = np.random.uniform(ranges_min, ranges_max,(nwalkers,len(ranges_max)))

pool= MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

name = model + '_'

for data_sample in sys.argv[1:-1]:
    name += data_sample + '_'



h5chain = HDFBackend('chains/' + name + str(nwalkers) + 'x' + str(nsteps) + '.h5')

sampler = EnsembleSampler(nwalkers, ndim, Likelihood, pool=pool, backend=h5chain)

sampler.run_mcmc(pos0, nsteps)

pool.close()
