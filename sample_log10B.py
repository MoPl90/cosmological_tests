import numpy as np
from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from schwimmbad import MPIPool
from Cosmology_log10B import *
import sys

#KMA 15 Nov 19: updated parameters in correspondence with current jupyter nb
#KMA 11 Jan 20: added option to use numerical or analytical formulae for z_d and r_d
#KMA 24 Jun 20: introduced log10 sampling for B parameters in bigravity and a variable which contains the sign of B


#########################################################################

#Parameters for BAO and CMB
z_d = 1089
h = .6727
cLight=3E5
omega_baryon_preset = 0.02235/h**2
omega_gamma_preset = 2.469E-5/h**2

#Cosmological parameters
Omega_m_preset = 0.3166
#Omega_r_preset = 2.469E-5/h**2 # this is the photon density
Omega_c_preset = 0.6834

#Whether to use the numerical approximation for the comoving sound horizon, see arXiv:1411.1074:
rd_num = False
#Whether to use the numerical approximation for the redshift of last scattering and end of drag epoch,
#cf arXiv:astro-ph/9510117, or use z_d = z_star = 1089:
z_num = True

#########################################################################


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


# #BAO data
# # load all data points except for WiggleZ
# dataBAO = np.loadtxt('data/BOSS.txt', usecols=(1,3))
# # add WiggleZ
# dataBAO = np.append(dataBAO, np.loadtxt('data/WiggleZ.txt', usecols=(1,3)), axis=0)

# # the error for BAO is a cov mat, due to the addition of the WiggleZ data.
# errBAO  = np.pad(np.diag(np.loadtxt('data/BOSS.txt', usecols=4)), [(0, 3), (0, 3)], mode='constant', constant_values=0)
# # now add the WiggleZ cov mat. Note that this is the sqrt, as all the other errors are given in this format, too.
# errBAO += np.pad(np.sqrt(np.loadtxt('data/WiggleZ_cov.txt', usecols=(3,4,5))), [(len(errBAO)-3, 0), (len(errBAO)-3, 0)], mode='constant', constant_values=0)

# typeBAO = np.genfromtxt('data/BOSS.txt',dtype=str, usecols=2)
# typeBAO = np.append(typeBAO, np.genfromtxt('data/WiggleZ.txt',dtype=str, usecols=2), axis=0)

# BAOdata = BAO_data(dataBAO, errBAO, np.array([omega_baryon_preset, omega_gamma_preset]), typeBAO)
#load BOSS data points
dataBAO = np.loadtxt('data/BOSS.txt', usecols=(1,3))
#load BOSS errors & measurement quantity
errBAO  = np.diag(np.loadtxt('data/BOSS.txt', usecols=4))
typeBAO = np.genfromtxt('data/BOSS.txt',dtype=str, usecols=2)

#BOSS DR12 covariance matrix from 1607.03155
sigmaDR12 = np.diag(np.diag(errBAO)[np.where(np.loadtxt('data/BOSS.txt', usecols=0, dtype=str) == 'BOSS_DR12')])
corrDR12 = np.tril(np.loadtxt('data/BOSS_DR12_cov.txt', usecols=(1,2,3,4,5,6))*1E-4)
corrDR12 += corrDR12.T
corrDR12 -= np.eye(len(corrDR12))
CovDR12 = np.dot(sigmaDR12, np.dot(corrDR12, sigmaDR12))

#eBOSS Quasar covariance matrix from 1801.03043
sigmaeBOSS = np.diag(np.diag(errBAO)[np.where(np.loadtxt('data/BOSS.txt', usecols=0, dtype=str) == 'eBOSS_QSO')])
correBOSS = np.triu(np.loadtxt('data/eBOSS_QSO_cov.txt', usecols=(1,2,3,4,5,6,7,8))*1E-4)
correBOSS += correBOSS.T
correBOSS -= np.eye(len(correBOSS[0]))
CoveBOSS = np.dot(sigmaeBOSS, np.dot(correBOSS, sigmaeBOSS))

#assemble the full covariance matrix
load_cov = np.diag([1 if i else 0 for i in np.loadtxt('data/BOSS.txt',usecols=5)==1]) 
CovBAO = (errBAO - np.dot(load_cov,errBAO))**2
#CovBAO = (errBAO)**2
CovBAO += np.pad(CovDR12,[(2,len(errBAO)-2-len(CovDR12)),(2,len(errBAO)-2-len(CovDR12))], mode='constant', constant_values=0)
CovBAO += np.pad(CoveBOSS,[(2 + len(CovDR12) + 1,len(errBAO)- 3 - len(CovDR12) - len(CoveBOSS)),(2 + len(CovDR12) + 1,len(errBAO)- 3 - len(CovDR12) - len(CoveBOSS))], mode='constant', constant_values=0)

#Finally, add the correlations given in 1904.03430 
add_cov = np.array([i if i<0 else 0 for i in np.loadtxt('data/BOSS.txt',usecols=5)])
Cov = np.pad(np.diag(add_cov[1:]), [(1,0),(0,1)], mode='constant', constant_values=0) + np.pad(np.diag(add_cov[1:]), [(0,1), (1,0)], mode='constant', constant_values=0)
for ind in np.array([np.where(add_cov<0)[0]-1., np.where(add_cov<0)[0]], dtype=int).T:
    Cov[ind] = Cov[ind] * np.diag(errBAO)[ind[0]] * np.diag(errBAO)[ind[1]]
    
CovBAO += Cov

BAOdata = BAO_data(dataBAO, CovBAO, typeBAO)



model = sys.argv[-1]


#CMB Planck 2018 data
CMBdata = CMB_data(model, 'Planck18')

#Sign for bigravity parameter scans
B_signs = None



if not model in ['LCDM', 'kLCDM', 'wLCDM', 'conformal', 'bigravity', 'kbigravity']:
    raise(NameError('Specify a cosmological model as the last argument.'))

ranges_min = np.array([0, 0, 60., -5, -10, -30, -.5, 0, 0]) #Omegam, Omegab, H0, alpha, beta, MB, delda_M, beta_prime, s
ranges_max = np.array([1, 0.1, 80., 5, 10, -10, .5, 10, 3]) #Omegam, Omegab, H0, alpha, beta, MB, delda_M, beta_prime, s

if model == 'kLCDM' or model == 'wLCDM': #insert Omegac prior
    ranges_min = np.insert(ranges_min, 1, 0)
    ranges_max = np.insert(ranges_max, 1, 1.5)

if model == 'wLCDM': #insert w prior
    ranges_min = np.insert(ranges_min, -6, -2.5)
    ranges_max = np.insert(ranges_max, -6, -1/3.)

elif model == 'conformal': #replace Omegam, Omegac -> gamma0, kappa priors
    #ranges_min[2] = np.insert(ranges_min, 1, 0) #gamma0 range identical to Omegam
    #ranges_max[2] = 2500 
    #ranges_max[0] = 10 
    
    ranges_min = np.insert(ranges_min, 1, 0) #gamma0 range identical to Omegam
    ranges_max = np.insert(ranges_max, 1, 100) #gamma0 range identical to Omegam
    
elif model == 'bigravity': #add Bigravity model priors
    ranges_min = np.append([-10,-10,-10], ranges_min)
    ranges_max = np.append([ 10, 10, 10], ranges_max)
    
    B_signs = np.fromstring(sys.argv[-2], dtype=int, sep=',')
    
elif model == 'kbigravity': #add Bigravity model priors and Omagc
    ranges_min = np.insert(ranges_min, 1, 0)
    ranges_max = np.insert(ranges_max, 1, 1.5)
    ranges_min = np.append([-10,-10,-10], ranges_min)
    ranges_max = np.append([ 10, 10, 10], ranges_max)
    
    B_signs = np.fromstring(sys.argv[-2], dtype=int, sep=',')
    
data_sets = []
if 'SN' in sys.argv:
    data_sets.append(SNdata)
if 'Quasars' in sys.argv:
    data_sets.append(Qdata)
if 'BAO' in sys.argv:
    data_sets.append(BAOdata)
if 'CMB' in sys.argv:
    data_sets.append(CMBdata)
if 'RC' in sys.argv:
    data_sets.append(RCdata)
     
def Likelihood(theta): 
    if model == 'bigravity' or model == 'kbigravity': l = likelihood(theta, data_sets, ranges_min, ranges_max, model = model, rd_num = rd_num, z_num = z_num, B_signs=B_signs)
    else: l = likelihood(theta, data_sets, ranges_min, ranges_max, model = model, rd_num = rd_num, z_num = z_num)
    return l.logprobability_gauss_prior()

ndim, nwalkers, nsteps = len(ranges_min), 512, 1000

pool= MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
    
#B signs to sring for name of h5 file:
if model == 'bigravity' or model == 'kbigravity': Bsgn_string = ''.join([str(sign) for sign in B_signs])

name = model + '_'

if model == 'bigravity' or model == 'kbigravity':
    name += 'log10B_B' + Bsgn_string + '_'

for data_sample in sys.argv[1:-1]:
    name += data_sample + '_'
    
    




h5chain = HDFBackend('chains/' + name + str(nwalkers) + 'x' + str(nsteps) + '.h5')

try:
    pos0 = h5chain.get_chain()[-1] #continue existing chains...
    nsteps -= len(h5chain.get_chain()) #...for missing steps only
    if nsteps <= 0:
        raise ValueError("Chain is already complete.")

    sampler = EnsembleSampler(nwalkers, ndim, Likelihood, pool=pool, backend=h5chain)

    sampler.run_mcmc(pos0, nsteps)
    pool.close()

except AttributeError:
    pos0 = np.random.uniform(ranges_min, ranges_max,(nwalkers,len(ranges_max)))

    sampler = EnsembleSampler(nwalkers, ndim, Likelihood, pool=pool, backend=h5chain)

    sampler.run_mcmc(pos0, nsteps)
    pool.close()
except ValueError:
    pool.close()





