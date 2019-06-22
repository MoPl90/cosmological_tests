import numpy as np
import scipy.integrate as integrate
from covmat.covmat import mu_cov

class cosmology:
    """This class implements the base class for any cosmological model we study. More involved models inherit from this class."""
    
    H0 = 70.# the Hubble rate today in [km/s/Mpc]
    c = 3E5 # speed of light in km/s
    dH = c / H0 # Hubble length in Mpc
    
    
    def __init__(self, omegam, omegac, w=-1):
        """Initialise a cosmological model"""
        self.Omegam = omegam #non-relativistic matter energy density
        self.Omegac = omegac #dark energy density
        self.Omegak = 1 - omegam - omegac #curvature energy density
        self.eos = w #the dark energy equation of state
        
    def set_energy_densities(self, omegam, omegac):
        """Change the initialised values of Omega_i."""
        self.Omegam = omegam #non-relativistic matter energy density
        self.Omegac = omegac #dark energy density
        self.Omegak = 1 - omegam - omegac #curvature energy density
        
    def get_energy_densities(self, omegam, omegac):
        """Return the current values of Omega_i as a numpy array."""
        
        return np.array([self.Omegam, self.Omegac, self.Omegak])

    def get_eos(self, omegam, omegac):
        """Return the equation of state for this cosmology. This parameter can not be changed once initialized!"""
        
        return self.eos
    
    def E(self, z):
        """Compute the dimensionless Hubble rate H(z)/H0 for a given redshift z."""
        return np.sqrt(self.Omegam * (1+z)**3 + self.Omegak * (1+z)**2 + self.Omegac * (1+z)**(3*(1 + self.eos)))
        

    def luminosity_distance(self, z, eps = 1E-3):
        """Compute the luminosity distance for a given redshift z in this cosmology in [Mpc]. 
        eps is the desired accuracy for the curvature energy density"""
        
        #first integrate to obtain the comoving distance
        dC = self.dH * integrate.quad(lambda x: 1/self.E(x), 0, z)[0]
        
        if self.Omegak > eps: #negative curvature Universe
            sinhk = self.dH*np.sinh(np.sqrt(np.abs(self.Omegak)) * dC/self.dH)/np.sqrt(np.abs(self.Omegak))
        elif self.Omegak < - eps: #positive curvature Universe
            sinhk = self.dH*np.sin(np.sqrt(np.abs(self.Omegak)) * dC/self.dH)/np.sqrt(np.abs(self.Omegak))
        else: #flat Universe
            sinhk = dC

        return (1+z) * sinhk 
        
    def distance_modulus(self, z):
        """Compute the distance modulus for a given redshift, defined via DM = 5 log10(luminosity_distance/Mpc) + 25."""
        
        return 5*(np.log10(np.abs(self.luminosity_distance(z))) + 5)
    
    
    def log_likelihood(self, data, Cov):
        """Compute the Gaussian log-likelihood for given data tuples (z, DM(z)) and covariance."""
        
        if data.shape[0] == 2:
            z = data[0]
            DM_data = data[1]
        elif data.shape[1] == 2:
            z = data[:,0]
            DM_data = data[:,1]
        else:
            raise ValueError('Data has wrong format.')
            
        model = np.array([self.distance_modulus(zi) for zi in z])
        
        return -0.5 * np.dot(np.dot((model - DM_data), np.linalg.inv(Cov)), (model - DM_data)) - .5 * np.sum(np.diag(np.log(np.linalg.eigvalsh(Cov))))
        
        
        
        
class Supernova_data:
    """Objects of this class represent SN data sets. Can be used to calculate the standardized distance modulus for given nuissance parameters (a, b, MB, delta_Mhost). Data must have shape (N,5)."""
    
    def __init__(self, data, err, param):
        if data.shape[1] != 5 and err.shape[1] != 3:
            raise ValueError('Data has wrong format: data = (z, mb, X1, C, Mhost)')
        elif len(param) != 4: 
            raise ValueError('Parameters have wrong format: param = (a, b, MB, delta_Mhost)')
        else:
            self.data = data   
            self.err = err   
            self.param = param
        
    def get_data(self):
        return self.data
        
    def get_err(self):
        return self.err
    
    def get_param(self):
        return self.param
    
    def set_param(self, new_param):
        if new_param.shape[1] != 4:
            raise ValueError('Parameters have wrong format: param = (a, b, MB, delta_Mhost)')
        self.param = new_param
        
    def distance_modulus(self):
        a, b, MB, delta_Mhost = self.param.T
        z, mb, x1, c, Mhost = self.data.T
            
        DM_SN = mb + a * x1 - b * c - (MB + np.heaviside(Mhost-10,1) * delta_Mhost)
        
        return np.array([z, DM_SN]).T
    
    def delta_distance_modulus(self):
        a, b, MB = self.param.T[:3]
        z = self.data.T[0]
        del_mb, del_x1, del_C = self.err.T
        
        sigmaSN =  del_mb**2 + a**2 * del_x1**2 + b**2 * del_C**2
        
        return mu_cov(a,b) + np.diag(sigmaSN) 
    
    
class Quasar_data:
    """Objects of this class represent Quasar data sets. Can be used to calculate the standardized distance modulus for given nuissance parameters (a, b, MB, delta_Mhost). Data must have shape (N,5)."""
    
    gamma = 0.634 #slope of the logLUV-logLX relation common to all Quasars
    
    def __init__(self, data, err, param):
        if data.shape[1] != 6 and len(param) != 4 and len(err) != len(data):
            raise ValueError('Data has wrong format: data = (z, logLUV, logLX, logFUV, logFX)')
        elif len(param) != 2: 
            raise ValueError('Parameters have wrong format: param = (beta_prime, s)')
        else:
            self.data = data   
            self.err = err   
            self.param = param
            
    def get_data(self):
        return self.data
    
    def get_err(self):
        return self.err
    
    def get_param(self):
        return self.param
    
    def get_gamma(self):
        return self.gamma
    
    def set_param(self, new_param):
        if new_param.shape[1] != 2: 
            raise ValueError('Parameters have wrong format: param = (beta_prime, s)')
        self.param = new_param
        
    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

        
    def distance_modulus(self):
        beta_prime = self.param.T[:1]
        z, logLUV, logLX, logFUV, logFX = self.data.T
    
        DM_Q = 5 / 2 / (self.gamma-1) * (logFX - self.gamma * logFUV + beta_prime)
        
        return np.array([z, DM_Q]).T
        
    def delta_distance_modulus(self):
        s = self.param.T[1]
        z = self.data.T[0]
        err_logFX = self.err
       
        sigmaQ = np.sqrt((5/2/(1-self.gamma) * err_logFX)**2 + s**2)
        
        return np.diag(sigmaQ)