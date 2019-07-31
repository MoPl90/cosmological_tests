import numpy as np
import scipy.integrate as integrate
from covmat.covmat import mu_cov
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import i0, i1, k0, k1


class cosmology:
    """This class implements the base class for any cosmological model we study. More involved models inherit from this class."""
    
    cLight = 3E5 # speed of light in km/s
    #H0 = 70. #the present day Hubble rate in km/s/Mps
    
    def __init__(self, omegam, omegac, omegar=0, w=-1, Hzero=70):
        """Initialise a cosmological model"""
        self.Omegam = omegam #non-relativistic matter energy density
        self.Omegar = omegar #relativistic matter energy density
        self.Omegac = omegac #dark energy density
        self.Omegak = 1 - omegam - omegac #curvature energy density
        self.eos = w #the dark energy equation of state
        self.H0 = Hzero #the present day Hubble rate in km/s/Mps
        
    def set_energy_densities(self, omegam, omegac):
        """Change the initialised values of Omega_i."""
        self.Omegam = omegam #non-relativistic matter energy density
        self.Omegar = omegar #relativistic matter energy density
        self.Omegac = omegac #dark energy density
        self.Omegak = 1 - omegam - omegac #curvature energy density
        
    def get_energy_densities(self, omegam, omegar, omegac):
        """Return the current values of Omega_i as a numpy array."""
        
        return np.array([self.Omegam, self.Omegar, self.Omegac, self.Omegak])

    def get_eos(self, omegam, omegar, omegac):
        """Return the equation of state for this cosmology. This parameter can not be changed once initialized!"""
        
        return self.eos
    
    def H(self, z):
        """Compute the Hubble rate H(z) for a given redshift z."""
        return self.H0 * np.sqrt(self.Omegar * (1+z)**4 + self.Omegam * (1+z)**3 + self.Omegak * (1+z)**2 + self.Omegac * (1+z)**(3*(1 + self.eos)))
        

    def luminosity_distance(self, z, eps = 1E-3):
        """Compute the luminosity distance for a given redshift z in this cosmology in [Mpc]. 
        eps is the desired accuracy for the curvature energy density"""
        
        dH = self.cLight / self.H0 # Hubble length in Mpc
        
        #first integrate to obtain the comoving distance
        if isinstance(z, float):
            dC = self.cLight * integrate.quad(lambda x: 1/self.H(x), 0, z)[0]
        elif isinstance(z, (list, np.ndarray)):
            z_int = np.append([0], z)
            dC = self.cLight * integrate.cumtrapz(1/self.H(z_int), z_int)
        else: 
            raise ValueError('z must be a float or array!')
            
        
        if self.Omegak > eps: #negative curvature Universe
            sinhk = dH*np.sinh(np.sqrt(np.abs(self.Omegak)) * dC/dH)/np.sqrt(np.abs(self.Omegak))
        elif self.Omegak < - eps: #positive curvature Universe
            sinhk = dH*np.sin(np.sqrt(np.abs(self.Omegak)) * dC/dH)/np.sqrt(np.abs(self.Omegak))
        else: #flat Universe
            sinhk = dC

        return (1+z) * sinhk 
        
    def distance_modulus(self, z):
        """Compute the distance modulus for a given redshift, defined via DM = 5 log10(luminosity_distance/Mpc) + 25."""
        
        return 5*(np.log10(np.abs(self.luminosity_distance(z))) + 5)
    
    
    def log_likelihood(self, dataObject):
        """Compute the Gaussian log-likelihood for given data tuples (z, DM(z)) and covariance."""
        
        
        
  
        if dataObject.name == 'Quasars' or dataObject.name == 'SN':
            data = dataObject.distance_modulus()
            Cov = dataObject.delta_distance_modulus()
        elif dataObject.name == 'BAO':
            data = dataObject.distance_modulus(self)
            Cov = dataObject.delta_distance_modulus(self)
        else:
            raise ValueError('Data type needs to be associated with input (e.g. BAO, quasars, ...)')
            
        
  
        # bring data into correct shape in case it isnt:
        
        if data.shape[0] == 2:
            z = data[0]
            DM_data = data[1]
        elif data.shape[1] == 2:
            z = data[:,0]
            DM_data = data[:,1]
        else:
            raise ValueError('Data has wrong format.')
            

 
        #if dataObject.name == 'Quasars' or dataObject.name == 'SN':
            #data = dataObject.distance_modulus()
            #Cov = dataObject.delta_distance_modulus()
        #elif dataObject.name == 'BAO':
            #data = dataObject.distance_modulus(self)
            #Cov = 1
       # else:
            #raise ValueError('Data type needs to be associated with input (e.g. BAO, quasuars, ...)')
 
 
        # define the model DM:
        
        model = self.distance_modulus(z)
        

        
  
        

        if len(Cov.shape) == 2:
            Cov_inv = np.linalg.inv(Cov)
            Cov_eigvals = np.linalg.eigvalsh(Cov)
        elif len(Cov.shape) == 1:
            Cov_inv = np.diag(1/Cov)
            Cov_eigvals = Cov
        else:
            raise ValueError('Cov must be 1d or 2d array')
        
        return -0.5 * ((model - DM_data) @ Cov_inv @ (model - DM_data)) - .5 * np.sum(np.log(Cov_eigvals))
        
        
        
        
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
            self.name = "SN"
        
    def get_data(self):
        return self.data
        
    def get_err(self):
        return self.err
    
    def get_param(self):
        return self.param
    
    def set_param(self, new_param):
        if len(new_param) != 4:
            raise ValueError('Parameters have wrong format: param = (a, b, MB, delta_Mhost)')
        self.param = new_param
        
    def distance_modulus(self):
        a, b, MB, delta_Mhost = self.param
        z, mb, x1, c, Mhost = self.data.T
            
        DM_SN = mb + a * x1 - b * c - (MB + np.heaviside(Mhost-10,1) * delta_Mhost)
        
        return np.array([z, DM_SN]).T
    
    def delta_distance_modulus(self):
        a, b, MB = self.param[:3]
        z = self.data.T[0]
        del_mb, del_x1, del_C = self.err.T
        
        covSN =  del_mb**2 + a**2 * del_x1**2 + b**2 * del_C**2
        
        return mu_cov(a,b) + np.diag(covSN) 
    
    
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
            self.name = "Quasars"
            
    def get_data(self):
        return self.data
    
    def get_err(self):
        return self.err
    
    def get_param(self):
        return self.param
    
    def get_gamma(self):
        return self.gamma
    
    def set_param(self, new_param):
        if len(new_param) != 2: 
            raise ValueError('Parameters have wrong format: param = (beta_prime, s)')
        self.param = new_param
        
    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

        
    def distance_modulus(self):
        beta_prime = self.param[0]
        z, logLUV, logLX, logFUV, logFX = self.data.T
    
        DM_Q = 5 / 2 / (self.gamma-1) * (logFX - self.gamma * logFUV + beta_prime)
        
        return np.array([z, DM_Q]).T
        
    def delta_distance_modulus(self):
        s = self.param[1]
        z = self.data.T[0]
        err_logFX = self.err
       
        #sigmaQ = np.sqrt((5/2/(1-self.gamma) * err_logFX)**2 + s**2)
        covQ = (5/2/(1-self.gamma) * err_logFX)**2 + s**2
        
        return covQ


class BAO_data:
    """Objects of this class represent BAO data sets."""
    
    cLight = 3E5 # speed of light in km/s
    z_d = 1089   # redshift of decoupling
    
    
    def __init__(self, data, err, param, dataType):
        if data.shape[1] != 2 and err.shape[1] != 1:
            raise ValueError('Data has wrong format: data = (z, DM/rd)')
        else:
            self.data = data   
            self.err = err   
            self.param = param
            self.dataType = dataType
            self.name = "BAO"
            
            # take average of errors if not symmetrical
            if self.err.shape[1] == 2:
                self.err = (self.err[:, 0]+self.err[:, 1])/2.
            
    def get_data(self):
        return self.data
    
    def get_err(self):
        return self.err
    
    def get_param(self):
        return self.param
    
    def set_param(self, new_param):
        if len(new_param) != 2: 
            raise ValueError('Parameters have wrong format: param = (beta_prime, s)')
        self.param = new_param
        
    def sound_speed(self,z):
        omega_baryon, omega_gamma = self.param
    
        soundspeed = self.cLight/np.sqrt(3*(1+3/4*omega_baryon/omega_gamma /(1+z)))  
        
        return soundspeed
    
    def com_sound_horizon(self,z_d,cosmo):
            
        comsoundhorizon = integrate.quad(lambda z:self.sound_speed(z)/(cosmo.H(z)),z_d,np.inf)[0]
        
        return comsoundhorizon
 
         
    def distance_modulus(self,cosmo):
        # where our heroes convert all BAO data (which come from a variety of formats) into
        # the standardised dist mod
        
        z, meas = self.data.T
        dtype = self.dataType
        
        rd_fid = 147.78 # fiducial sound horizon in MPc
        z_d = self.z_d
        
        rd = self.com_sound_horizon(z_d,cosmo) # sound horizon for given cosmology
        
        DMpairs = np.empty([len(dtype), 2])
        
        for line in range(0,len(dtype)): 
            if dtype[line]=='DM*rd_fid/rd':
                # calculate lumi dist and DM
                lumiDist = (1 + z[line]) * meas[line]*rd/rd_fid
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='DA*rd_fid/rd':
                lumiDist = (1 + z[line])**2 * meas[line]*rd/rd_fid
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='DV*rd_fid/rd':
                lumiDist =  (1 + z[line]) * ( (meas[line]*rd/rd_fid)**3 * cosmo.H(z[line])/self.cLight/z[line] )**(1/2)
                DMpairs[line] =  ( z[line], 5*(np.log10(lumiDist) + 5) )    
                
            elif dtype[line]=='rd/DV':
                lumiDist =  (1 + z[line]) * ( (rd/meas[line])**3 * cosmo.H(z[line])/self.cLight/z[line] )**(1/2)
                DMpairs[line] =  ( z[line], 5*(np.log10(lumiDist) + 5) )             
                
            else:
                raise ValueError('input data doesn\'t have a recognised format')

        return DMpairs
        
    def delta_distance_modulus(self,cosmo):

        z, meas = self.data.T
        dtype = self.dataType
        
        rd_fid = 147.78 # fiducial sound horizon in MPc
        z_d = self.z_d
        
        rd = self.com_sound_horizon(z_d,cosmo) # sound horizon for given cosmology
        
        covDM = np.empty([len(dtype), 1])
                
        for line in range(0,len(dtype)): 
            if dtype[line]=='DM*rd_fid/rd' or dtype[line]=='DA*rd_fid/rd':
                # calculate com dist and DM:
                covDM[line] =  (5*self.err[line]/meas[line])**2
                
            elif dtype[line]=='rd/DV' or dtype[line]=='DV*rd_fid/rd':
                covDM[line] =  (5*3/2*self.err[line]/meas[line])**2
                
            else:
                raise ValueError('input data doesn\'t have a recognised format')
        
        #return sigmaDM.T[0]
        return np.diag(covDM.T[0])   # BAO errors are now also arrays due to the correlations in WiggleZ data




import warnings
warnings.simplefilter("ignore")


class RCdata:
	"""Objects of this class represent the SPARC rotation curve data sample (arXiv:1606.09251), relevant for testing conformal gravity."""

	GN =  4.301E3 #GN * 10^9 Msol / 1kpc in (km/s)^2
	cLight = 3E5 # speed of light in km/s

	def __init__(self, params):
		self.data = []
		self.names = []
		for fname in glob.glob('data/Rotmod_LTG/*.dat'):
			file = open(fname, 'r')
			self.names.append(fname.replace('data/Rotmod_LTG/', '').replace('_rotmod.dat', ''))
			galaxy = [] 
			for line in file:
				if line[0] !='#':
				    galaxy.append([eval(el) for el in line.lstrip('*').split()])
			self.data.append(np.asarray(galaxy))    
		self.data = np.asarray(self.data) #The SPARC data sample
		self.gamma0, self.kappa = params #Conformal Gravity parameters in kpc^-1 and kpc^-2 respectively, see e.g. arXiv:1211:0188
		self.loglike = 0. 

	def get_names(self, which = -1):
		if which == -1:
			return self.names
		else:
			return np.asarray(self.names)[which]

	def get_data(self, which = 'all'):
		if isinstance(which, str) and which == 'all':
			return self.data
		elif isinstance(which, str):
			return self.data[self.names.index(which)]
		else:
			index = [self.names.index(w) for w in which] 
			return self.data[index] 

	def get_loglike(self):
		return self.loglike

	def reset_loglike(self):
		self.loglike = 0.

	def set_param(self, new_params):
		self.gamma0, self.kappa = new_params 

	def vCG_square(self, r): # units of 1E9 Msol
		"""Compute the *non-local* Conformal Gravity contribution to the rotational velocity."""
		r = r / 1E6# Gpc ##or * 3.086E21 #cm
		return self.cLight**2 * self.gamma0 * r / 2 - self.kappa * self.cLight**2 * r**2

	def vlocal_square (self, r, r0, M0, gamma):
		"""Compute the *local* standard plus Conformal Gravity contribution to the rotational velocity."""
		return self.GN * M0 * (r/r0)**2 * (i0(r/2/r0) * k0(r/2/r0) - i1(r/2/r0) * k1(r/2/r0)) + gamma * (r/r0)**2 * i1(r/2/r0) * k1(r/2/r0)


	def fit(self):
		"""Perform a fit of the CG model with given log10(gamma0) and log10(kappa) and compute the log likelihood."""
		self.reset_loglike()

		for galaxy in self.data:
			pGas, _ = curve_fit(self.vlocal_square,  galaxy[:,0], galaxy[:,3]**2, p0=[1,10,1], bounds=(0, 500))
			pDisk, _ = curve_fit(self.vlocal_square,  galaxy[:,0], galaxy[:,4]**2, p0=[1,10,1], bounds=(0, 500))
			visibleCG = lambda r, YD, YB: self.vlocal_square(r, *pGas) + YD * self.vlocal_square(r, *pDisk) + YD * interp1d(galaxy[:,0], galaxy[:,5]**2, kind='cubic')(r)


			def model_vSquared(r, YD, YB):
				return visibleCG(r, YD, YB) + self.vCG_square(r)
			(YD, YB), _ = curve_fit(model_vSquared, galaxy[:,0], galaxy[:,1]**2, sigma=galaxy[:,2]**2, p0=[1,1], bounds = (0,5))

				
			res= -.5 * np.sum( (galaxy[:,1]**2 - model_vSquared(galaxy[:,0], YD, YB))**2 / galaxy[:,2]**2)
			if ~np.isnan(res):
				self.loglike += res

