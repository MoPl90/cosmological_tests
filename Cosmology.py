import numpy as np
import scipy.integrate as integrate
from covmat.covmat import mu_cov
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import i0, i1, k0, k1

eV = 7.461E33#km/s/Mpc


class cosmology:
    """This class implements the base class for any cosmological model we study. More involved models inherit from this class."""
    
    cLight = 3E5 # speed of light in km/s
    #H0 = 70. #the present day Hubble rate in km/s/Mps
    
    def __init__(self, omegam, omegac, rs = 147.27, omegar=None, omegab = None, w=-1, Hzero=70):
        """Initialise a cosmological model"""
        self.r_sound = rs
        self.Omegac = omegac #dark energy density
        self.Omegam = omegam #non-relativistic matter energy density
        self.Omegar = omegar #relativistic matter energy density
        self.Omegab = omegab
        
        self.Omegak = 1 - self.Omegam - self.Omegac if self.Omegar is None else 1 - self.Omegam - self.Omegac - self.Omegar #curvature energy density
        
        self.eos = w #the dark energy equation of state
        self.H0 = Hzero #the present day Hubble rate in km/s/Mps
    
    def get_energy_densities(self):
        """Return the current values of Omega_i as a numpy array. 
        
        Returns: [Omega_m, Omega_c, Omega_r, Omega_b, Omega_k]"""
        
        return np.array([self.Omegam, self.Omegac, self.Omegar, self.Omegab, self.Omegak])

    def get_eos(self, omegam, omegar, omegac):
        """Return the equation of state for this cosmology. This parameter can not be changed once initialized!"""
        
        return self.eos        
    
    def set_energy_densities(self, omegam=None, omegac=None, omegar=None, omegab=None):
        """Change the initialised values of Omega_i."""
        if not omegam is None:
            self.Omegam = omegam #non-relativistic matter energy density
        if not omegac is None:
            self.Omegac = omegac #dark energy density
        if not omegar is None:
            self.Omegar = omegar #relativistic matter energy density
        if not omegab is None:
            self.Omegab = omegab #baryonic matter energy density
        
        self.Omegak = 1 - self.Omegam - self.Omegac if self.Omegar is None else 1 - self.Omegam - self.Omegac - self.Omegar #curvature energy density
        
        return self
    
    def set_eos(self, w):
        """Change the value of the equation of state of Dark Energy."""
        
        self.eos = w
        return self
    
    def set_sound_horizon(self, rs):
        """Set a new value for the sound horizon."""
        self.r_sound = rs
        
        return self
        
    def sound_speed(self,z):
        """Compute the speed of sound in the baryon-phonton plasma."""
        if self.Omegar is None or self.Omegab is None:
            raise(ValueError('To compute a speed of sound, set Omega_r and Omega_b to numerical values first.'))
        else:
            rel_density = self.Omegar + 7/8 * (4/11)**(4/3) * 3.0 * self.Omegar #Neutrinos are not yet decoupled!
            soundspeed = self.cLight/np.sqrt(3*(1+3/4*self.Omegab/self.Omegar /(1+z)))  
        
        return soundspeed
    
    def com_sound_horizon(self,z_d=1089.):
        """Compute the sound horizon at a given redshift. If Omega_r and Omega_b do not have numerical values, the default r_s is returned."""
        
        if self.Omegar is None or self.Omegab is None:
            rs = self.r_sound
        else:
            rs = integrate.quad(lambda z:self.sound_speed(z)/(self.H(z)),z_d,np.inf)[0]
            self.r_sound = rs
        
        return rs
    
    def rd(self, m_nu = 1E-3): 
        """Accuarte Numerical approximation to the sound horizon, cf. arXiv:1411.1074"""
        Omega_nu = .0101 * m_nu#eV
        h = self.H0/100

        if self.Omegab is None:
            raise(ValueError("Omega_b must have numerical value"))
        elif self.Omegam == 0.:
            return self.r_sound
        else:
            return 55.154 * np.exp(-72.3 * (Omega_nu * h**2 + .0006)**2) / (h**2 * self.Omegam)**.25351 / (h**2 * self.Omegab)**.12807

    
    def H(self, z):
        """Compute the Hubble rate H(z) for a given redshift z."""
        if not self.Omegar is None:
            return self.H0 * np.sqrt(self.Omegar * (1+z)**4 + self.Omegam * (1+z)**3 + self.Omegak * (1+z)**2 + self.Omegac * (1+z)**(3*(1 + self.eos)))
        else:
            return self.H0 * np.sqrt(self.Omegam * (1+z)**3 + self.Omegak * (1+z)**2 + self.Omegac * (1+z)**(3*(1 + self.eos)))
        

    def luminosity_distance(self, z, eps = 1E-3):
        """Compute the luminosity distance for a given redshift z in this cosmology in [Mpc]. 
        eps is the desired accuracy for the curvature energy density"""
        
        dH = self.cLight / self.H0 # Hubble length in Mpc
        
        #first integrate to obtain the comoving distance
        if isinstance(z, (float, int)):
            dC = self.cLight * integrate.quad(lambda x: 1/self.H(x), eps, z)[0]
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
            Cov = dataObject.data_cov()
        elif dataObject.name == 'BAO':
            data = dataObject.distance_modulus(self) + dataObject.Hubble(self)
            Cov = dataObject.data_cov(self)
        elif dataObject.name == 'CMB':
            return dataObject.log_likelihood(self)
        
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
           
 
        # define the model DM:
        if dataObject.name == 'Quasars' or dataObject.name == 'SN':
            model = self.distance_modulus(z)
        elif dataObject.name == 'BAO':
            model = np.array([self.distance_modulus(zi) if zi != 0. else 0. for zi in dataObject.distance_modulus(self).T[0]])
            model += np.array([self.H(zi) if zi != 0. else 0. for zi in dataObject.Hubble(self).T[0]])
        

        
  
        

        if len(Cov.shape) == 2:
            Cov_inv = np.linalg.inv(Cov)
            Cov_eigvals = np.linalg.eigvalsh(Cov)
        elif len(Cov.shape) == 1:
            Cov_inv = np.diag(1/Cov)
            Cov_eigvals = Cov
        else:
            raise ValueError('Cov must be 1d or 2d array')
        
        return -0.5 * ((model - DM_data) @ Cov_inv @ (model - DM_data)) - .5 * np.sum(np.log(Cov_eigvals))


class bigravity_cosmology(cosmology):
    """This class inherits from the cosmology base class and implements a bigravity cosmology."""

    def __init__(self, log10m, theta, b0, b1, b2, b3, omegam, rs=147.27, omegar=None, omegab=None, Hzero=70.):
        super().__init__(omegam, omegac=0, rs=rs, omegar=omegar, omegab=omegab, w=-1, Hzero=Hzero)
        self.log10mg = log10m
        self.t = theta
        self.betas = np.array([b0, b1, b2, b3])

    def set_bigra_params(self, log10m, theta, b0, b1, b2, b3):
        """
        Change the bigravity model parameters of an existing object.

        Returns:
        The modified bigravity_cosmology object
        """

        self.log10mg = log10m
        self.t = theta
        self.betas = np.array([b0, b1, b2, b3])
        return self
    
    def set_cosmo_params(self, omegam, omegar=0, Hzero=70.):
        """
        Change the cosmological parameters of an existing object.

        Returns:

        The modified bigravity_cosmology object
        """

        super().__init__(omegam, 0, omegar, w=-1, Hzero=Hzero)

        return self

    def Bianchi(self, z): 
        """
        This is Kevin's exact solution to the Bianchi-/Master-Equation of bigravity cosmology.

        Input:    
        redshift z

        Returns:
        y = b(z) / a(z) 
        """ 
        
        x = self.log10mg + 32 # log10(mg/10**-32)
        b0, b1, b2, b3 = self.betas
        
        
        a0 = - b1 / ( np.tan(self.t) * b3) + 0j
        a1 = lambda z: (-3*b2 + b0*np.tan(self.t) + (0.140096333403*0.7**2*(1 + z)**3*self.Omegam*(1 + np.tan(self.t)))/10**(2*x))/b3/np.tan(self.t)+0j
        a2 = (3*b1)/b3 - 3*1/np.tan(self.t)+0j
        try:
            x1 = a2/3. - (2**(1/3.)*(-12*a0 - a2**2))/(3.*(27*a1(z)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3.)) +  (27*a1(z)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3.)/(3.*2**(1/3.))
            if np.any(np.imag(x1) > 10**-6 * np.real(x1)):
                raise RuntimeError
        except RuntimeError:
            return -10 * np.ones_like(z)
        
        if np.all(x1 >= a2) and np.all(-a2 - x1 + (2*a1(z))/np.sqrt(-a2 + x1) >= 0.) and np.all(4*(-12*a0 - a2**2)**3 + (27*a1(z)**2 - 72*a0*a2 + 2*a2**3)**2 >= 0.) and np.all(27*a1(z)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z)**2 - 72*a0*a2 + 2*a2**3)**2) >= 0.) and np.all(1/6.*(-8*a2 - (2*2**(1/3.)*(12*a0 + a2**2))/(27*a1(0)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(-4*(12*a0 + a2**2)**3 + (27*a1(0)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3.) - 2**(2/3.)*(27*a1(0)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(-4*(12*a0 + a2**2)**3 + (27*a1(0)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3.) + (12*np.sqrt(6)*a1(0))/np.sqrt(-4*a2 + (2*2**(1/3.)*(12*a0 + a2**2))/(27*a1(0)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(-4*(12*a0 + a2**2)**3 + (27*a1(0)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3.) + 2**(2/3.)*(27*a1(0)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(-4*(12*a0 + a2**2)**3 + (27*a1(0)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3.)))):
            return np.real(-np.sqrt(-a2 + x1)/2. + np.sqrt(-a2 - x1 + (2*a1(z))/np.sqrt(-a2 + x1))/2.)
        else: 
            return -1 * np.ones_like(z)
        
        
    def H(self, z):
        """
        This method overrides the cosmology.H(z) function, which computes the Hubble rate as a function of redshift. This method uses the bigravity cosmology instead.
        
        Input:
        resdshift z
        
        Output:
        Hubble rate H(z)
        """
        
        b0, b1, b2, b3 = self.betas 
        y = self.Bianchi(z)
        m = 10**self.log10mg
        
        if isinstance(y, float) or isinstance(y, int):
            if y<0:
                return 0.
            else:
                CC_dyn = m**2 * np. sin(self.t)**2 * (b0*np.ones_like(y) + 3*b1*y + 3*b2*y**2 + b3*y**3)/3.
                hubble_squared = self.H0**2 * (self.Omegam * (1+z)**3 )#+ Omegak * (1+z)**2 + Omegac) 
                
                return np.sqrt(hubble_squared + CC_dyn*eV**2)
            
        if np.any(y<0):
            return np.zeros_like(z)
        else:
            CC_dyn = m**2 * np. sin(self.t)**2 * (b0*np.ones_like(y) + 3*b1*y + 3*b2*y**2 + b3*y**3)/3.
            hubble_squared = self.H0**2 * (self.Omegam * (1+z)**3 )#+ Omegak * (1+z)**2 + Omegac) 

            return np.sqrt(hubble_squared + CC_dyn.T*eV**2)
        

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
        
        return self
        
    def distance_modulus(self):
        a, b, MB, delta_Mhost = self.param
        z, mb, x1, c, Mhost = self.data.T
            
        DM_SN = mb + a * x1 - b * c - (MB + np.heaviside(Mhost-10,1) * delta_Mhost)
        
        return np.array([z, DM_SN]).T
    
    def data_cov(self):
        a, b, MB = self.param[:3]
        z = self.data.T[0]
        del_mb, del_x1, del_C = self.err.T
        
        covSN =  del_mb**2 + a**2 * del_x1**2 + b**2 * del_C**2
        
        return mu_cov(a,b) + np.diag(covSN) 
    
    def delta_distance_modulus(self):
        # this function returns the error, that is the sqrt of the diagonal entries of cov


        return np.sqrt(np.diagonal(self.data_cov()))
    
    
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
        
        return self
    
    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

        return self
        
    def distance_modulus(self):
        beta_prime = self.param[0]
        z, logLUV, logLX, logFUV, logFX = self.data.T
    
        DM_Q = 5 / 2 / (self.gamma-1) * (logFX - self.gamma * logFUV + beta_prime)
        
        return np.array([z, DM_Q]).T
        
    def data_cov(self):
        s = self.param[1]
        z = self.data.T[0]
        err_logFX = self.err
       
        #sigmaQ = np.sqrt((5/2/(1-self.gamma) * err_logFX)**2 + s**2)
        covQ = (5/2/(1-self.gamma) * err_logFX)**2 + s**2
        
        return covQ
    
    def delta_distance_modulus(self):
        # this function returns the error, that is the sqrt of the diagonal entries of cov


        return np.sqrt(self.data_cov())


class BAO_data:
    """Objects of this class represent BAO data sets."""
    
    cLight = 3E5 # speed of light in km/s
    z_d = 1089   # redshift of decoupling
    
    
    def __init__(self, data, err, dataType):
        if data.shape[1] != 2 and err.shape[1] != 1:
            raise ValueError('Data has wrong format: data = (z, DM/rd)')
        else:
            self.data = data   
            self.err = err   
            self.dataType = dataType
            self.name = "BAO"
            

            
    def get_data(self):
        return self.data
    
    def get_err(self):
        return self.err
    
    def distance_modulus(self,cosmo):
        # where our heroes convert all BAO data (which come from a variety of formats) into
        # the standardised dist mod
        
        z, meas = self.data.T
        dtype = self.dataType
        
        rd_fid = 147.78 # fiducial sound horizon in MPc
        z_d = self.z_d
        
        rd = cosmo.com_sound_horizon() # sound horizon for given cosmology
        
        DMpairs = np.zeros([len(dtype), 2])
        
        for line in range(0,len(dtype)): 
            if dtype[line]=='DM*rd_fid/rd':
                # calculate lumi dist and DM
                lumiDist = (1 + z[line]) * meas[line]*rd/rd_fid
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='DM/rd':
                # calculate lumi dist and DM
                lumiDist = (1 + z[line]) * meas[line]*rd
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
                
            elif dtype[line]=='DA*rd_fid/rd':
                lumiDist = (1 + z[line])**2 * meas[line]*rd/rd_fid
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )

            elif dtype[line]=='DA/rd':
                lumiDist = (1 + z[line])**2 * meas[line]*rd
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='DV*rd_fid/rd':
                lumiDist =  (1 + z[line]) * ( (meas[line]*rd/rd_fid)**3 * cosmo.H(z[line])/self.cLight/z[line] )**(1/2)
                DMpairs[line] =  ( z[line], 5*(np.log10(lumiDist) + 5) )    
            
            elif dtype[line]=='DV/rd':
                lumiDist =  (1 + z[line]) * ( (rd*meas[line])**3 * cosmo.H(z[line])/self.cLight/z[line] )**(1/2)
                DMpairs[line] =  ( z[line], 5*(np.log10(lumiDist) + 5) )
                
            elif dtype[line]=='rd/DV':
                lumiDist =  (1 + z[line]) * ( (rd/meas[line])**3 * cosmo.H(z[line])/self.cLight/z[line] )**(1/2)
                DMpairs[line] =  ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='A':
                lumiDist =  (1 + z[line]) * (meas[line]/100)**(3/2) * (cosmo.H(z[line]))**(1/2) *self.cLight*z[line]/(cosmo.Omegam*(cosmo.H0/100)**2)**(3/4)
                DMpairs[line] =  ( z[line], 5*(np.log10(lumiDist) + 5) )
                    
            elif dtype[line] == 'H*rdfid/rd' or dtype[line] == 'DH/rd':
                continue
                
            else:
                raise ValueError('input data doesn\'t have a recognised format')

        return DMpairs
        
        
    def Hubble(self, cosmo):
        """This function extracts the measured Hubble rate at a given redshift."""
        
        z, meas = self.data.T
        dtype = self.dataType
        Hpairs = np.zeros([len(dtype), 2])
        z_d = self.z_d
        rd = cosmo.com_sound_horizon()
        
        for line in range(0,len(dtype)): 
            if dtype[line] == 'H*rdfid/rd':
                H = meas[line]
                Hpairs[line] = (z[line], H)
                    
            elif dtype[line] == 'DH/rd':
                H = self.cLight / meas[line] / rd
                Hpairs[line] = (z[line], H)
            else:
                continue
                
        return Hpairs
        
    def data_cov(self,cosmo):
        # this function returns the covariance matrix.

        z, meas = self.data.T
        dtype = self.dataType
        
        rd_fid = 147.78 # fiducial sound horizon in Mpc
        z_d = self.z_d
        
        rd = cosmo.com_sound_horizon() # sound horizon for given cosmology
        
        covDM = np.zeros([len(self.err), len(self.err)])
                
        for i in range(0,len(self.err)):
            for j in range(0,len(self.err)):
                if dtype[i]==dtype[j]=='DM*rd_fid/rd' or dtype[i]==dtype[j]=='DM/rd' or dtype[i]==dtype[j]=='DA*rd_fid/rd' or dtype[i]==dtype[j]=='DA/rd':
                    # calculate cov:
                    covDM[i,j] =  (5/meas[i]) * self.err[i,j] *  (5/meas[j])
                
                elif dtype[i]==dtype[j]=='rd/DV' or dtype[i]==dtype[j]=='DV*rd_fid/rd' or dtype[i]==dtype[j]=='DV/rd' or dtype[i]==dtype[j]=='A':
                    covDM[i,j] =  (5*3/2/meas[i]) * self.err[i,j] *  (5*3/2/meas[j])
                
                elif dtype[i]==dtype[j]=='H*rdfid/rd':
                    covDM[i,j] = self.err[i,j] 
                elif dtype[i]==dtype[j]=='DH/rd':
                    covDM[i,j] = self.err[i,j] / meas[i] / meas[j] * self.cLight / rd
            
        #return sigmaDM.T[0]
        return covDM   # BAO errors are now also arrays due to the correlations in WiggleZ data
    
    def delta_distance_modulus(self,cosmo):
        # this function returns the error, that is the sqrt of the diagonal entries of cov


        return np.sqrt(np.diagonal(self.data_cov(cosmo)))
    


class CMB_data:
    """Objects of this class represent simplified CMB measurements."""
    # Ref.: 1411.1074
    C_Planck18 = np.array([[2.8714501E-08, -1.8525566E-07, 2.5062628E-08],
                           [-1.8525566E-07, 2.5811906E-06, -2.3468816E-07],
                           [2.5062628E-08, -2.3468816E-07, 1.0694029E-07]])
    mu_Planck18 = np.array([2.2531710E-02, 1.1862903E-01, 1.0410763E+00])# Omega_b, Omega_m - Omega_b, 100 rd / DM!!!!!!!!!!!!!!!!!!!!

    
    C_Planck13 = 1E-7 * np.array([[1.286,-6.033, -144.3],
                                  [-6.033, 75.42, -360.5],
                                  [-144.3, -360.5, 42640]])
    mu_Planck13 = np.array([.02245, .1386, 94.33]) # Omega_b, Omega_m, DM/rd

    C_WMAP = 1E-7 * np.array([[2.864, -4.809, -111.1],
                              [-4.809, 190.8, -74.95],
                              [-111.1, -74.95, 254200]])
    mu_WMAP = np.array([.02259, .1354, 94.51]) # Omega_b, Omega_m, DM/rd
    
    def __init__(self, satellite):
        
        self.name = 'CMB'
        self.satellite = satellite
        if satellite == 'Planck18':
            self.data  = self.mu_Planck18  # Omega_b, Omega_DM, 100 rd/DM
            self.cov = self.C_Planck18
        if satellite == 'Planck13':
            self.data  = self.mu_Planck13  # Omega_b, Omega_m, DM/rd
            self.cov = self.C_Planck13
        elif satellite == 'WMAP' or  satellite == 'Wmap':
            self.data  = self.mu_WMAP  # Omega_b, Omega_m, DM/rd
            self.cov = self.C_WMAP 
            
    
    def get_data(self):
        return self.data
    
    def log_likelihood(self, cosmo):
        """Compute the log likelihood for the CMB data given a cosmology. THIS MUST GO IN COSMOLOGY CLASS EVENTUALLY!!!"""
        
        if cosmo.Omegab is None:
            raise(ValueError("Must set Omega_b before computing sound horizon!"))
        
        h = cosmo.H0 / 100.
        Omegab = cosmo.Omegab
        Omegam = cosmo.Omegam
        rs = cosmo.com_sound_horizon()
        if self.satellite == 'Planck18':
            mu = self.data - np.array([Omegab*h**2, Omegam*h**2 - Omegab*h**2, 100 * rs / (cosmo.luminosity_distance(1089) / (1089 + 1))])
        else:
            mu = self.data - np.array([Omegab*h**2, Omegam*h**2, cosmo.luminosity_distance(1089) / (1089 + 1) / rs])
        
        Cov_inv = np.linalg.inv(self.cov)
        
        return -0.5 * mu @ Cov_inv @ mu

    
    

import warnings
warnings.simplefilter("ignore")


class RC_data:
    """Objects of this class represent the SPARC rotation curve data sample (arXiv:1606.09251), relevant for testing conformal gravity."""

    GN =  4.301E3 #GN * 10^9 Msol / 1kpc in (km/s)^2
    cLight = 3E5 # speed of light in km/s

    def __init__(self, params):
        self.name = "RC"
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
        
        return self

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


class likelihood:
    """This class implements a generic likelihood function to pass to a emcee sampler.
    
    Input: parameter vector theta, data sets, cosmological model name."""
    
    
    h = .6727
    omega_baryon_preset = 0.02236/h**2
    omega_gamma_preset = 2.469E-5/h**2


    
    def __init__(self, theta, data_sets, ranges_min, ranges_max, model ='LCDM'):
        
        self.params = theta
        self.data_sets = {}
        self.model = model
        
        self.ranges_min, self.ranges_max = np.array(ranges_min), np.array(ranges_max) # prior ranges
        if len(ranges_min) != len(self.params) or len(ranges_max) != len(self.params):
            raise(ValueError("You must specify a minimum and a maximum value for each parameter."))
        if np.any(self.ranges_min > self.ranges_max):
            raise(ValueError("You must specify ranges with min <= max."))
            
        
        
        for sample in data_sets:
            if sample.name == 'SN':
                self.data_sets['SN'] = sample
            elif sample.name == 'Quasars':
                self.data_sets['Quasars'] = sample
            elif sample.name == 'BAO':
                self.data_sets['BAO'] = sample
            elif sample.name == 'CMB':
                self.data_sets['CMB'] = sample
            elif sample.name == 'RC' and self.model == 'conformal':
                self.data_sets['RC'] = sample
                

        if self.model == 'LCDM':
            Omegam, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = cosmology(omegam=Omegam, omegac=1 - Omegam, omegab = Omegab, omegar = self.omega_gamma_preset, Hzero=H0)
        elif self.model == 'oLCDM':
            Omegam, Omegac, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = cosmology(omegam=Omegam, omegac=Omegac, omegab = Omegab, omegar = self.omega_gamma_preset, Hzero=H0)
        elif self.model == 'wLCDM':
            Omegam, Omegac, Omegab, H0, w, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = cosmology(omegam=Omegam, omegac=Omegac, omegab = Omegab, omegar = self.omega_gamma_preset, w = w, Hzero=H0)
        elif self.model == 'conformal':
            gamma0, kappa, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            Omegak =  (gamma0)**2 / 2 *cosmology.cLight**2/(70./1E-3)**2#Gpc^-1
            Omegac = 1-Omegak
            self.cosmo = cosmology(omegam=0., omegac=Omegac, omegab = Omegab, omegar = self.omega_gamma_preset, Hzero=H0)
        elif self.model == 'bigravity':
            log10m, t,  Omegam, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = bigravity_cosmology(log10m, t, 1, 1, -1, 1, omegam=Omegam, omegab = Omegab, omegar = self.omega_gamma_preset, Hzero=H0)
        else: 
            raise(TypeError('Please specify which cosmology to use from [LCDM, wLCDM, bigravity]'))

            

        if 'SN' in self.data_sets.keys():
            self.data_sets['SN'].set_param([a, b, MB, delta_Mhost])
        if 'Quasars' in self.data_sets.keys():
            self.data_sets['Quasars'].set_param([beta_prime, s])
        if 'RC' in self.data_sets.keys() and self.model == 'conformal':
            self.data_sets['RC'].set_param([gamma0, kappa])
            self.data_sets['RC'].fit()
        


    def lnlike(self):
        """Compute the likelihood for the given data and 
        """
        
        log_prob = 0.
        for sample_name in self.data_sets.keys():
            
            if sample_name == 'RC':
                log_prob += self.data_sets['RC'].get_loglike()
            else:
                log_prob += self.cosmo.log_likelihood(self.data_sets[sample_name])

            if np.isnan(log_prob):
                return -np.inf
            
        return log_prob


    def lnprior_flat(self):
        """Compute a flat prior for the posterior distribution."""
        
        if self.model == 'conformal':
            gamma0 = self.params[0]
            Omegak =  (gamma0)**2 / 2 *cosmology.cLight**2/(self.cosmo.H0/1E-3)**2#Gpc^-1
            if Omegak > 1: #Omegac < 0
                return -np.inf
        
        for i in range(len(self.params)):
            if self.params[i] < self.ranges_min[i]:
                return -np.inf
            if self.params[i] > self.ranges_max[i]:
                return -np.inf
            
        return 0    
    
    def lnprior_gauss(self):
        """Compute a prior which is gaussian in Omegab h^2"""
        if self.model == 'LCDM':
            Omegab, H0 = self.params[1:3]
        elif self.model == 'bigravity':
            Omegab, H0 = self.params[3:5]
        else:
            Omegab, H0 = self.params[2:4]
        h = H0 / 100

        gauss = - (Omegab*h**2 - 0.02235)**2 / (2 * 7.4E-4**2)

        return self.lnprior_flat() + gauss    

        
    def logprobability_flat_prior(self):
        lp = self.lnprior_flat()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike()


    def logprobability_gauss_prior(self):
        lp = self.lnprior_gauss()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike()
