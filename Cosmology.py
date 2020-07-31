import numpy as np
import scipy.integrate as integrate
from covmat.covmat import mu_cov
import glob
from scipy.optimize import curve_fit, fsolve
from scipy.interpolate import interp1d
from scipy.special import i0, i1, k0, k1

#eV = 7.461E33#km/s/Mpc
eV = 4.6282E34#km/s/Mpc -- convertes eV to km/s/Mpc


class cosmology:
    """This class implements the base class for any cosmological model we study. More involved models inherit from this class."""
    
    cLight = 3E5 # speed of light in km/s
    #H0 = 70. #the present day Hubble rate in km/s/Mps
    
    def __init__(self, omegam, omegac, rs = 147.78, omegag=None, omegab = None, w=-1, Hzero=70, rd_num = False, z_num = True, T_CMB = 2.7255):
        """Initialise a cosmological model"""
        self.rd_num = rd_num # whether the acoustic oscillation scale should be calculated numerically or analytically
        self.z_num = z_num # whether to use redshift of recombination z_d = 1089 or numerical approximation
        self.r_sound = rs
        self.Omegac = omegac #dark energy density
        self.Omegam = omegam #non-relativistic matter energy density
        self.Omegag = omegag #photon energy density 
        self.Omegab = omegab
        self.T_CMB = T_CMB
        self.H0 = Hzero #the present day Hubble rate in km/s/Mps
        if not self.Omegag is None: # if photon density is defined, calculate radiation by adding neutrinos
            self.Omegar =  self.Omegag*(1 + 7/8 * (4/11)**(4/3) * 3.046) #including neutrinos
        else: #calculate rad density from CMB temperature
            self.Omegar = self.omegar()
        
        self.Omegak = 1 - self.Omegam - self.Omegac if self.Omegar is None else 1 - self.Omegam - self.Omegac - self.Omegar #curvature energy density
        
        self.eos = w #the dark energy equation of state
    
    def get_energy_densities(self):
        """Return the current values of Omega_i as a numpy array. 
        
        Returns: [Omega_m, Omega_c, Omega_r, Omega_b, Omega_k]"""
        
        return np.array([self.Omegam, self.Omegac, self.Omegar, self.Omegab, self.Omegak])

    def get_eos(self, omegam, omegar, omegac):
        """Return the equation of state for this cosmology. This parameter can not be changed once initialized!"""
        
        return self.eos        
    
    def set_energy_densities(self, omegam=None, omegac=None, omegag=None, omegab=None):
        """Change the initialised values of Omega_i."""
        if not omegam is None:
            self.Omegam = omegam #non-relativistic matter energy density
        if not omegac is None:
            self.Omegac = omegac #dark energy density
        if not omegag is None:
            self.Omegag = omegag
            self.Omegar = omegag * (1 + 7/8 * (4/11)**(4/3) * 3.046) #relativistic matter energy density
        if not omegab is None:
            self.Omegab = omegab #baryonic matter energy density
        
        self.Omegak = 1 - self.Omegam - self.Omegac if self.Omegar is None else 1 - self.Omegam - self.Omegac - self.Omegar #curvature energy density
        
        return self
    
    def omegar(self):
        h = self.H0/100
        z_eq = 2.5e4 * self.Omegam * h**2 * (self.T_CMB/2.7)**-4
        
        return self.Omegam / (1 + z_eq)
    
    def set_eos(self, w):
        """Change the value of the equation of state of Dark Energy."""
        
        self.eos = w
        return self
    
    def z_star(self):
        #calculate redshift of last scattering, see astro-ph:9510117
        
        if self.z_num == False or self.Omegam == 0 or self.Omegab == 0:
            return 1089
        else:
            h = self.H0/100
        
            g1 = ( .0783*(self.Omegab*h**2)**-.238 )/ (1 + 39.5 *(self.Omegab*h**2)**.763)
            g2 = .560 / (1 + 21.1 * (self.Omegab*h**2)**1.81)
        
            return 1048 * (1+ .00124 * (self.Omegab*h**2)**-.738) * (1 + g1 * (self.Omegam*h**2)**g2)

    def z_d(self):
        #calculate redshift of end of drag epoch, see astro-ph:9510117
        
        if self.z_num == False or self.Omegam == 0 or self.Omegab == 0:
            return 1089
        else:
            h = self.H0/100
        
            b1 = .313 * (self.Omegam*h**2)**-0.419 * (1 + .607*(self.Omegam*h**2)**.674)
            b2 = .238 * (self.Omegam*h**2)**.223
        
            return 1345 * (self.Omegam*h**2)**.251 / (1 + .659 * (self.Omegam*h**2)**.828) * (1 + b1 * (self.Omegab*h**2)**b2)
    
    def set_sound_horizon(self, rs):
        """Set a new value for the sound horizon."""
        self.r_sound = rs
        
        return self
        
    def sound_speed(self,z, m_nu = 0.06):
        """Compute the speed of sound in the baryon-phonton plasma."""
        if self.Omegar is None or self.Omegab is None:
            raise(ValueError('To compute a speed of sound, set Omega_r and Omega_b to numerical values first.'))
        else:
            h = self.H0/100
            #photon_density =  self.Omegar / (1 + 7/8 * (4/11)**(4/3) * 3.046)  
            
            soundspeed = self.cLight/np.sqrt(3*(1 + self.Omegab*h**2 * (31500*(2.7255/2.7)**-4 ) /(1+z)))  
        
        return soundspeed
    
    def com_sound_horizon(self, z_rec, m_nu = 0.06):
        """Compute the sound horizon at a given redshift. If Omega_r and Omega_b do not have numerical values, the default r_s is returned."""
        """For CMB, z_rec should be z_star. For BAO, use z_d"""
        
       
        if self.Omegar is None or self.Omegab is None:
            rs = self.r_sound
        elif self.rd_num == True:
            rs = self.rd(m_nu)
        else:
            rs = integrate.quad(lambda z:self.sound_speed(z,m_nu)/(self.H(z)),z_rec,np.inf)[0]
            self.r_sound = rs
        return rs
    
    def rd(self, m_nu = 0.06): 
        """Accurate Numerical approximation to the sound horizon, cf. arXiv:1411.1074"""
        h = self.H0/100
        #Omega_nu = .0101 * m_nu / h**2#eV
        Omega_nu = .0107 * m_nu / h**2#eV
        
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
        

    def luminosity_distance(self, z, eps = 1E-10):
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
    
    def model_prediction(self, dataObject):
        """Data comes in different types. Here we calculate the model prediction for each quantity"""
        
        z_d = self.z_d()
        
        data = dataObject.get_data()
        dataType = dataObject.get_dataType()
        
        rdfid = 147.78 # fiducial sound horizon in MPc
        
        
        rd = self.com_sound_horizon(z_rec = z_d) # sound horizon for given cosmology
        
        model_pred = np.zeros([len(dataType), 1])
        
        for line in range(0,len(dataType)):
            
            z = data.T[0,line]
            
            if dataType[line]=='DM*rdfid/rd':
                model_pred[line] =  self.luminosity_distance(z)/(1+z) * rdfid / rd
            
            elif dataType[line]=='DM/rd':
                model_pred[line] =  self.luminosity_distance(z)/(1+z) / rd
                
            elif dataType[line]=='DA*rdfid/rd':
                model_pred[line] =  self.luminosity_distance(z)/(1+z)**2 * rdfid / rd

            elif dataType[line]=='DA/rd':
                model_pred[line] =  self.luminosity_distance(z)/(1+z)**2 / rd
            
            elif dataType[line]=='DV*rdfid/rd':
                model_pred[line] =  ( z * self.cLight / self.H(z) * (self.luminosity_distance(z)/(1+z))**2 )**(1/3) * rdfid / rd
            
            elif dataType[line]=='DV/rd':
                model_pred[line] =  ( z * self.cLight / self.H(z) * (self.luminosity_distance(z)/(1+z))**2 )**(1/3) / rd
                
            elif dataType[line]=='rd/DV':
                model_pred[line] =  rd / ( z * self.cLight / self.H(z) * (self.luminosity_distance(z)/(1+z))**2 )**(1/3)
            
            #elif dataType[line]=='A':
                    
            elif dataType[line] == 'H*rd/rdfid':
                model_pred[line] = self.H(z) * rd / rdfid
            
            elif dataType[line] == 'DH/rd':
                model_pred[line] = self.cLight / self.H(z) / rd
                
            else:
                raise ValueError('input data doesn\'t have a recognised format')
        
        
        return model_pred.T[0]
    
    
    def log_likelihood(self, dataObject):
        """Compute the Gaussian log-likelihood for given data tuples (z, DM(z)) and covariance."""
        
        # load data:
        
        if dataObject.name == 'Quasars' or dataObject.name == 'SN':
            data = dataObject.distance_modulus()
            Cov = dataObject.data_cov()
        elif dataObject.name == 'BAO':
            data = dataObject.get_data()
            Cov = dataObject.get_cov()
            if not np.isfinite(self.com_sound_horizon(z_rec = self.z_d())):
                return -np.inf
        elif dataObject.name == 'CMB':
            if not np.isfinite(self.com_sound_horizon(z_rec = self.z_star())):
                return -np.inf
            return dataObject.log_likelihood(self)
        
        else:
            raise ValueError('Data type needs to be associated with input (e.g. BAO, quasars, ...)')
            
        
  
        # bring data into correct shape in case it isnt:
        
        if data.shape[0] == 2:
            z = data[0]
            data = data[1]
        elif data.shape[1] == 2:
            z = data[:,0]
            data = data[:,1]
        else:
            raise ValueError('Data has wrong format.')
        
        # if H(z) is nan (can occur for bigravity), return 0 likelihood
            if np.any(np.isnan(self.H(z))):
                return -np.inf
           
 
        # calculate the model predictions:
        if dataObject.name == 'Quasars' or dataObject.name == 'SN':
            model = self.distance_modulus(z)
        elif dataObject.name == 'BAO' : # or dataObject.name == 'CMB'
            model = self.model_prediction(dataObject)

        
  
        

        if len(Cov.shape) == 2:
            Cov_inv = np.linalg.inv(Cov)
            Cov_eigvals = np.linalg.eigvalsh(Cov)
            cov_len = np.shape(Cov)[1]
        elif len(Cov.shape) == 1:
            Cov_inv = np.diag(1/Cov)
            Cov_eigvals = Cov
            cov_len = 1
        else:
            raise ValueError('Cov must be 1d or 2d array')
        
        
        
        
        return -0.5 * ((model - data) @ Cov_inv @ (model - data)) - .5 * (np.sum(np.log(Cov_eigvals)) + cov_len * np.log(2*np.pi))
        #return  - .5 * (np.sum(np.log(Cov_eigvals)) + np.log(2*np.pi))
        #return model-data

class bigravity_cosmology(cosmology):
    """This class inherits from the cosmology base class and implements a bigravity cosmology."""

    #def __init__(self, log10m, theta, b1, b2, b3, omegam, omegac, rs=147.78, omegag=None, omegab=None, Hzero=70.):
    def __init__(self, log10alpha, B1, B2, B3, omegam, omegac, rs = 147.78, omegag=None, omegab = None, Hzero=70, rd_num = False, z_num = False, T_CMB = 2.7255, verbose = False):
        super().__init__(omegam, omegac=omegac, rs=rs, omegag=omegag, omegab=omegab, w=-1, Hzero=Hzero)
        #self.log10mg = log10m
        self.log10alpha = log10alpha
        self.B = np.array([B1, B2, B3])
        self.verbose = verbose #enable printing comments for analysis of solutions for y and H(z)



    def set_bigra_params(self, log10alpha,B1, B2, B3):
        """
        Change the bigravity model parameters of an existing object.

        Returns:
        The modified bigravity_cosmology object
        """

        self.log10alpha = log10alpha
        self.B = np.array([B1, B2, B3])
        return self
    
    def set_cosmo_params(self, omegam, omegag=0, Hzero=70.):
        """
        Change the cosmological parameters of an existing object.

        Returns:

        The modified bigravity_cosmology object
        """

        super().__init__(omegam, 0, omegag, w=-1, Hzero=Hzero)

        return self
    
    def graviton_mass(self,omegac):
        #Calculates the physical graviton mass using B1, B2, B3 and y0:
        
        B1, B2, B3 = self.B
        

        #B4 is determined from a3=0
        B4 = 3*B2*(10.**self.log10alpha)**2

        #ystar is determined by master-eq, plugging in b0 as defined by the dynamical CC-equation:
        #ystar = np.roots([B4*(1/(1 + 10.**(2*self.log10alpha))),
        #                  3*B3*(1/(1 + 10.**(2*self.log10alpha))),
        #                  -3*omegac+3*B2*(1/(1 + 10.**(2*self.log10alpha))),
        #                  B1*(1/(1 + 10.**(2*self.log10alpha)))])
        
        
        ystar = self.Bianchi(-1)
        
        if np.any(np.isnan(ystar)):
            if self.verbose: print('ystar has returned nan')
            return np.nan
        
        if self.verbose: print('ystar =', ystar)
        # accept only real roots:
        #print(ystar)
        #ystar = ystar.real[abs(ystar.imag)<1e-8][0]
       # print(ystar)
        
        #ystar = np.roots([-B3*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2,
        #                  (B4*(1/(1 + 10.**(2*self.log10alpha)))-3*B2*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2),
        #                  (3*B3*(1/(1 + 10.**(2*self.log10alpha)))-3*B1*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2),
        #                  (3*B2*(1/(1 + 10.**(2*self.log10alpha)))-b0*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2),
        #                  B1*(1/(1 + 10.**(2*self.log10alpha)))])
        #np.roots(B1*(1/(1 + 10.**(2*self.log10alpha)))/ystar 
        #         + (3*B2*(1/(1 + 10.**(2*self.log10alpha)))-b0*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2) 
        #         + (3*B3*(1/(1 + 10.**(2*self.log10alpha)))-3*B1*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2)*ystar  
        #         + (B4*(1/(1 + 10.**(2*self.log10alpha)))-3*B2*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2)*ystar**2
        #         - B3*(10.**(self.log10alpha)/(1 + 10.**(2*self.log10alpha))**(1/2))**2*ystar**3 )
        #print(B1,B2,B3)
        #print(self.t)
        #print(ystar)
        #print(self.log10mg)
        
        #print(ystar)
        
        mg = np.sqrt(ystar*(B1+2*ystar*B2+ystar**2*B3)) * self.H0

        if self.verbose: print('Solving cubic equation for y* and plugging into mg gives:')
        if self.verbose: print('(need to pick real solution, corresponding to e- or d-sol of master-eq)')
        return mg
    
    def del_graviton_mass(self,omegac,delH0,delB1,delB2,delB3):
        #Calculates the error:
        
        B1, B2, B3 = self.B
        
        B4 = 3*B2*(10.**self.log10alpha)**2

        #ystar is determined by master-eq, plugging in b0 as defined by the dynamical CC-equation:
        #ystar = np.roots([B4*(1/(1 + 10.**(2*self.log10alpha))),
        #                  3*B3*(1/(1 + 10.**(2*self.log10alpha))),
        #                  -3*omegac+3*B2*(1/(1 + 10.**(2*self.log10alpha))),
        #                  B1*(1/(1 + 10.**(2*self.log10alpha)))])
        
        
        ystar = self.Bianchi(-1)
        
        if np.any(np.isnan(ystar)):
            if self.verbose: print('ystar has returned nan')
            return np.nan
        
        if self.verbose: print('ystar =', ystar)
        # accept only real roots:
        #ystar = ystar.real[abs(ystar.imag)<1e-8][0]
        
        
        mg = np.sqrt(ystar*(B1+2*ystar*B2+ystar**2*B3)) * self.H0
        #calculate error of mg: (use 1/eV to compensate mass units of H0).
        #We do not propagate the error into ystar, as it does not increase the margins significantly
        delmg = np.sqrt((mg*delH0/self.H0)**2
                        + (ystar*self.H0**2/eV**2*delB1/(2*mg))**2
                        + (2*ystar**2*self.H0**2/eV**2*delB2/(2*mg))**2
                        + (ystar**3*self.H0**2/eV**2*delB3/(2*mg))**2 )

         
        return delmg
    
    
    def Bianchi(self, z): 
        """
        This is Kevin's exact solution to the Bianchi-/Master-Equation of bigravity cosmology.

        Input:    
        redshift (array!) z

        Returns:
        y = b(z) / a(z) 
        """ 
        
        # precision when taking roots and evaluating imaginary parts:
        eps = 10.**-10
        
        B1, B2, B3 = self.B
        
        if self.verbose: print('B =', self.B)
        if self.verbose: print('log10alpha =',self.log10alpha)
        
        # Determine the coefficients a:
        
        a0 = - B1 / ( (10.**self.log10alpha)**2 * B3) + 0j
        
        if not self.Omegar is None:
            #a1 = lambda z, b0: (-3*B2 + b0*(10.**self.log10alpha)**2 + (3*(self.Omegar * (1+z)**4 + self.Omegam * (1 + z)**3 + self.Omegak * (1 + z)**2 + self.Omegac)*(1 + (10.**self.log10alpha)**2)))/B3/(10.**self.log10alpha)**2+0j
            # a1 should only contain CC, rad and mat components of densities: (KMA)
            a1 = lambda z, b0: (-3*B2 + b0*(10.**self.log10alpha)**2 + (3*(self.Omegar * (1+z)**4 + self.Omegam * (1 + z)**3 + self.Omegac)*(10.**self.log10alpha)**2))/B3/(10.**self.log10alpha)**2+0j
        else:
            #a1 = lambda z, b0: (-3*B2 + b0*(10.**self.log10alpha)**2 + (3*(self.Omegam * (1 + z)**3 + self.Omegak * (1 + z)**2 + self.Omegac)*(1 + (10.**self.log10alpha)**2)))/B3/(10.**self.log10alpha)**2+0j
            # a1 should only contain CC, rad and mat components of densities: (KMA)
            a1 = lambda z, b0: (-3*B2 + b0*(10.**self.log10alpha)**2 + (3*(self.Omegam * (1 + z)**3 + self.Omegac)*(10.**self.log10alpha)**2))/B3/(10.**self.log10alpha)**2+0j
        
        a2 = (3*B1)/B3 - 3*1/(10.**self.log10alpha)**2+0j
        
        # a3 is zero, see paper
        
        
        # We now determine the solution to the cubic eq of x(z):
        
        cubic_sol = lambda z, b0: a2/3 - (2**(1/3)*(-12*a0 - a2**2))/(3*(27*a1(z,b0)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z,b0)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3)) + (27*a1(z,b0)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z,b0)**2 - 72*a0*a2 + 2*a2**3)**2))**(1/3)/(3.*2**(1/3))
        
        # We fix B0 by demanding the CC == 0 for stationary y0 (y at z=0). This depends on the correct y-sol (D-case or E-case:), so we calculate y0 in both cases.
        
        # This finds the stationary y0 (y at z=0) for the E-solution:
        y0e = fsolve(lambda y0: y0 + np.real(-np.sqrt(-a2 + cubic_sol(0., -3 * B1 * y0 - 3 * B2 * y0**2 - B3 * y0**3))/2. - np.sqrt(-a2 - cubic_sol(0., -3 * B1 * y0 - 3 * B2 * y0**2 - B3 * y0**3) + (2*a1(0,-3*B1*y0 - 3*B2*y0**2 - B3*y0**3))/np.sqrt(-a2 + cubic_sol(0., -3 * B1 * y0 - 3 * B2 * y0**2 - B3 * y0**3)))/2.), 1.)[0]
        
        # This finds the stationary y0 (y at z=0) for the D-solution:
        y0d = fsolve(lambda y0: y0 - np.real(-np.sqrt(-a2 + cubic_sol(0., -3 * B1 * y0 - 3 * B2 * y0**2 - B3 * y0**3))/2. + np.sqrt(-a2 - cubic_sol(0., -3 * B1 * y0 - 3 * B2 * y0**2 - B3 * y0**3) - (2*a1(0,-3*B1*y0 - 3*B2*y0**2 - B3*y0**3))/np.sqrt(-a2 + cubic_sol(0., -3 * B1 * y0 - 3 * B2 * y0**2 - B3 * y0**3)))/2.), 1.)[0]
                
        B0e = -3 * B1 * y0e - 3 * B2 * y0e**2 - B3 * y0e**3
        B0d = -3 * B1 * y0d - 3 * B2 * y0d**2 - B3 * y0d**3
        

        if self.verbose: print('B0d =',B0d)
        if self.verbose: print('B0e =',B0e)

        x1e = cubic_sol(z, B0e)
        x1d = cubic_sol(z, B0d)
        
        
        """
        try:
            if self.verbose: print('x1d =',x1d)
            if self.verbose: print('x1e =',x1e)
            # These line seems to be problematic, because it aborts if x1 is negative.
            #if np.any(np.imag(x1) > 10.**-6 * np.real(x1)):
            # I've added an abs on both sides (KMA). Also, the check needs to be done such that *at least one* cubicsol is real:
            #if np.any(np.abs(np.imag(x1e)) > np.abs(10.**-6 * np.real(x1e))) or np.any(np.abs(np.imag(x1d)) > np.abs(10.**-6 * np.real(x1d))):
            if np.any(np.abs(np.imag(x1e)) > np.abs(eps * np.real(x1e))) or np.any(np.abs(np.imag(x1d)) > np.abs(eps * np.real(x1d))):
                raise ValueError
        except ValueError:
            if self.verbose: print('x1 is not real')
            return np.nan

        # We may now discard any imaginary part of x1 below working precision:
        x1d = np.real(x1d)
        x1e = np.real(x1e)
        
        """
        
        if self.verbose: print('x1d =',x1d)
        if self.verbose: print('x1e =',x1e) 
        
        # We now have all ingredients to solve the master-eq:
        
        
        
      
        ye = -np.sqrt(-a2 + x1e)/2. + np.sqrt(-a2 - x1e + (2*a1(z,B0e))/np.sqrt(-a2 + x1e))/2.
        yd = np.sqrt(-a2 + x1d)/2. - np.sqrt(-a2 - x1d - (2*a1(z,B0d))/np.sqrt(-a2 + x1d))/2.
        
        if self.verbose: print('yd = ', yd)
        if self.verbose: print('ye = ', ye)

        # Check that there is exactly one y-sol which is real:
        
        yeSolReal = False if np.any(np.abs(np.imag(ye)) > np.abs(eps * np.real(ye))) else True
        ydSolReal = False if np.any(np.abs(np.imag(yd)) > np.abs(eps * np.real(yd))) else True
        
        if not ydSolReal and not yeSolReal:
            if self.verbose: print('no real solution for y')
            return np.nan
        
        # catch the exception that both y-sols are equally valid -- this can happen if the working precision is too low and e.g. y0d = real + 0.j, y0e = real + #.j, but numerically, y0d appears as real + tiny*j
        elif ydSolReal and np.any(0. <= np.real(yd)) and yeSolReal and np.any(0. <= np.real(ye)):
            if self.verbose: print('B = ', self.B)
            if self.verbose: print('log10alpha = ', self.log10alpha)
            if self.verbose: print('yd = ', yd)
            if self.verbose: print('ye = ', ye)
            if self.verbose: raise ValueError('Ambigious result: two possible solutions for y found.')
            # in the real simulation, we discard these values, assuming that the affected param space is small:
            return np.nan
        
        elif ydSolReal and np.any(0. <= np.real(yd)):
            if self.verbose: print('d-Sol selected\n')
            return np.real(yd)
            
        elif yeSolReal and np.any(0. <= np.real(ye)):
            if self.verbose: print('e-Sol selected\n')
            return np.real(ye)
        else:
            if self.verbose: print('no positive solution for y')
            return np.nan
        
        

        
        """

        if (np.all(x1e >= a2) or np.all(x1d >= a2)) and (np.all(4*(-12*a0 - a2**2)**3 + (27*a1(z,B0e)**2 - 72*a0*a2 + 2*a2**3)**2 >= 0.) or  np.all(4*(-12*a0 - a2**2)**3 + (27*a1(z,B0d)**2 - 72*a0*a2 + 2*a2**3)**2 >= 0.)) and (np.all(27*a1(z,B0e)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z,B0e)**2 - 72*a0*a2 + 2*a2**3)**2) >= 0.) or np.all(27*a1(z,B0d)**2 - 72*a0*a2 + 2*a2**3 + np.sqrt(4*(-12*a0 - a2**2)**3 + (27*a1(z,B0d)**2 - 72*a0*a2 + 2*a2**3)**2) >= 0.)):
            # for a1 -> infty as z->infty, the third solution (E) is selected. Check that the sqrts are real, that the solution is real for z=0, and that y>0: (Note that the last condition also gives a final check if y is real)
            if np.all(a1(z,B0e) >= 0) and np.all(-a2 - x1e + (2*a1(z,B0e))/np.sqrt(-a2 + x1e) >= 0.) and np.all(-a2 - cubic_sol(0,B0e) + (2*a1(0,B0e))/np.sqrt(-a2 + cubic_sol(0,B0e)) >= 0.) and np.all(-np.sqrt(-a2 + x1e)/2. + np.sqrt(-a2 - x1e + (2*a1(z,B0e))/np.sqrt(-a2 + x1e))/2. >= 0):
                #print(B0e)
                if self.verbose: print('e-sol selected')
                return np.real(-np.sqrt(-a2 + x1e)/2. + np.sqrt(-a2 - x1e + (2*a1(z,B0e))/np.sqrt(-a2 + x1e))/2.)
            
            # same for a1 -> - infty as z->infty. Now, the second solution (D) is selected.
            elif np.all(a1(z,B0d) <= 0) and np.all(-a2 - x1d - (2*a1(z,B0d))/np.sqrt(-a2 + x1d) >= 0.) and np.all(-a2 - cubic_sol(0,B0d) - (2*a1(0,B0d))/np.sqrt(-a2 + cubic_sol(0,B0d)) >= 0.) and np.all(np.sqrt(-a2 + x1d)/2. - np.sqrt(-a2 - x1d - (2*a1(z,B0d))/np.sqrt(-a2 + x1d))/2. >= 0):
                #print(B0d)
                if self.verbose: print('d-sol selected')
                return np.real(np.sqrt(-a2 + x1d)/2. - np.sqrt(-a2 - x1d - (2*a1(z,B0d))/np.sqrt(-a2 + x1d))/2.)
            
            else:
                if self.verbose: print('e, d-sol NOT selected')
                return np.nan
        else:
            if self.verbose: print('no real solution for y: a square root has evaluated to complex value')
            return np.nan
        
        """
            
        return -1
        
        
    def H(self, z):
        """
        This method overrides the cosmology.H(z) function, which computes the Hubble rate as a function of redshift. This method uses the bigravity cosmology instead.
        
        Input:
        resdshift z
        
        Output:
        Hubble rate H(z)
        """
        
        B1, B2, B3 = self.B
        y = self.Bianchi(z)
        y0 = self.Bianchi(0.)
        
        if np.any(np.isnan(y)) or np.any(np.isnan(y0)):
            if self.verbose: print('y or y0 have returned 0')
            return -np.nan* np.ones_like(z)
        
        #as we also incorporate a fit parameter omegac, we set B0 such that the dynamical CC is zero at y0:
        B0 = -3*B1*y0 - 3*B2*y0**2 - B3*y0**3
        #m = 10.**self.log10mg
        
        #after subtracting the constant part, the dynamical part of the CC contributes to H as Lambda/3:
        CC_dyn = self.H0**2 * (10.**(2*self.log10alpha)/(1 + 10.**(2*self.log10alpha))) * (B0*np.ones_like(y) + 3*B1*y + 3*B2*y**2 + B3*y**3)/3.
        
        if not self.Omegar is None:
            hubble_squared = self.H0**2 * (self.Omegar * (1+z)**4 + self.Omegam * (1+z)**3 + self.Omegak * (1 + z)**2 + self.Omegac)
        else:
            hubble_squared = self.H0**2 * (self.Omegam * (1+z)**3 + self.Omegak * (1 + z)**2 + self.Omegac)
        
        if np.any(hubble_squared + CC_dyn.T < 0):
            if self.verbose: print('Hubble squared without dynamical CC: ', hubble_squared)
            if self.verbose: print('Hubble squared, only dynamical CC component: ', CC_dyn.T)
            if self.verbose: print('Hubble rate has returned complex value')
            return -np.nan* np.ones_like(z)
        else:
            return np.sqrt(hubble_squared + CC_dyn.T)
        
        
        
    def com_sound_horizon(self, z_rec, m_nu = 0.06):
        """Compute the sound horizon at a given redshift. If Omega_r and Omega_b do not have numerical values, the default r_s is returned."""
        
        #z_d = self.z_d()
        
        if self.Omegar is None or self.Omegab is None:
            rs = self.r_sound
        else:
            x = np.logspace(-6,0,1000)
            dx = x[1:] - np.delete(x,-1)
            integrand = 1/2 * ((self.sound_speed(z_rec/x,m_nu)/(self.H(z_rec/x)) * x**-2)[1:] + np.delete((self.sound_speed(z_rec/x, m_nu)/(self.H(z_rec/x)) * x**-2), -1))
            rs = z_rec * np.sum(integrand * dx)
            self.r_sound = rs
        return rs
        
        
    def luminosity_distance(self,z,eps=1e-5):
        """This method overrides the Cosmology.luminosity_distance method. Better handling of H(z) integration."""


        dH = self.cLight / self.H0

        if isinstance(z, (float, int)):
            z_int = np.linspace(0,z,int(1/eps))
            dC = self.cLight * integrate.cumtrapz(1/self.H(z_int), z_int)[-1]
            
        
            if self.Omegak > eps: #negative curvature Universe
                sinhk = dH*np.sinh(np.sqrt(np.abs(self.Omegak)) * dC/dH)/np.sqrt(np.abs(self.Omegak))
            elif self.Omegak < - eps: #positive curvature Universe
                sinhk = dH*np.sin(np.sqrt(np.abs(self.Omegak)) * dC/dH)/np.sqrt(np.abs(self.Omegak))
            else: #flat Universe
                sinhk = dC

            return (1+z) * sinhk 
        else:
            return super().luminosity_distance(z,eps)
    
        
    def log_likelihood(self, dataObject):
        """
        This method overrides the cosmology.log_likelihood function, which computes the logarithmic likelihood given a data object. This method uses the bigravity cosmology instead.
        
        Input: 
        dataObject 
        
        Output:
        log likelihood
        """
        
        z_d = self.z_d()
        
        
        if self.Bianchi(0) == 0 or self.Bianchi(z_d) == 0:#Boring solutions with y=0 for small z or everywhere
            return -np.inf
        else:
            return super().log_likelihood(dataObject)
        
        
        

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
    
    
    def __init__(self, data, cov, dataType):
        if data.shape[1] != 2 and cov.shape[1] != 1:
            raise ValueError('Data has wrong format: data = (z, DM/rd)')
        else:
            self.data = data   
            self.cov = cov   
            self.dataType = dataType
            self.name = "BAO"
            

            
    def get_data(self):
        return self.data
    
    def get_cov(self):
        return self.cov
    
    def get_dataType(self):
        return self.dataType
    
    def distance_modulus(self,cosmo):
        # where our heroes convert all BAO data (which come from a variety of formats) into
        # the standardised dist mod
        
        z, meas = self.data.T
        dtype = self.dataType
        
        rdfid = 147.78 # fiducial sound horizon in MPc
        z_d = cosmo.z_d()
        
        rd = cosmo.com_sound_horizon(z_rec = z_d) # sound horizon for given cosmology
        
        DMpairs = np.zeros([len(dtype), 2])
        
        for line in range(0,len(dtype)): 
            if dtype[line]=='DM*rdfid/rd':
                # calculate lumi dist and DM
                lumiDist = (1 + z[line]) * meas[line]*rd/rdfid
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='DM/rd':
                # calculate lumi dist and DM
                lumiDist = (1 + z[line]) * meas[line]*rd
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
                
            elif dtype[line]=='DA*rdfid/rd':
                lumiDist = (1 + z[line])**2 * meas[line]*rd/rdfid
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )

            elif dtype[line]=='DA/rd':
                lumiDist = (1 + z[line])**2 * meas[line]*rd
                DMpairs[line] = ( z[line], 5*(np.log10(lumiDist) + 5) )
            
            elif dtype[line]=='DV*rdfid/rd':
                lumiDist =  (1 + z[line]) * ( (meas[line]*rd/rdfid)**3 * cosmo.H(z[line])/self.cLight/z[line] )**(1/2)
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
                    
            elif dtype[line] == 'H*rd/rdfid' or dtype[line] == 'DH/rd':
                continue
                
            else:
                raise ValueError('input data doesn\'t have a recognised format')

        return DMpairs
        
        
    def Hubble(self, cosmo):
        """This function extracts the measured Hubble rate at a given redshift."""
        
        z, meas = self.data.T
        dtype = self.dataType
        Hpairs = np.zeros([len(dtype), 2])
        z_d = cosmo.z_d()
        rd = cosmo.com_sound_horizon(z_rec = z_d)
        rdfid = 147.78
        
        for line in range(0,len(dtype)): 
            if dtype[line] == 'H*rd/rdfid':
                H = meas[line]*rdfid/rd
                Hpairs[line] = (z[line], H)
                    
            elif dtype[line] == 'DH/rd':
                H = self.cLight / meas[line] / rd
                Hpairs[line] = (z[line], H)
            else:
                continue
                
        return Hpairs
        
    def data_cov(self,cosmo):
        # this function returns the covariance matrix for the transformed variables (H, DM).

        z, meas = self.data.T
        dtype = self.dataType
        
        rdfid = 147.78 # fiducial sound horizon in Mpc
        z_d = cosmo.z_d()
        
        rd = cosmo.com_sound_horizon(z_rec = z_d) # sound horizon for given cosmology
        
        covDM = np.zeros([len(self.cov), len(self.cov)])
        #covDM = # self.cov
                
        for i in range(0,len(self.cov)):
            for j in range(0,len(self.cov)):
                covDM[i,j] = self.cov[i,j]
                
                #Convert distance type data into distance modulus
                if dtype[i]=='DM*rdfid/rd' or dtype[i]=='DM/rd' or dtype[i]=='DA*rdfid/rd' or dtype[i]=='DA/rd':
                    covDM[i,j] *=  (5/meas[i])/np.log(10)
                elif dtype[i]=='DV*rdfid/rd' or dtype[i]=='DV/rd' or dtype[i]=='A':
                    covDM[i,j] *=  (5*3/2/meas[i])/np.log(10)
                #Convert Hubble type data into Hubble rate
                elif dtype[i]=='H*rd/rdfid':
                    covDM[i,j] *= rdfid/rd
                elif dtype[i]=='DH/rd':
                    covDM[i,j] *= 1/meas[i]**2  * (self.cLight / rd)
                else:
                    raise(ValueError("Data type unknown"))
                    
                if dtype[j]=='DM*rdfid/rd' or dtype[j]=='DM/rd' or dtype[j]=='DA*rdfid/rd' or dtype[j]=='DA/rd':
                    covDM[i,j] *=  (5/meas[j])/np.log(10)
                elif dtype[j]=='DV*rdfid/rd' or dtype[j]=='DV/rd' or dtype[j]=='A':
                    covDM[i,j] *=  (5*3/2/meas[j])/np.log(10)
                elif dtype[j]=='H*rd/rdfid':
                    covDM[i,j] *= rdfid/rd
                elif dtype[j]=='DH/rd':
                    covDM[i,j] *= 1/meas[j]**2  * (self.cLight / rd)
                else:
                    raise(ValueError("Data type unknown"))                    

            
        #return sigmaDM.T[0]
        return covDM   # BAO errors are now also arrays due to the correlations in WiggleZ data
    
    def delta_distance_modulus(self,cosmo):
        # this function returns the error, that is the sqrt of the diagonal entries of cov


        return np.sqrt(np.diagonal(self.data_cov(cosmo)))
    


class CMB_data:
    """Objects of this class represent simplified CMB measurements."""
    #C_Planck18 = np.array([[2.1238517E-08, -9.0296572E-08, 1.7632299E-08],
    #                       [-9.0296572E-08, 1.3879427E-06, -1.2602979E-07],
    #                       [1.7632299E-08, -1.2602979E-07, 9.7141363E-08]])
    #mu_Planck18 = np.array([2.2287960E-02, 1.2116800E-01, 1.0407000E+00])# Omega_b*h**2, Omega_m*h**2 - Omega_b, 100 rd / DM!!!!!!!!!!!!!!!!!!!!
    
    
    #Using base_plikHM_TTTEEE_lowl_lowE.covmat and base_plikHM_TTTEEE_lowl_lowE.likestat of PLANCK PLA 18
    #C_Planck18 = np.array([[2.2139987E-08, -1.1786703E-07, 1.6777190E-08],
    #                       [-1.1786703E-07, 1.8664921E-06, -1.4772837E-07],
    #                       [1.6777190E-08, -1.4772837E-07, 9.5788538E-08]])
    #mu_Planck18 = np.array([2.2337930E-02, 1.2041740E-01, 1.0409010E+00])# Omega_b*h**2, Omega_m*h**2 - Omega_b*h**2, 100 rd / DM!!!!!!!!!!!!!!!!!!!!
    
    #Data and covmat from Planck2018 TTTEEE+lowE, marginalised over all other parameters. See 1808.05724; given therein are the distance priors for various models:
    mu_Planck18 = {}
    C_Planck18 = {}
    err_Planck18 = {}
    Corr_Planck18 = {}
    
    mu_Planck18['LCDM'] = np.array([1.750235, 301.4707, 0.02235976])# R, lA, Omegab*h**2
    C_Planck18['LCDM'] = np.linalg.inv(np.array([[94392.3971, -1360.4913, 1664517.2916],
                                         [-1360.4913, 161.4349, 3671.6180],
                                         [1664517.2916, 3671.6180, 79719182.5162]]))
    
    mu_Planck18['kLCDM'] = np.array([1.7429, 301.409, 0.02260])# R, lA, Omegab*h**2
    Corr_Planck18['kLCDM'] = np.array([[1, .54, -.75],
                                       [.54, 1, -.42],
                                       [-.75, -.42, 1]])
    err_Planck18['kLCDM'] = np.array([.0051,.091,.00017])
    C_Planck18['kLCDM']  = np.dot(np.diag(err_Planck18['kLCDM']), np.dot(Corr_Planck18['kLCDM'], np.diag(err_Planck18['kLCDM'])))
    
    mu_Planck18['wLCDM'] = np.array([1.7493, 301.462, 0.02239])# R, lA, Omegab*h**2
    Corr_Planck18['wLCDM'] = np.array([[1, .47, -.66],
                                       [.47, 1, -.34],
                                       [-.66, -.34, 1]])
    err_Planck18['wLCDM'] = np.array([.0047,.090,.00015])
    C_Planck18['wLCDM']  = np.dot(np.diag(err_Planck18['wLCDM']), np.dot(Corr_Planck18['wLCDM'], np.diag(err_Planck18['wLCDM'])))
    
    #Planck13 and WMAP from 1411.1074:
    C_Planck13 = 1E-7 * np.array([[1.286,-6.033, -144.3],
                                  [-6.033, 75.42, -360.5],
                                  [-144.3, -360.5, 42640]])
    mu_Planck13 = np.array([.02245, .1386, 94.33]) # Omega_b*h**2, Omega_m*h**2, DM/rd

    C_WMAP = 1E-7 * np.array([[2.864, -4.809, -111.1],
                              [-4.809, 190.8, -74.95],
                              [-111.1, -74.95, 254200]])
    mu_WMAP = np.array([.02259, .1354, 94.51]) # Omega_b*h**2, Omega_m*h**2, DM/rd
    
    def __init__(self, cosmo, satellite):
        
        if satellite == 'Planck18':
            if cosmo == 'LCDM':
                pass
            elif cosmo == 'bigravity':
                cosmo = 'LCDM'
            elif cosmo == 'kLCDM' or cosmo == 'kbigravity' or  cosmo == 'conformal':
                cosmo = 'kLCDM'
            elif cosmo == 'wLCDM':
                cosmo = 'wLCDM'
            else:
                raise(ValueError("Cosmology needs to be specified to load Planck18 data"))
        
        
        self.name = 'CMB'
        self.satellite = satellite
        if self.satellite == 'Planck18':
            self.data  = self.mu_Planck18[cosmo]  # Omega_b, Omega_DM, 100 rd/DM
            self.cov = self.C_Planck18[cosmo]
        elif self.satellite == 'Planck13':
            self.data  = self.mu_Planck13  # Omega_b, Omega_m, DM/rd
            self.cov = self.C_Planck13
        elif self.satellite == 'WMAP' or  satellite == 'Wmap':
            self.data  = self.mu_WMAP  # Omega_b, Omega_m, DM/rd
            self.cov = self.C_WMAP
        else:
            raise(ValueError("CMB data set unknown"))
            
    
    def get_data(self):
        return self.data
        
    def get_cov(self):
        return self.cov
    
    def get_satellite(self):
        return self.satellite
    
    def log_likelihood(self, cosmo):
        """Compute the log likelihood for the CMB data given a cosmology. THIS MUST GO IN COSMOLOGY CLASS EVENTUALLY!!!"""
        
        if cosmo.Omegab is None:
            raise(ValueError("Must set Omega_b before computing sound horizon!"))
        
        h = cosmo.H0 / 100.
        Omegab = cosmo.Omegab
        Omegam = cosmo.Omegam
        z_cmb = cosmo.z_star()
        #z_cmb = 1089
        rs = cosmo.com_sound_horizon(z_rec = z_cmb)
        cLight = cosmo.cLight

        if self.satellite == 'Planck18':
            mu = self.data - np.array([cosmo.H0*np.sqrt(Omegam)/cLight * cosmo.luminosity_distance(z_cmb) / (z_cmb + 1) , np.pi * cosmo.luminosity_distance(z_cmb) / (z_cmb + 1) / rs, Omegab*h**2])
        elif self.satellite == 'Planck13' or self.satellite == 'WMAP' or self.satellite == 'Wmap':
            mu = self.data - np.array([Omegab*h**2, Omegam*h**2, cosmo.luminosity_distance(z_cmb) / (z_cmb + 1) / rs])
        else:
            raise(ValueError("CMB data set unknown"))
        
        Cov_inv = np.linalg.inv(self.cov)
        Cov_eigvals = np.linalg.eigvalsh(self.cov)
        
        return -0.5 * mu @ Cov_inv @ mu - .5 * (np.sum(np.log(Cov_eigvals)) + np.shape(self.cov)[1] * np.log(2*np.pi))

    
    

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

                
            res= -.5 * np.sum( (galaxy[:,1] - model_vSquared(galaxy[:,0], YD, YB)**.5)**2 / galaxy[:,2]**2)
            if not np.isnan(res):
                self.loglike += res
            else: 
                self.loglike += -np.inf
                

from copy import copy

class likelihood:
    """This class implements a generic likelihood function to pass to a emcee sampler.
    
    Input: parameter vector theta, data sets, cosmological model name."""
    
    
    h = .6727
    omega_baryon_preset = 0.02235/h**2
    omega_gamma_preset = 2.469E-5/h**2


    
    def __init__(self, theta, data_sets, ranges_min, ranges_max, model, rd_num, z_num):
        
        self.params = theta
        self.data_sets = {}
        self.model = model
        self.rd_num = rd_num # whether the acoustic oscillation scale should be calculated numerically or analytically
        self.z_num = z_num # whether to use redshift of recombination z_d = 1089 or numerical approximation
        
        self.ranges_min, self.ranges_max = np.array(ranges_min), np.array(ranges_max) # prior ranges
        if len(ranges_min) != len(self.params) or len(ranges_max) != len(self.params):
            raise(ValueError("You must specify a minimum and a maximum value for each parameter."))
        if np.any(self.ranges_min > self.ranges_max):
            raise(ValueError("You must specify ranges with min <= max."))
            
        
        
        for sample in data_sets:
            if sample.name == 'SN':
                self.data_sets['SN'] = copy(sample)
            elif sample.name == 'Quasars':
                self.data_sets['Quasars'] = copy(sample)
            elif sample.name == 'BAO':
                self.data_sets['BAO'] = copy(sample)
            elif sample.name == 'CMB':
                self.data_sets['CMB'] = copy(sample)
            elif sample.name == 'RC' and self.model == 'conformal':
                self.data_sets['RC'] = copy(sample)
                

        if self.model == 'LCDM':
            #initialise without omega_g so that calculated omega_r is used
            Omegam, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = cosmology(omegam=Omegam, omegac= 1 - Omegam , omegab = Omegab, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num )
        elif self.model == 'kLCDM':
            Omegam, Omegac, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = cosmology(omegam=Omegam, omegac=Omegac, omegab = Omegab, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num )
            #self.cosmo = cosmology(omegam=Omegam, omegac=1 - Omegam-self.omega_gamma_preset, omegab = Omegab, omegag = self.omega_gamma_preset, Hzero=H0)
        elif self.model == 'wLCDM':
            Omegam, Omegac, Omegab, H0, w, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = cosmology(omegam=Omegam, omegac=Omegac, omegab = Omegab,  w = w, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num )
        elif self.model == 'conformal':
            gamma0, kappa, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            Omegak =  (gamma0)**2 / 2 *cosmology.cLight**2/(70./1E-3)**2#Gpc^-1
            Omegac = 1-Omegak
            #For CG, we use z_d = 1089:
            self.cosmo = cosmology(omegam=0., omegac=Omegac, omegab = Omegab, Hzero=H0, rd_num = False, z_num = False )
        elif self.model == 'bigravity':
            #B1, B2, B3, log10alpha,  Omegam, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            #self.cosmo = bigravity_cosmology(log10alpha, B1, B2, B3, omegam=Omegam, omegac=1-Omegam-self.omega_gamma_preset, omegab = Omegab, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num)
            
            #for our sampling, we set alpha = 1:
            B1, B2, B3, Omegam, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = bigravity_cosmology(0, B1, B2, B3, omegam=Omegam, omegac=1-Omegam-self.omega_gamma_preset, omegab = Omegab, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num)
        elif self.model == 'kbigravity':
            #B1, B2, B3, log10alpha,  Omegam, Omegac, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            #self.cosmo = bigravity_cosmology(log10alpha, B1, B2, B3, omegam=Omegam, omegac=Omegac, omegab = Omegab, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num )
            
            #for our sampling, we set alpha = 1:           
            B1, B2, B3,  Omegam, Omegac, Omegab, H0, a, b, MB, delta_Mhost, beta_prime, s = self.params
            self.cosmo = bigravity_cosmology(0, B1, B2, B3, omegam=Omegam, omegac=Omegac, omegab = Omegab, Hzero=H0, rd_num = self.rd_num, z_num = self.z_num )
        else: 
            raise(TypeError('Please specify which cosmology to use from [LCDM, wLCDM, bigravity, kbigravity, conformal]'))

            

        if 'SN' in self.data_sets.keys():
            self.data_sets['SN'].set_param([a, b, MB, delta_Mhost])
        if 'Quasars' in self.data_sets.keys():
            self.data_sets['Quasars'].set_param([beta_prime, s])
        if 'RC' in self.data_sets.keys() and self.model == 'conformal':
            self.data_sets['RC'].set_param([gamma0, kappa])
            self.data_sets['RC'].fit()
            
    
    def get_settings(self):
        """returns current settings of cosmology 
        """
        
        print("Calculate rd numerically: " + str(self.rd_num))
        print("Calculate zd numerically: " + str(self.z_num))
        
        


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
        """Compute a flat prior and add a gaussian prior in Omegab h^2"""
        
        if self.model == 'conformal':
            gamma0 = self.params[0]
            Omegak =  (gamma0)**2 / 2 *cosmology.cLight**2/(self.cosmo.H0/1E-3)**2#Gpc^-1
            if Omegak > 1: #Omegac < 0
                return -np.inf
        
        if self.model == 'LCDM':
            Omegab, H0 = self.params[1:3]
        elif self.model == 'bigravity':
            Omegab, H0 = self.params[4:6]
        elif self.model == 'kbigravity':
            Omegab, H0 = self.params[5:7]
        else:
            Omegab, H0 = self.params[2:4]
        h = H0 / 100

        #gauss = - (Omegab*h**2 - 0.02235)**2 / (2 * (3.7E-4)**2)
        gauss = - (Omegab*h**2 - 0.0222)**2 / (2 * (0.0005)**2)

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



# Additional code used to run the jupyter notebook:

def calcdellist(chain):
    #calculate dellist for each chain. Dellist: number of MCMC walkers which have not moved from initial position and thus need to be deleted.
        return np.unique(np.where(np.isclose(chain.get_chain()[-1] - chain.get_chain()[0], 0))[0])
    
def convertValsToTeX(vec):
    if not isinstance(vec,np.ndarray):
        return '$' + str(vec) + '$' 
    elif len(vec) == 3:
        return '$' + str(vec[0]) + '^{+' + str(vec[1]) + '}_{-' + str(vec[2]) + '}$' 
