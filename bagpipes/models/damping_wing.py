from __future__ import print_function, division, absolute_import

import numpy as np
from exojax.spec import hjert

from .. import config
from .. import utils



class damping_wing(object):
    ''' adds the damping wing from the IGM from Dijkstra 2014, equation 30
    added by CJP; and DLA system, inspired by Heintz+23 and using formula from Tepper-Garcia 2006.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values 

    tau_IGM needs: 
    x_HI:  average IGM neutral hydrogen fraction
    R : the comoving distance from the galaxy to the IGM bubble (in Mpc)

    tau_DLA needs: 
    logN_HI: in units of cm^-2 (log)

    '''

    def __init__(self, wavelengths, redshift, param):
        self.wavelengths = wavelengths
        self.wave_lya = 1215.56

        attenuation = np.ones_like(wavelengths)
        if ("R" in param) & ("x_HI" in param) : 
            self.R = param['R']
            self.x_HI = param['x_HI']
        if 'logN_HI' in param: 
            self.logN_HI = float(param['logN_HI'])

        if ("R" in param) & ("x_HI" in param) : 
            myatten = self._tau_IGM(redshift, wavelengths,param)
            attenuation *= myatten
            
        if 'logN_HI' in param: 
            myatten = self._tau_DLA(redshift, wavelengths,param)
            attenuation *= myatten
        self.attenuation = attenuation
        
        # Call update method 
        self.update(redshift, param)

    
    ####
    ''' These are used for tau_ISM to get the DLA absorption if asked for '''
    ''' See Ryden+2021 textbook, and Tepper-Gacria 2006, MNRAS, 369, 2025. '''

    def _a_alpha(self, b=36, #in km/s
                        gamma_ul = 4.70e8, 
                        wave_lya = 1215.56) :

        # gamma_ul in s^-1, for Ly-alpha gamma_ul is the "damping width", = A21 = 4.70e8 / s for Ly-alpha
            
        # Ryden+21, page 42 says that for T = 80K, a should be ~0.004 for Ly-alpha: 
        # For Tepper Garcia (2006, MNRAS), b = 36 km, and it doesn't matter much from 10-100 km/s (I checked)
    
        aa2km = 1./(1e10 * 1.e3) # km per angstrom
    
        return( gamma_ul * wave_lya * aa2km / 4 / np.pi / float(b) )


    def _C_alpha(self,    f21 =0.4164,     Gamma21 = 6.265e8 ) :  # values from Morton (2003)
        # f21 is the oscliator strength of Ly-alpha
        # Gamma21  is the inverse time constant for Lyman-alpha 
        e = 4.8032e-10  # cgs
        m_e = 9.1094e-28 # cgs
        c = 2.998e10 # cgs
    
        return( 4 * np.sqrt( np.pi**3) * e**2 / m_e / c * float(f21) / float(Gamma21) ) # units of cm^2
        

    # DLA  damping wing from DLA, only variable is N_HI and redshift
    def _tau_DLA(self, redshift, wavs, param) :

        # worked in ~/Work/CEERS/highz_sed/Test\ IGM\ absorption.ipynb
        b = 36.
        a21 = self._a_alpha(b=b) # use 36 km/s
        C21 = self._C_alpha(f21=0.4164,Gamma21=6.265e8) 
        
        yp = (1+redshift) # 1 + z
        yp_obs = wavs * (1+redshift) / self.wave_lya  
    
        v = 2.998e5 * ( -yp**2 + yp_obs**2 ) / (yp**2 + yp_obs**2) # km/s
        x = v / b
    
        Hax = np.asarray( [hjert(xx,a21) for xx in x])          #self._Hax(a21, x) # np.asarray([hjert(xx,a) for xx in x])
        # the Voigt-Hjerting  function. See Tepper-Garcia 2006, MNRAS, 369, 2025.
        # This is the real part of the Fadd... function.  See Ryden+21

        N_HI = np.power(10., param["logN_HI"])

        ''' for debuging 
        ok = (wavs > 1180) & (wavs < 1300)
        print(a21, C21, N_HI)
        for w, vv, xx, h in zip(wavs[ok], v[ok], x[ok], Hax[ok]) : 
            print(w,vv,xx,h, -h * C21 * a21 * N_HI)
        '''
        
        return( np.exp(-(C21 * a21 *  N_HI * Hax)))



        
    # damping wing:  from Dijkstra 2014, equation 30.  
    def _tau_IGM( self, redshift, wavs, param ) : # comoving distance from galaxy to neutral IGM (bubble)

        R = param["R"]
        x_HI = param['x_HI']
        dv = np.asarray((wavs - 1215.56) / 1215.56 * 2.998e5) # convert to delta velocity in km/s
        mask = wavs > 1400
        tau = np.zeros_like( wavs )
        tau[mask] = 0.0
        mask = dv < 0
        tau[mask] = np.inf

        mask = (dv > 0) #& (wavs < 1400) 
        Hz = utils.cosmo.H(redshift).value
        zp1 = 1+redshift
        dv_b1 = dv + Hz * R/zp1
        tau[mask] = 2.3 * x_HI *  (600/dv_b1[mask]) * (zp1/10)**1.5

        return(np.exp( -1*tau))

    def update( self, redshift, param) :
        self.param = param

        attenuation = np.ones_like(self.wavelengths)
        if ("R" in param) & ("x_HI" in param) : 
            myatten = self._tau_IGM(redshift, self.wavelengths,param)
            attenuation *= myatten
            
        if 'logN_HI' in param:
            myatten = self._tau_DLA(redshift, self.wavelengths,param)
            '''
            print("logN_HI = ", param["logN_HI"])
            ok = np.argmin( np.abs(self.wavelengths - 1216))
            print("tau at 1216 = ", myatten[ok])
            '''
            attenuation *= myatten
        self.attenuation = attenuation
        
   
