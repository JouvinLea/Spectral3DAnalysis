import numpy as np
import sherpa.astro.ui as sau
from sherpa.astro.ui import erf
from sherpa.models import ArithmeticModel, Parameter
import astropy.wcs as pywcs
from astropy.coordinates import Angle

fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
fwhm_to_sigma_erf = np.sqrt(2) * fwhm_to_sigma

class NormGauss2DInt(ArithmeticModel):
    def __init__(self, name='normgauss2dint'):
        # Gauss source parameters
        self.wcs = pywcs.WCS()
        self.coordsys = "galactic"  # default
        self.binsize = 1.0
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.fwhm = Parameter(name, 'fwhm', 1, min=0)  # p[3]
        self.shape= None
        self.n_ebins=None
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.fwhm))
        
    def set_wcs(self, wcs):
        self.wcs = wcs
        # We assume bins have the same size along x and y axis
        self.binsize = np.abs(self.wcs.wcs.cdelt[0])
        if self.wcs.wcs.ctype[0][0:4] == 'GLON':
            self.coordsys = 'galactic'
        elif self.wcs.wcs.ctype[0][0:2] == 'RA':
            self.coordsys = 'fk5'
            #        print self.coordsys
            
    def calc(self, p, xlo,xhi, ylo,yhi, *args, **kwargs):
        """
        The normgauss2dint model uses the error function to evaluate the
        the gaussian. This corresponds to an integration over bins.
        """
        
        return self.normgauss2d(p, xlo, xhi, ylo, yhi)

   
    def normgauss2d(self,p, xlo,xhi, ylo,yhi):
        sigma_erf = p[3] * fwhm_to_sigma_erf
        return p[2] / 4. * ((erf.calc.calc([1, p[0], sigma_erf], xhi)
                         - erf.calc.calc([1, p[0], sigma_erf], xlo))
                     * (erf.calc.calc([1, p[1], sigma_erf], yhi)
                        - erf.calc.calc([1, p[1], sigma_erf], ylo)))
    
sau.add_model(NormGauss2DInt)
