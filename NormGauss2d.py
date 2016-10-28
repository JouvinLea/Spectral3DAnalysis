import numpy as np
import sherpa.astro.ui as sau
from sherpa.astro.ui import erf
from sherpa.models import ArithmeticModel, Parameter
import astropy.wcs as pywcs
from astropy.coordinates import Angle

fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
fwhm_to_sigma_erf = np.sqrt(2) * fwhm_to_sigma

def normgauss2d_erf(p, x, y):
    """Evaluate 2d gaussian using the error function"""
    #import IPython; IPython.embed()
    delt=0.5
    sigma_erf = p[3] * fwhm_to_sigma_erf
    return p[2] / 4. * ((erf.calc.calc([1, p[0], sigma_erf], x + delt)
                     - erf.calc.calc([1, p[0], sigma_erf], x - delt))
                     * (erf.calc.calc([1, p[1], sigma_erf], y + delt)
                     - erf.calc.calc([1, p[1], sigma_erf], y - delt)))

def normgauss2d_erf_deg(p, x, y, binsize):
    """Evaluate 2d gaussian using the error function"""
    sigma_erf = p[3] * fwhm_to_sigma_erf
    delt=Angle(binsize/2.,"deg")
    return p[2] / 4. * ((erf.calc.calc([1, p[0], sigma_erf], (x + delt).value)
                     - erf.calc.calc([1, p[0], sigma_erf], (x - delt).value))
                     * (erf.calc.calc([1, p[1], sigma_erf], (y + delt).value)
                        - erf.calc.calc([1, p[1], sigma_erf], (y - delt).value)))
    

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


    def calc(self, p, x, y, *args, **kwargs):
        """
        The normgauss2dint model uses the error function to evaluate the
        the gaussian. This corresponds to an integration over bins.
        """
        
        return normgauss2d_erf_deg(p, x, y,self.binsize)
        #version utilisePIXEL
        """
        coord=np.array(zip(x.value,y.value))
        pix=self.wcs.wcs_world2pix(coord,1)
        x=np.asarray([i[0] for i in pix])
        y=np.asarray([i[1] for i in pix])
        p[0],p[1]=self.wcs.wcs_world2pix(np.array([p[0],p[1]],ndmin=2),1)[0]
        p[3]=p[3]/self.binsize
        return normgauss2d_erf(p, x, y)
        """
        

sau.add_model(NormGauss2DInt)
