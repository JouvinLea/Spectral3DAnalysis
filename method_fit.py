import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from gammapy.image import SkyImageCollection, SkyMask
from gammapy.utils.energy import EnergyBounds
import pylab as pt
from gammapy.utils.scripts import make_path
import math
import astropy.units as u
from scipy.optimize import curve_fit
import os

pt.ion()


def make_outdir_data(source_name, name_bkg, n_binE):
    """
    directory where the images of the source are stored
    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image

    Returns
    -------
    directory where your fits file ill go
    """
    outdir = os.path.expandvars('$Image') + "/Image_" + source_name + "_bkg_" + name_bkg + "/binE_" + str(n_binE)
    if os.path.isdir(outdir):
        return outdir
    else:
        print("The directory" + outdir + " doesn't exist")


def make_outdir_plot(source_name, name_bkg, n_binE):
    """
    directory where we will store the plots
    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image

    Returns
    -------
    directory where your fits file ill go
    """
    outdir = os.path.expandvars('$Image') + "/Image_" + source_name + "_bkg_" + name_bkg + "/binE_" + str(
        n_binE) + "/plot"
    if os.path.isdir(outdir):
        return outdir
    else:
        make_path(outdir).mkdir()


def histo_significance(significance_map, exclusion_map):
    """

    Parameters
    ----------
    significance_map: SkyImage
        significance map
    exclusion_map: SkyMask
        exlusion mask to use for excluding the source to build the histogram of the resulting significance of the map
        without the source (normally centered on zero)

    Returns
    -------

    """
    refheader = significance_map.to_image_hdu().header
    exclusion_mask = exclusion_map.reproject(reference=refheader)
    significance_map.data[np.isnan(significance_map.data)] = -1000
    i = np.where((exclusion_mask.data != 0) & (significance_map.data != -1000))
    n, bins, patches = pt.hist(significance_map.data[i], 100)
    return n, bins, patches


def norm(x, A, mu, sigma):
    """
    Norm of a gaussian
    Parameters
    ----------
    x
    A
    mu
    sigma

    Returns
    -------

    """
    y = A / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(x - mu) ** 2 / (2 * (sigma) ** 2))
    return y
