import numpy as np
from gammapy.image import SkyImageCollection, SkyMask
import pylab as pt
from gammapy.utils.scripts import make_path
import math
import astropy.units as u
from scipy.optimize import curve_fit
from sherpa_models import normgauss2dint
import os
from sherpa.astro.ui import *
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
import astropy.units as u
from IPython.core.display import Image
import astropy.units as u
import pylab as pt
from gammapy.background import fill_acceptance_image
from gammapy.utils.energy import EnergyBounds
from astropy.coordinates import Angle
from astropy.units import Quantity
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from utilities import *

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


def make_outdir_filesresult(source_name, name_bkg, n_binE):
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
        n_binE) + "/files_result"
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


def source_punctual_model(source_center, fwhm_init, fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen,
                          ypos_init, ypos_frozen):
    """

    Parameters
    ----------
    source_center: SkyCoord
        coordinates of the source
    fwhm_init: float
        value initial of the fwhm
    fwhm_frozen: bool
        True if you want to froze the parameter
    ampl_init: float
        value initial of the amplitude
    ampl_frozen: bool
        True if you want to froze the parameter
    xpos_init: float
        value initial of the xpos of the source
    xpos_frozen: bool
        True if you want to froze the parameter
    ypos_init: float
        value initial of the ypos of the source
    ypos_frozen: bool
        True if you want to froze the parameter

    Returns
    -------

    """
    mygaus = normgauss2dint("g2")
    set_par(mygaus.fwhm, val=fwhm_init, min=None, max=None, frozen=fwhm_frozen)
    set_par(mygaus.ampl, val=ampl_init, min=0, max=None, frozen=ampl_frozen)
    set_par(mygaus.xpos, val=xpos_init, min=None, max=None, frozen=xpos_frozen)
    set_par(mygaus.ypos, val=ypos_init, min=None, max=None, frozen=ypos_frozen)
    return mygaus


def make_exposure_model(outdir, E1, E2):
    """

    Parameters
    ----------
    outdir: str
        directory chere are stored the data
    E1: float
        energy min
    E2: float
        energy max

    Returns
    -------

    """
    exp = SkyImageCollection.read(outdir + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["exposure"]
    exp.write(outdir + "/exp_maps" + str(E1) + "_" + str(E2) + "_TeV.fits", clobber=True)
    load_table_model("exposure", outdir + "/exp_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")
    exposure.ampl = 1
    freeze(exposure.ampl)
    return exposure


def make_bkg_model(outdir, E1, E2, freeze_bkg):
    """

    Parameters
    ----------
    outdir: str
        directory chere are stored the data
    E1: float
        energy min
    E2: float
        energy max
    freeze_bkg: bool
        True if you want to froze the norm of the bkg in the fit

    Returns
    -------

    """
    bkgmap = SkyImageCollection.read(outdir + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["bkg"]
    bkgmap.write(outdir + "/off_maps" + str(E1) + "_" + str(E2) + "_TeV.fits", clobber=True)
    load_table_model("bkg", outdir + "/off_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")
    set_par(bkg.ampl, val=1, min=0, max=None, frozen=freeze_bkg)
    return bkg


def make_psf_model(outdir, E1, E2, data_image, source_name):
    """

    Parameters
    ----------
    outdir: str
        directory chere are stored the data
    E1: float
        energy min
    E2: float
        energy max
    data_image: SkyImage
        on map to reproject the psf
    source_name: str
        name of the source to load the psf file

    Returns
    -------

    """
    psf_file = Table.read(outdir + "/psf_table_" + source_name + "_" + str(E1) + "_" + str(E2) + ".fits")
    header = data_image.to_image_hdu().header
    psf_image = fill_acceptance_image(header, data_image.center, psf_file["theta"].to("deg"),
                                      psf_file["psf_value"].data, psf_file["theta"].to("deg")[-1])
    psf_image.writeto(outdir + "/psf_image.fits", clobber=True)
    load_psf("psf", outdir + "/psf_image.fits")
    return psf


def make_EG_model(outdir, data_image):
    """

    Parameters
    ----------
    outdir: str
        directory chere are stored the data
    data_image: SkyImage
        on map to reproject the psf

    Returns
    -------

    """
    EmGal_map = SkyImage.read("HGPS_large_scale_emission_model.fits", ext=1)
    emgal_reproj = EmGal_map.reproject(data_image)
    emgal_reproj.data[np.where(np.isnan(emgal_reproj.data))] = 0
    emgal_reproj.write(outdir + "/emgal_reproj.fits", clobber=True)
    load_table_model("EmGal", outdir + "/emgal_reproj.fits")
    set_par(EmGal.ampl, val=1e-8, min=0, max=None, frozen=None)
    return EmGal


def region_interest(source_center, data_image, extraction_region):
    """

    Parameters
    ----------
    source_center: SkyCoord
        coordinates of the source
    data_image: SkyImage
        on map to determine the source position in pixel
    extraction_region: int
        size in pixel of the region we want to use around the source for the fit

    Returns
    -------

    """
    x_pix = skycoord_to_pixel(source_center, data_image.wcs)[0]
    y_pix = skycoord_to_pixel(source_center, data_image.wcs)[1]
    name_interest = "box(" + str(x_pix) + "," + str(y_pix) + "," + str(extraction_region) + "," + str(
        extraction_region) + ")"
    return name_interest


def make_name_model(model, additional_component):
    """

    Parameters
    ----------
    model: sherpa.model
        initial model
    additional_component: sherpa.model
        additional model we want to fit

    Returns
    -------

    """
    model = model + additional_component
    return model
