import numpy as np
from gammapy.image import SkyImageList, SkyMask, SkyImage
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
#from utilities import *
from sherpa.models import Gauss2D, NormGauss2D

pt.ion()


def make_outdir_data(source_name, name_bkg, n_binE,config_name,image_size,for_integral_flux, use_cube=False, use_etrue=False):
    """
    directory where the images of the source are stored
    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image
    n_binE: number of ereco bin
    config_name: 
    image_size:
    for_integral_flux: True if you want to compute the exposure to get the integral flux
    use_cube: True if you want to compute cube analisys
    use_etrue: True if you want to compute the exposure cube and psf mean cube in true energy

    Returns
    -------
    directory where your fits file ill go
    """
    outdir = os.path.expandvars('$Image') +"/"+config_name + "/Image_" + source_name + "_bkg_" + name_bkg + "/binE_" + str(n_binE) +"_size_image_"+str(image_size)+"_pix"
    if not for_integral_flux:
        outdir+= "_exposure_flux_diff"
    if use_cube:
        outdir+= "_cube_images"
        if use_etrue:
            outdir+= "_use_etrue"
    if os.path.isdir(outdir):
        return outdir
    else:
        print("The directory" + outdir + " doesn't exist")


def make_outdir_profile(source_name, name_bkg, n_binE,config_name,image_size,for_integral_flux,use_cube=False,use_etrue=False):
    """
    directory where we will store the profiles on lattitutde and longitude
    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image

    Returns
    -------
    directory where your fits file ill go
    """
    outdir = make_outdir_plot(source_name, name_bkg, n_binE,config_name,image_size,for_integral_flux,use_cube,use_etrue) + "/profiles"
    if os.path.isdir(outdir):
        return outdir
    else:
        make_path(outdir).mkdir()
        return outdir

def make_outdir_plot(source_name, name_bkg, n_binE,config_name, image_size,for_integral_flux, use_cube=False,use_etrue=False):
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
    outdir = make_outdir_data(source_name, name_bkg, n_binE, config_name, image_size,for_integral_flux,use_cube,use_etrue) + "/plot"
    if os.path.isdir(outdir):
        return outdir
    else:
        make_path(outdir).mkdir()
        return outdir

def make_outdir_filesresult(source_name, name_bkg, n_binE, config_name, image_size,for_integral_flux, use_cube=False,use_etrue=False):
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
    outdir = make_outdir_data(source_name, name_bkg, n_binE, config_name, image_size,for_integral_flux,use_cube,use_etrue) + "/files_result"
    if os.path.isdir(outdir):
        return outdir
    else:
        make_path(outdir).mkdir()
        return outdir

def histo_significance(significance_map, exclusion_mask):
    """

    Parameters
    ----------
    significance_map: SkyImage
        significance map
    exclusion_mask: SkyMask
        exlusion mask to use for excluding the source to build the histogram of the resulting significance of the map
        without the source (normally centered on zero)

    Returns
    -------

    """
    significance_map.data[np.isnan(significance_map.data)] = -1000
    i = np.where((exclusion_mask.data != 0) & (significance_map.data != -1000))
    #n, bins, patches = pt.hist(significance_map.data[i], range=[-3,3],bins=100)
    n, bins, patches = pt.hist(significance_map.data[i],bins=100)
    
    #import IPython; IPython.embed()
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


def source_Gauss2D(name, fwhm_init, fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen,
                          ypos_init, ypos_frozen,ellep_init=None, ellep_frozen=None, ampl_max=None):
    mygaus = Gauss2D(name)
    set_par(mygaus.fwhm, val=fwhm_init, min=None, max=None, frozen=fwhm_frozen)
    set_par(mygaus.ampl, val=ampl_init, min=0, max=ampl_max, frozen=ampl_frozen)
    set_par(mygaus.xpos, val=xpos_init, min=None, max=None, frozen=xpos_frozen)
    set_par(mygaus.ypos, val=ypos_init, min=None, max=None, frozen=ypos_frozen)
    return mygaus


def source_NormGauss2D(name, fwhm_init, fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen,
                ypos_init, ypos_frozen, ellep_fit=False, ellep_init=None, ellep_frozen=None, ampl_max=None):
    mygaus = NormGauss2D(name)
    set_par(mygaus.ampl, val=ampl_init, min=0, max=ampl_max, frozen=ampl_frozen)
    set_par(mygaus.fwhm, val=fwhm_init, min=None, max=None, frozen=fwhm_frozen)
    set_par(mygaus.xpos, val=xpos_init, min=None, max=None, frozen=xpos_frozen)
    set_par(mygaus.ypos, val=ypos_init, min=None, max=None, frozen=ypos_frozen)
    if ellep_fit:
        mygaus.ellip = ellep_init
        if ellep_frozen:
            freeze(mygaus.ellip)
        else:
            thaw(mygaus.ellip)

    return mygaus


def source_punctual_model(name, fwhm_init, fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen,
                          ypos_init, ypos_frozen,ellep_init=None, ellep_frozen=None, ampl_max=None):
    """

    Parameters
    ----------
    name: str
        name of the gaussian in the fit result
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
    mygaus = normgauss2dint(name)
    set_par(mygaus.fwhm, val=fwhm_init, min=None, max=None, frozen=fwhm_frozen)
    set_par(mygaus.ampl, val=ampl_init, min=0, max=ampl_max, frozen=ampl_frozen)
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
    exp = SkyImageList.read(outdir + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["exposure"]
    exp.write(outdir + "/exp_maps" + str(E1) + "_" + str(E2) + "_TeV.fits", clobber=True)
    load_table_model("exposure", outdir + "/exp_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")
    exposure.ampl = 1
    freeze(exposure.ampl)
    return exposure


def make_bkg_model(outdir, E1, E2, freeze_bkg, ampl_init=1):
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
    bkgmap = SkyImageList.read(outdir + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["bkg"]
    bkgmap.write(outdir + "/off_maps" + str(E1) + "_" + str(E2) + "_TeV.fits", clobber=True)
    load_table_model("bkg", outdir + "/off_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")
    set_par(bkg.ampl, val=ampl_init, min=0, max=None, frozen=freeze_bkg)
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


def make_CS_model(outdir, data_image, ampl_init,ampl_frozen, threshold_zero_value):
    CS_map = SkyImage.read("CStot.fits")
    if 'COMMENT' in CS_map.meta:
        del CS_map.meta['COMMENT']
    cs_reproj = CS_map.reproject(data_image)
    cs_reproj.data[np.where(np.isnan(cs_reproj.data))] = 0
    cs_reproj.data[np.where(cs_reproj.data < threshold_zero_value)] = 0
    cs_reproj.data = cs_reproj.data / cs_reproj.data.sum()
    cs_reproj.write(outdir + "/cs_map_reproj.fits", clobber=True)
    load_table_model("CS", outdir + "/cs_map_reproj.fits")
    set_par(CS.ampl, val=ampl_init, min=0, max=None, frozen=ampl_frozen)
    return  CS


def region_interest(source_center, data_image, height, width):
    """

    Parameters
    ----------
    source_center: SkyCoord
        coordinates of the source
    data_image: SkyImage
        on map to determine the source position in pixel
    height: int
        size in pixel of the height of the region we want to use around the source for the fit
    width: int
        size in pixel of the width of the region we want to use around the source for the fit

    Returns
    -------

    """
    x_pix = skycoord_to_pixel(source_center, data_image.wcs)[0]
    y_pix = skycoord_to_pixel(source_center, data_image.wcs)[1]
    name_interest = "box(" + str(x_pix) + "," + str(y_pix) + "," + str(width) + "," + str(height) + ")"
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


def result_table(result, energy):
    """

    Parameters
    ----------
    result : `~sherpa.fit.FitResults`
        result object from sherpa fit()
    energy : float
        energy of the band
    Returns
    -------
    table : `astropy.table.Table`
        containes the fitted parameters value of the sherpa model
    """
    list_result_vals=list(result.parvals)
    list_result_vals.insert(0, energy.value)
    list_result_vals.append(result.dof)
    list_result_vals.append(result.statval)
    list_result_names=list(result.parnames)
    list_result_names.insert(0,'energy')
    list_result_names.append('dof')
    list_result_names.append('statval')
    table=Table(np.asarray(list_result_vals), names= list_result_names)
    return table

def covar_table(covar, energy):
    """

    Parameters
    ----------
    covar : `~sherpa.fit.ErrorEstResults`
        covariance object from sherpa covar()

    Returns
    -------
    table : `astropy.table.Table`
        containes the error min and max on each fitted parameters of the model
    """
    list_covar_names=list()
    list_covar_vals=list()
    list_covar_vals.insert(0, energy.value)
    list_covar_names.insert(0,'energy')
    for i_covar,name in enumerate(covar.parnames):
        list_covar_names.append(name+"_min")
        list_covar_names.append(name+"_max")
        if not covar.parmins[i_covar]:
             list_covar_vals.append(0)
        else:
            list_covar_vals.append(covar.parmins[i_covar])
        if not covar.parmaxes[i_covar]:
            list_covar_vals.append(0)
        else:
            list_covar_vals.append(covar.parmaxes[i_covar])
    table=Table(np.asarray(list_covar_vals), names= list_covar_names)
    return table

def result_table_CG(result, step):
    """

    Parameters
    ----------
    result : `~sherpa.fit.FitResults`
        result object from sherpa fit()
    step : int
        integer matching with a particular model
    Returns
    -------
    table : `astropy.table.Table`
        containes the fitted parameters value of the sherpa model
    """
    list_result_vals=list(result.parvals)
    list_result_vals.insert(0, step)
    list_result_vals.append(result.dof)
    list_result_vals.append(result.statval)
    list_result_names=list(result.parnames)
    list_result_names.insert(0,'step')
    list_result_names.append('dof')
    list_result_names.append('statval')
    table=Table(np.asarray(list_result_vals), names= list_result_names)
    return table

def covar_table_CG(covar, step):
    """

    Parameters
    ----------
    covar : `~sherpa.fit.ErrorEstResults`
        covariance object from sherpa covar()

    Returns
    -------
    table : `astropy.table.Table`
        containes the error min and max on each fitted parameters of the model
    """
    list_covar_names=list()
    list_covar_vals=list()
    list_covar_vals.insert(0, step)
    list_covar_names.insert(0,'step')
    for i_covar,name in enumerate(covar.parnames):
        list_covar_names.append(name+"_min")
        list_covar_names.append(name+"_max")
        if not covar.parmins[i_covar]:
             list_covar_vals.append(0)
        else:
            list_covar_vals.append(covar.parmins[i_covar])
        if not covar.parmaxes[i_covar]:
            list_covar_vals.append(0)
        else:
            list_covar_vals.append(covar.parmaxes[i_covar])
    table=Table(np.asarray(list_covar_vals), names= list_covar_names)
    return table

def rebin_profile2(value, nrebin):
    """

    Parameters
    ----------
    value
    bin
    err
    nrebin

    Returns
    -------

    """

    i_rebin = np.arange(0, len(value), nrebin)
    value_rebin = np.array([])
    for i in range(len(i_rebin[:-1])):
        value_rebin = np.append(value_rebin, np.mean(value[i_rebin[i]:i_rebin[i + 1]]))
    value_rebin = np.append(value_rebin, np.mean(value[i_rebin[i + 1]:]))
    return value_rebin

def rebin_profile(value, bin, err, nrebin):
    """

    Parameters
    ----------
    value
    bin
    err
    nrebin

    Returns
    -------

    """

    i_rebin = np.arange(0, len(value), nrebin)
    value_rebin = np.array([])
    bin_rebin = np.array([])
    err_rebin = np.array([])
    for i in range(len(i_rebin[:-1])):
        value_rebin = np.append(value_rebin, np.mean(value[i_rebin[i]:i_rebin[i + 1]]))
        bin_rebin = np.append(bin_rebin, np.mean(bin[i_rebin[i]:i_rebin[i + 1]]))
        err_rebin = np.append(err_rebin, np.mean(err[i_rebin[i]:i_rebin[i + 1]]))
    value_rebin = np.append(value_rebin, np.mean(value[i_rebin[i + 1]:]))
    bin_rebin = np.append(bin_rebin, np.mean(bin[i_rebin[i + 1]:]))
    err_rebin = np.append(err_rebin, np.mean(err[i_rebin[i + 1]:]))
    return value_rebin, bin_rebin, err_rebin
