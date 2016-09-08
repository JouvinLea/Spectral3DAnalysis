#! /usr/bin/env python
from sherpa.astro.ui import *
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
import astropy.units as u
from IPython.core.display import Image
from gammapy.image import SkyImageCollection, SkyImage
import astropy.units as u
import pylab as pt
from gammapy.background import fill_acceptance_image
from gammapy.utils.energy import EnergyBounds
from astropy.coordinates import Angle
from astropy.units import Quantity
from gammapy.detect import compute_ts_map
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from utilities import *
from method_fit import *
from astropy.convolution import Gaussian2DKernel
import yaml
import sys
pt.ion()

"""
./morphology_fit.py "config_crab.yaml"
Estimation de la morphologie de la source (notamment de la fwhm de la gaussienne qui modelise la source ponctuelle)
 a partir de la psf et de l exposure: on=bkg+psf(model*exposure)
"""

input_param=yaml.load(open(sys.argv[1]))
#Input param fit and source configuration
extraction_region=input_param["param_fit"]["extraction_region"]
freeze_bkg=input_param["param_fit"]["freeze_bkg"]
source_name=input_param["general"]["source_name"]
name_method_fond = input_param["general"]["name_method_fond"]
name="_region_"+str(extraction_region)+"pix"
if freeze_bkg:
    name+="_bkg_free"
else:
    name+="_bkg_fix"
    
#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_centers=energy_bins.log_centers

#outdir data and result
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins))
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins))

#Pour pouvoir definir la gaussienne centre sur la source au centre des cartes en general
E1 = energy_bins[0].value
E2 = energy_bins[1].value
on = SkyImageCollection.read(outdir_data+"/fov_bg_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")["counts"]

if "dec" in input_param["general"]["sourde_name_skycoord"]:
        source_center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg")
else:
        source_center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])  
"""
Source model paramaters initial
"""
#Dans HGPS, c est une gaussienne de 0.05deg en sigma donc *2.35 pour fwhm
#avec HESS meme une source pontuelle ne fera jamais en dessous de 0.03-0.05 degre,
fwhm_init=None
ampl_init=None
xpos_init,ypos_init=skycoord_to_pixel(source_center, on.wcs)
fwhm_frozen=False
ampl_frozen=False
xpos_frozen=input_param["param_fit"]["gauss_configuration"]["xpos_frozen"]
ypos_frozen=input_param["param_fit"]["gauss_configuration"]["ypos_frozen"]
source_model=source_punctual_model(source_center, fwhm_init,fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen, ypos_init, ypos_frozen)

for i_E, E in enumerate(energy_bins[0:-2]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E+1].value
    
    on = SkyImageCollection.read(outdir_data+"/fov_bg_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")["counts"]
    on.write(outdir_data+"/on_maps"+str(E1)+"_"+str(E2)+"_TeV.fits", clobber=True)
    data = fits.open(outdir_data+"/on_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")
    load_image(1, data)
    exposure=make_exposure_model(outdir_data, E1, E2)
    bkg=make_bkg_model(outdir_data, E1, E2, freeze_bkg)
    psf=make_psf_model(outdir_data, E1, E2, on, source_name)
    name_interest=region_interest(source_center, on, extraction_region)

    notice2d(name_interest)

    set_stat("cstat")
    set_method("neldermead")

    model=bkg
    model=make_name_model(model, psf(source_model))
    #FULL MODEL
    set_full_model(model)
    fit()
    #set_full_model(bkg+psf(exposure*mygaus)+psf(exposure*EmGal))
    #fit()
    result= get_fit_results()
    try:
        table_models = join(table_models, result_table(result, energy_centers[i_E]), join_type='outer')
    except NameError:
        table_models = result_table(result, energy_centers[i_E])


    covar()
    covar_res=get_covar_results()
    #conf()
    #covar_res=get_conf_results()
    try:
        table_covar = join(table_covar, covar_table(covar_res, energy_centers[i_E]), join_type='outer')
    except NameError:
        table_covar = covar_table(covar_res, energy_centers[i_E])

    save_resid(outdir_result+"/residual_"+name+"_"+ str("%.2f"%E1) + "_" + str("%.2f"%E2) + "_TeV.fits", clobber=True)

    #plot en significativite et ts des maps
    shape = np.shape(on.data)
    mask = get_data().mask.reshape(shape)
    map_data=SkyImage.empty_like(on)
    model_map =SkyImage.empty_like(on)
    exp_map=SkyImage.empty_like(on)
    map_data.data = get_data().y.reshape(shape) * mask
    model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
    exp_map.data= np.ones(map_data.data.shape)* mask
    kernel = Gaussian2DKernel(5)
    TS=compute_ts_map(map_data, model_map, exp_map, kernel)
    TS.write(outdir_result+"/TS_map_"+name+"_"+str("%.2f"%E1) + "_" + str("%.2f"%E2) + "_TeV.fits", clobber=True)
    sig=SkyImage.empty(TS["ts"])
    sig.data=np.sqrt(TS["ts"].data)
    sig.name="sig"
    sig.write(outdir_result+"/significance_map_"+name+"_"+ str("%.2f"%E1) + "_" + str("%.2f"%E2) + "_TeV.fits", clobber=True)
    
filename_table_result=outdir_result+"/morphology_fit_result"+name+".txt"
table_models.write(filename_table_result, format="ascii")
filename_covar_result=outdir_result+"/morphology_fit_covar_result"+name+".txt"
table_covar.write(filename_covar_result, format="ascii")
