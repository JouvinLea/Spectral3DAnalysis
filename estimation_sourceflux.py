#! /usr/bin/env python
from sherpa.astro.ui import *
from astropy.io import fits
from astropy.table import Table
from astropy.table import join
from astropy.table import Column
import astropy.units as u
from IPython.core.display import Image
from gammapy.image import SkyImageList, SkyImage
import astropy.units as u
import pylab as pt
from gammapy.background import fill_acceptance_image
from gammapy.utils.energy import EnergyBounds
from astropy.coordinates import Angle
from astropy.units import Quantity
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from method_fit import *
import yaml
import sys
pt.ion()

"""
./estimation_sourceflux.py "config_crab.yaml"
Estimation du flux du source model a partir de la psf et de l exposure: on=bkg+psf(model*exposure)
"""


input_param=yaml.load(open(sys.argv[1]))
#Input param fit and source configuration
#Sur quelle taille de la carte on fait le fit
image_size= input_param["general"]["image_size"]
extraction_region=input_param["param_fit"]["extraction_region"]
freeze_bkg=input_param["param_fit"]["freeze_bkg"]
source_name=input_param["general"]["source_name"]
name_method_fond = input_param["general"]["name_method_fond"]
name="_region_"+str(extraction_region)+"pix"
if freeze_bkg:
    name+="_bkg_fix"
else:
    name+="_bkg_free"
for_integral_flux=input_param["exposure"]["for_integral_flux"]
#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_centers=energy_bins.log_centers

#outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux)
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux)

#Pour pouvoir definir la gaussienne centre sur la source au centre des cartes en general
E1 = energy_bins[0].value
E2 = energy_bins[1].value
on = SkyImageList.read(outdir_data+"/fov_bg_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")["counts"]

if "dec" in input_param["general"]["sourde_name_skycoord"]:
        source_center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg")
else:
        source_center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])   
"""
Source model paramaters initial
"""
#Dans HGPS, c est une gaussienne de 0.05deg en sigma donc *2.35 pour fwhm
#avec HESS meme une source pontuelle ne fera jamais en dessous de 0.03-0.05 degre,
fwhm_init=input_param["param_fit"]["gauss_configuration"]["fwhm_init"]*2.35
ampl_init=input_param["param_fit"]["gauss_configuration"]["ampl_init"]
xpos_init,ypos_init=skycoord_to_pixel(source_center, on.wcs)
fwhm_frozen=input_param["param_fit"]["gauss_configuration"]["fwhm_frozen"]
ampl_frozen=input_param["param_fit"]["gauss_configuration"]["ampl_frozen"]
xpos_frozen=input_param["param_fit"]["gauss_configuration"]["xpos_frozen"]
ypos_frozen=input_param["param_fit"]["gauss_configuration"]["ypos_frozen"]
name+="_fwhm_gauss"+str(fwhm_frozen)
if fwhm_frozen:
    name+="_value"+str(fwhm_init)
if input_param["param_fit"]["use_EM_model"]:
        name+="_emission_galactic_True"
#source_model=source_NormGauss2D(source_name, fwhm_init,fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen, ypos_init, ypos_frozen)
source_model=source_punctual_model(source_name, fwhm_init,fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen, ypos_init, ypos_frozen)

for i_E, E in enumerate(energy_bins[0:-2]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E+1].value
    print "energy band: ",E1, " TeV- ",E2, "TeV"
    on = SkyImageList.read(outdir_data+"/fov_bg_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")["counts"]
    on.write(outdir_data+"/on_maps"+str(E1)+"_"+str(E2)+"_TeV.fits", clobber=True)
    data = fits.open(outdir_data+"/on_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")
    load_image(1, data)
    exposure=make_exposure_model(outdir_data, E1, E2)
    bkg=make_bkg_model(outdir_data, E1, E2, freeze_bkg)
    psf=make_psf_model(outdir_data, E1, E2, on, source_name)
    name_interest=region_interest(source_center, on, extraction_region, extraction_region)
    notice2d(name_interest)
    set_stat("cstat")
    set_method("neldermead")
    model=bkg
    model=make_name_model(model, psf(exposure*source_model))
    #Pour le modele d emission diffuse galactic
    if input_param["param_fit"]["use_EM_model"]:
        EmGal=make_EG_model(outdir_data, on)
        model=make_name_model(model, psf(exposure*EmGal))
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
   
filename_table_result=outdir_result +"/flux_fit_result"+name+".txt"
table_models.write(filename_table_result, format="ascii")
filename_covar_result=outdir_result +"/flux_covar_result"+name+".txt"
table_covar.write(filename_covar_result, format="ascii")
