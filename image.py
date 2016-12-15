#! /usr/bin/env python
"""Example how to make an acceptance curve and background model image.
"""
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from astropy.table import Column
from gammapy.datasets import gammapy_extra
from gammapy.background import EnergyOffsetBackgroundModel
from gammapy.utils.energy import EnergyBounds, Energy
from gammapy.data import DataStore
from gammapy.utils.axis import sqrt_space
from gammapy.image import SkyImage, SkyMask, SkyImageList
from gammapy.background import fill_acceptance_image
from gammapy.stats import significance
from gammapy.background import OffDataBackgroundMaker
from gammapy.data import ObservationTable
import matplotlib.pyplot as plt
from gammapy.utils.scripts import make_path
from gammapy.extern.pathlib import Path
from gammapy.scripts import StackedObsImageMaker
from gammapy.data import ObservationList
import shutil
import time
import yaml
import sys

plt.ion()

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from ImageSource import *
"""
./image.py "config_crab.yaml"
"""

if __name__ == '__main__':
    input_param=yaml.load(open(sys.argv[1]))
    config_directory = input_param["general"]["config_directory"]
    config_name = input_param["general"]["config_name"]
    source_name=input_param["general"]["source_name"]
    name_method_fond = input_param["general"]["name_method_fond"]
    if "dec" in input_param["general"]["sourde_name_skycoord"]:
        center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg")
    else:
        center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])    
    bkg_model_directory = input_param["general"]["bkg_model_directory"]
    name_bkg = input_param["general"]["name_method_fond"]
    nobs = input_param["general"]["nobs"]
    image_size= input_param["general"]["image_size"]
    for_integral_flux=input_param["exposure"]["for_integral_flux"]
    use_cube=input_param["general"]["use_cube"]
    # Make the directory where the data are located and create a new hdutable with the link to the acceptance curve to build the bkg images
    obsdir = make_obsdir(source_name, name_bkg,config_name)
    if input_param["general"]["make_data_outdir"]:
        if input_param["general"]["use_list_obs_file"]:
            file_obs=np.loadtxt(input_param["general"]["obs_file"])
            list_obs=[obs for obs in file_obs]
            make_new_directorydataset_listobs(nobs, config_directory+"/"+config_name, source_name, center, obsdir, list_obs)
            add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir)
        elif input_param["general"]["use_list_obs"]:
            make_new_directorydataset_listobs(nobs, config_directory+"/"+config_name, source_name, center, obsdir, input_param["general"]["list_obs"])
            add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir)
        else:
            make_new_directorydataset(nobs, config_directory+"/"+config_name, source_name, center, obsdir)
            add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir)

    # Make the images and psf model for different energy bands
    #Energy binning
    energy_reco_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
    energy_true_bins = EnergyBounds.equal_log_spacing(input_param["energy true binning"]["Emin"], input_param["energy true binning"]["Emax"], input_param["energy true binning"]["nbin"], 'TeV')
    
    outdir = make_outdir(source_name, name_bkg, len(energy_reco_bins),config_name, image_size,for_integral_flux,use_cube)
    offset_band = Angle([0, 2.49], 'deg')
    data_store = DataStore.from_dir(obsdir)
    exclusion_mask = SkyMask.read(input_param["general"]["exclusion_mask"])
    obs_table_subset = data_store.obs_table[0:nobs]
    # Cette observetion n est pas utilise par gammapy car cette observation est centre sur le Crab donc ne peut pas trouve de reflected region...
    if ((source_name=="Crab") | (source_name=="Crab_Test")):
        i_remove = np.where(obs_table_subset["OBS_ID"] == 18373)
        if len(i_remove[0]) != 0:
            obs_table_subset.remove_row(i_remove[0][0])
    
    if use_cube:
        energy_reco=[Energy(input_param["energy binning"]["Emin"],"TeV"),Energy(input_param["energy binning"]["Emax"],"TeV"), input_param["energy binning"]["nbin"]]
        energy_true=[Energy(input_param["energy true binning"]["Emin"],"TeV"),Energy(input_param["energy true binning"]["Emax"],"TeV"), input_param["energy true binning"]["nbin"]]
        make_cube(image_size, energy_reco, energy_true, offset_band, center, data_store, obs_table_subset, exclusion_mask, outdir,
                  make_background_image=True, radius=10.,save_bkg_norm=True)
    else:
        make_images_several_energyband(image_size,energy_reco_bins, offset_band, source_name, center, data_store, obs_table_subset,
                                   exclusion_mask, outdir, make_background_image=True, spectral_index=2.3,
                                   for_integral_flux=for_integral_flux, radius=10.,save_bkg_norm=True)
    
    obslist = [data_store.obs(id) for id in obs_table_subset["OBS_ID"]]
    ObsList = ObservationList(obslist)
    if use_cube:
        make_psf_cube(image_size,energy_true, source_name, center, center, ObsList, outdir,spectral_index=2.3)
        make_mean_rmf(energy_true_bins,energy_reco_bins,center,ObsList, outdir)
    else:
        make_psf_several_energyband(energy_reco_bins, source_name, center, ObsList, outdir,
                                spectral_index=2.3)
        
