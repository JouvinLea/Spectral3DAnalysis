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
from gammapy.image import bin_events_in_image, disk_correlate, SkyImage, SkyMask, SkyImageCollection
from gammapy.background import fill_acceptance_image
from gammapy.stats import significance
from gammapy.background import OffDataBackgroundMaker
from gammapy.data import ObservationTable
import matplotlib.pyplot as plt
from gammapy.utils.scripts import make_path
from gammapy.extern.pathlib import Path
from gammapy.scripts import MosaicImage
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
./image_GC.py "config_GC.yaml"
"""

if __name__ == '__main__':
    input_param=yaml.load(open(sys.argv[1]))
    config_directory = input_param["general"]["config_directory"]
    source_name=input_param["general"]["source_name"]
    name_method_fond = input_param["general"]["name_method_fond"]
    if "dec" in input_param["general"]["sourde_name_skycoord"]:
        center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg")
    else:
        center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])    
    bkg_model_directory = input_param["general"]["bkg_model_directory"]
    name_bkg = input_param["general"]["name_method_fond"]
    nobs = input_param["general"]["nobs"]

    # Make the directory where the data are located and create a new hdutable with the link to the acceptance curve to build the bkg images
    obsdir = make_obsdir(source_name, name_bkg)
    if input_param["general"]["make_data_outdir"]:
        if input_param["general"]["use_list_obs"]:
            make_new_directorydataset_listobs(nobs, config_directory, source_name, center, obsdir, input_param["general"]["list_obs"])
        else:
            make_new_directorydataset(nobs, config_directory, source_name, center, obsdir)
        add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir)

    # Make the images and psf model for different energy bands
    #Energy binning
    energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')

    outdir = make_outdir(source_name, name_bkg, len(energy_bins))
    offset_band = Angle([0, 2.49], 'deg')
    data_store = DataStore.from_dir(obsdir)
    exclusion_mask = SkyMask.read(input_param["general"]["exclusion_mask"])
    obs_table_subset = data_store.obs_table[0:nobs]
    make_images_several_energyband(energy_bins, offset_band, source_name, center, data_store, obs_table_subset,
                                   exclusion_mask, outdir, make_background_image=True, spectral_index=2.3,
                                   for_integral_flux=False, radius=10.)
    obslist = [data_store.obs(id) for id in obs_table_subset["OBS_ID"]]
    ObsList = ObservationList(obslist)
    make_psf_several_energyband(energy_bins, source_name, center, ObsList, outdir,
                                spectral_index=2.3)
    #Make psf for the source G0p9
    if "l_gal" in input_param["param_G0p9"]["sourde_name_skycoord"]:
        center2 = SkyCoord(input_param["param_G0p9"]["sourde_name_skycoord"]["l_gal"], input_param["param_G0p9"]["sourde_name_skycoord"]["b_gal"], unit='deg', frame="galactic")
    else:
        center2 = SkyCoord.from_name(input_param["param_G0p9"]["sourde_name_skycoord"])
    source_name2=input_param["param_G0p9"]["name"]
    pointing=SkyCoord(obs_table_subset["RA_PNT"], obs_table_subset["DEC_PNT"], unit='deg', frame='fk5')
    sep=center2.separation(pointing)
    i = np.where(sep < 2 * u.deg)
    obslist2=[data_store.obs(id) for id in obs_table_subset["OBS_ID"][i]]
    ObsList2= ObservationList(obslist2)

    make_psf_several_energyband(energy_bins, source_name2, center2, ObsList2, outdir,
                                spectral_index=2.3)


