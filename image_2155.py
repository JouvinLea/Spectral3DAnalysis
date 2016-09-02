cd ."""Example how to make an acceptance curve and background model image.
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

plt.ion()

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from ImageSource import *

if __name__ == '__main__':
    config_directory = '/Users/jouvin/Desktop/these/FITS_DATA/PA/Model_Deconvoluted_Prod26/Mpp_Std'
    center = SkyCoord(329.7169379, -30.22558839, unit='deg')
    # center = SkyCoord(83.63, 22.01, unit='deg')
    source_name = "2155"
    #bkg_model_directory = "/Users/jouvin/Desktop/these/test_Gammapy/script/modelfond/out_Tevcat__coszenbinning_zen_0_34_49_61_72_eff"
    #name_bkg = "coszenbinning_zen_0_34_49_61_72_eff"
    bkg_model_directory = "/Users/jouvin/Desktop/these/test_Gammapy/script/modelfond/out_Tevcat__coszenbinning_zen_0_27_39_49_57_65_72_15binE"
    name_bkg = "coszenbinning_zen_0_27_39_49_57_65_72_15binE"
    nobs = 3

    # Make the directory where the data are located and create a new hdutable with the link to the acceptance curve to build the bkg images
    obsdir = make_obsdir(source_name, name_bkg)
    list_obs=[33746, 33747, 33748]
    make_new_directorydataset_listobs(nobs, config_directory, source_name, center, obsdir, list_obs)
    add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir)

    # Make the images and psf model for different energy bands
    #energy_bins = EnergyBounds.equal_log_spacing(0.5, 100, 5, 'TeV')
    energy_bins = EnergyBounds.equal_log_spacing(0.5, 40, 20, 'TeV')
    outdir = make_outdir(source_name, name_bkg, len(energy_bins))
    offset_band = Angle([0, 2.49], 'deg')
    data_store = DataStore.from_dir(obsdir)
    exclusion_mask = SkyMask.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    obs_table_subset = data_store.obs_table[0:nobs]
    make_images_several_energyband(energy_bins, offset_band, source_name, center, data_store, obs_table_subset,
                                   exclusion_mask, outdir, make_background_image=True, spectral_index=2.3,
                                   for_integral_flux=False, radius=10.)
    obslist = [data_store.obs(id) for id in obs_table_subset["OBS_ID"]]
    ObsList = ObservationList(obslist)
    make_psf_several_energyband(energy_bins, source_name, center, ObsList, outdir,
                                spectral_index=2.3)
