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
from gammapy.image import disk_correlate, SkyImage, SkyMask, SkyImageList
from gammapy.background import fill_acceptance_image
from gammapy.stats import significance
from gammapy.background import OffDataBackgroundMaker
from gammapy.data import ObservationTable
import matplotlib.pyplot as plt
from gammapy.utils.scripts import make_path
from gammapy.extern.pathlib import Path
from gammapy.scripts import StackedObsImageMaker
from gammapy.data import ObservationList
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from gammapy.data import ObservationTable
from gammapy.utils.scripts import make_path
from gammapy.extern.pathlib import Path
from gammapy.scripts import StackedObsImageMaker
from gammapy.data import ObservationList
from gammapy.image import SkyImage
from gammapy.data import DataStore
import os
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

import shutil

"""
Method to compute the images and psf of a given source
"""


def make_outdir(source_name, name_bkg, n_binE,config_name,image_size):
    """

    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image

    Returns
    -------
    directory where your fits file will go
    """
    outdir = os.path.expandvars('$Image') +"/"+config_name + "/Image_" + source_name + "_bkg_" + name_bkg + "/binE_" + str(n_binE) +"_size_image_"+str(image_size)+"_pix"
    try:
        shutil.rmtree(outdir)
    except Exception:
        pass
    make_path(outdir).mkdir()
    return outdir


def make_obsdir(source_name, name_bkg,config_name):
    """

    Parameters
    ----------
    source_name: name of the source you want to compute the image
    name_bkg: name of the bkg model you use to produce your bkg image

    Returns
    -------
    directory where your data for your image are located
    """
    return os.path.expandvars('$Image') +"/"+config_name + "/Image_" + source_name + "_bkg_" + name_bkg + "/data"


def make_new_directorydataset(nobs, config_directory, source_name, center, obsdir):
    """
    Creates a directory with only the run used for the Images of the source and create a new index table with the
    background aceeptance curve location to used for the bkg image

    Parameters
    ----------
    nobs: number of observation you want
    config_directory: name of the config chains used to produce the data
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    obsdir: directory where you want to put these data

    Returns
    -------

    """
    ds = DataStore.from_dir(config_directory)
    obs = ds.obs_table
    # center = SkyCoord.from_name("Crab")
    pointing = SkyCoord(obs["RA_PNT"], obs["DEC_PNT"], unit='deg', frame='fk5')
    sep = center.separation(pointing)
    i = np.where(sep < 2 * u.deg)
    obs_table_target = obs[i]
    ds.obs_table = obs_table_target[obs_table_target["QUALITY"] == 0]
    try:
        shutil.rmtree(obsdir)
    except Exception:
        pass
    ds.copy_obs(ds.obs_table[0:nobs], obsdir)


def make_new_directorydataset_listobs(nobs, config_directory, source_name, center, obsdir, list_obs):
    """
    Creates a directory with only the run used for the Images of the source and create a new index table with the
    background aceeptance curve location to used for the bkg image.
    Used a list of observation not a selection at less tat 2deg from the pointing position

    Parameters
    ----------
    nobs: number of observation you want
    config_directory: name of the config chains used to produce the data
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    obsdir: directory where you want to put these data
    list_obs: list of obs id we want to create the new data_store

    Returns
    -------

    """
    ds = DataStore.from_dir(config_directory)
    list_ind = list()
    for obs in list_obs:
        i = np.where(ds.obs_table["OBS_ID"] == obs)[0][0]
        list_ind.append(i)
    try:
        shutil.rmtree(obsdir)
    except Exception:
        pass
    ds.copy_obs(ds.obs_table[list_ind], obsdir)


def add_bkgmodel_to_indextable(bkg_model_directory, source_name, obsdir):
    """
    Creates an indextable with the location of the bkg files you want to use to compute the bkg model

    Parameters
    ----------
    bkg_model_directory: directory where is located the bkg model you want to use for your bkg image
    source_name: name of the source you want to compute the image
    obsdir: directory where you want to put these data

    Returns
    -------

    """
    ds = DataStore.from_dir(obsdir)
    bgmaker = OffDataBackgroundMaker(ds)
    bkg_model_outdir = Path(bkg_model_directory)
    group_filename = str(bkg_model_outdir / 'group_def.fits')
    index_table = bgmaker.make_total_index_table(ds, "2D", bkg_model_outdir, group_filename, True)
    fn = obsdir + '/hdu-index.fits.gz'
    index_table.write(fn, overwrite=True)


def make_images(image_size, energy_band, offset_band, center, data_store, obs_table_subset, exclusion_mask, outdir,
                make_background_image=True, spectral_index=2.3, for_integral_flux=False, radius=10.,save_bkg_norm=True):
    """
    MAke the counts, bkg, mask, exposure, significance and ecxees images
    Parameters
    ----------
    energy_band: energy band on which you want to compute the map
    offset_band: offset band on which you want to compute the map
    center: SkyCoord of the source
    data_store: DataStore object containing the data used to coompute the image
    obs_table_subset: obs_table of the data_store containing the observations you want to use to compute the image. Could
    be smaller than the one of the datastore
    exclusion_mask: SkyMask used for the escluded regions
    outdir: directory where the fits image will go
    make_background_image: if you want to compute the bkg for the images. Most of the case yes otherwise there is only the counts image
    spectral_index: assumed spectral index to compute the exposure
    for_integral_flux: True if you want to get the inegrak flux with the exposure
    radius: Disk radius in pixels for the significance

    Returns
    -------

    """
    # TODO: fix `binarize` implementation
    image = SkyImage.empty(nxpix=image_size, nypix=image_size, binsz=0.02, xref=center.galactic.l.deg,
                           yref=center.galactic.b.deg, proj='TAN', coordsys='GAL')
    refheader = image.to_image_hdu().header
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    mosaicimages = StackedObsImageMaker(image, energy_band=energy_band, offset_band=offset_band, data_store=data_store,
                                        obs_table=obs_table_subset, exclusion_mask=exclusion_mask,save_bkg_scale=save_bkg_norm)
    mosaicimages.make_images(make_background_image=make_background_image, spectral_index=spectral_index,
                             for_integral_flux=for_integral_flux, radius=radius)
    filename = outdir + '/fov_bg_maps' + str(energy_band[0].value) + '_' + str(energy_band[1].value) + '_TeV.fits'
    if 'COMMENT' in mosaicimages.images["exclusion"].meta:
        del mosaicimages.images["exclusion"].meta['COMMENT']
    write_mosaic_images(mosaicimages, filename)
    if save_bkg_norm:
        filename_bkg_norm = outdir + '/table_bkg_norm_' + str(energy_band[0].value) + '_' + str(energy_band[1].value) + '_TeV.fits'
        mosaicimages.table_bkg_scale.write(filename_bkg_norm)


def make_images_several_energyband(image_size,energy_bins, offset_band, source_name, center, data_store, obs_table_subset,
                                   exclusion_mask, outdir, make_background_image=True, spectral_index=2.3,
                                   for_integral_flux=False, radius=10.,save_bkg_norm=True):
    """
    MAke the counts, bkg, mask, exposure, significance and ecxees images for different energy bands

    Parameters
    ----------
    energy_bins: array of energy bands on which you want to compute the map
    offset_band: offset band on which you want to compute the map
    center: SkyCoord of the source
    data_store: DataStore object containing the data used to coompute the image
    obs_table_subset: obs_table of the data_store containing the observations you want to use to compute the image. Could
    be smaller than the one of the datastore
    exclusion_mask: SkyMask used for the escluded regions
    outdir: directory where the fits image will go
    make_background_image: if you want to compute the bkg for the images. Most of the case yes otherwise there is only the counts image
    spectral_index: assumed spectral index to compute the exposure
    for_integral_flux: True if you want to get the inegrak flux with the exposure
    radius: Disk radius in pixels for the significance

    Returns
    -------
    """
    for i, E in enumerate(energy_bins[0:-1]):
        energy_band = Energy([energy_bins[i].value, energy_bins[i + 1].value], energy_bins.unit)
        print energy_band
        make_images(image_size,energy_band, offset_band, center, data_store, obs_table_subset, exclusion_mask, outdir,
                    make_background_image, spectral_index, for_integral_flux, radius,save_bkg_norm)


def make_psf(energy_band, source_name, center, ObsList, outdir, spectral_index=2.3):
    """
    Compute the mean psf for a set of observation and a given energy band
    Parameters
    ----------
    energy_band: energy band on which you want to compute the map
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    ObsList: ObservationList to use to compute the psf (could be different that the data_store for G0p9 for the GC for example)
    outdir: directory where the fits image will go
    spectral_index: assumed spectral index to compute the psf

    Returns
    -------

    """
    energy = EnergyBounds.equal_log_spacing(energy_band[0].value, energy_band[1].value, 100, energy_band.unit)
    # Here all the observations have a center at less than 2 degrees from the Crab so it will be ok to estimate the mean psf on the Crab source postion (the area is define for offset equal to 2 degrees...)
    psf_energydependent = ObsList.make_psf(center, energy, theta=None)
    psf_table = psf_energydependent.table_psf_in_energy_band(energy_band, spectral_index=spectral_index)
    Table_psf = Table()
    c1 = Column(psf_table._dp_domega, name='psf_value', unit=psf_table._dp_domega.unit)
    c2 = Column(psf_table._offset, name='theta', unit=psf_table._offset.unit)
    Table_psf.add_column(c1)
    Table_psf.add_column(c2)
    filename_psf = outdir + "/psf_table_" + source_name + "_" + str(energy_band[0].value) + '_' + str(
        energy_band[1].value) + ".fits"
    Table_psf.write(filename_psf, overwrite=True)


def make_psf_several_energyband(energy_bins, source_name, center, ObsList, outdir,
                                spectral_index=2.3):
    """
    Compute the mean psf for a set of observation for different energy bands
    Parameters
    ----------
    energy_band: energy band on which you want to compute the map
    source_name: name of the source you want to compute the image
    center: SkyCoord of the source
    ObsList: ObservationList to use to compute the psf (could be different that the data_store for G0p9 for the GC for example)
    outdir: directory where the fits image will go
    spectral_index: assumed spectral index to compute the psf

    Returns
    -------

    """
    for i, E in enumerate(energy_bins[0:-2]):
        energy_band = Energy([energy_bins[i].value, energy_bins[i + 1].value], energy_bins.unit)
        print energy_band
        make_psf(energy_band, source_name, center, ObsList, outdir, spectral_index)


def write_mosaic_images(mosaicimages, filename):
    """
    Write mosaicimages
    Parameters
    ----------
    mosaicimages
    filename

    Returns
    -------

    """
    log.info('Writing {}'.format(filename))
    mosaicimages.images.write(filename, clobber=True)
