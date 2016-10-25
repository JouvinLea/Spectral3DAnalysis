#! /usr/bin/env python
from sherpa.astro.ui import *
from astropy.io import fits
from astropy.table import Table
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
from gammapy.detect import compute_ts_image
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from method_fit import *
from astropy.convolution import Gaussian2DKernel

from gammapy.cube import SkyCube
from sherpa.models import NormGauss2D, PowLaw1D, TableModel, Const2D

from sherpa.stats import Cash, Chi2ConstVar
from sherpa.optmethods import LevMar, NelderMead
from sherpa.fit import Fit
from gammapy.cube.sherpa_ import Data3D, CombinedModel3D

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
    name+="_bkg_fix"
else:
    name+="_bkg_free"
    
    
#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_centers=energy_bins.log_centers

#outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins),config_name)
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name)

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
fwhm_init=None
ampl_init=None
xpos_init,ypos_init=skycoord_to_pixel(source_center, on.wcs)
fwhm_frozen=False
ampl_frozen=False
xpos_frozen=input_param["param_fit"]["gauss_configuration"]["xpos_frozen"]
ypos_frozen=input_param["param_fit"]["gauss_configuration"]["ypos_frozen"]
source_model=source_punctual_model(source_name, fwhm_init,fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen, ypos_init, ypos_frozen)
imax=-2
counts=np.zeros((len(energy_bins[0:imax]),on.data.shape[0],on.data.shape[1]))
exposure_data=np.zeros((len(energy_bins[0:imax]),on.data.shape[0],on.data.shape[1]))
bkg_data=np.zeros((len(energy_bins[0:imax]),on.data.shape[0],on.data.shape[1]))
psf_data=np.zeros((len(energy_bins[0:imax]),on.data.shape[0],on.data.shape[1]))
omega=on.solid_angle().to("deg2").value
for i_E, E in enumerate(energy_bins[0:imax]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E+1].value
    
    on = SkyImageList.read(outdir_data+"/fov_bg_maps"+str(E1)+"_"+str(E2)+"_TeV.fits")["counts"]
    on.write(outdir_data+"/on_maps"+str(E1)+"_"+str(E2)+"_TeV.fits", clobber=True)
    counts[i_E,:,:]=on.data
    exposure_data[i_E,:,:] = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["exposure"].data*1e4*omega
    #exposure_data[i_E,:,:] = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["exposure"].data
    bkg_data[i_E,:,:] = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["bkg"].data
    psf_file = Table.read(outdir_data + "/psf_table_" + source_name + "_" + str(E1) + "_" + str(E2) + ".fits")
    header = on.to_image_hdu().header
    psf_data[i_E,:,:] = fill_acceptance_image(header, on.center, psf_file["theta"].to("deg"),psf_file["psf_value"].data, psf_file["theta"].to("deg")[-1]).data


  
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from gammapy.utils.fits import table_to_fits_table
w = wcs.WCS(naxis=2)
w.wcs.crpix = [125.5, 125.5]
w.wcs.cdelt = np.array([-0.02, 0.02])
w.wcs.cunit = np.array(['deg', 'deg'])
w.wcs.crval = [83.633333333, 22.014444444]
w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
#w.wcs.set_pv([(2, 1, 45.0)])




#counts_3D=SkyCube(name="counts3D",data=Quantity(counts," "),wcs=w,energy=energy_bins[0:imax+1])
counts_3D=SkyCube(name="counts3D",data=Quantity(counts," "),wcs=on.wcs,energy=energy_bins[0:imax+1])
#counts_3D.write("test.fits.gz")
#counts_3D=SkyCube.read("test.fits.gz")
cube=counts_3D.to_sherpa_data3d()
exposure = TableModel('exposure')
exposure.load(None, exposure_data.ravel())
# Freeze exposure amplitude
exposure.ampl.freeze()
bkg = TableModel('bkg')
bkg.load(None, bkg_data.ravel())
# Freeze bkg amplitude
bkg.ampl=1
bkg.ampl.freeze()

psf = TableModel('psf')
psf.load(None, psf_data.ravel())


# Setup combined spatial and spectral model
spatial_model = NormGauss2D('spatial-model')
#spatial_model = normgauss2dint('spatial-model')
spectral_model = PowLaw1D('spectral-model')
source_model = CombinedModel3D(spatial_model=spatial_model, spectral_model=spectral_model)

# Set starting values
if "dec" in input_param["general"]["sourde_name_skycoord"]:
        center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg").galactic
else:
        center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"]).galactic
#center=SkyCoord.from_name("Crab").galactic
source_model.gamma = 2.2
source_model.xpos = center.l.value
source_model.ypos = center.b.value
source_model.fwhm = 0.12
#source_model.fwhm = 0.15
#source_model.fwhm = 0.13
#source_model.fwhm = 0.12645763691915954
#source_model.ampl = 1e-11
#source_model.ampl = 0.05
#source_model.ampl = 1e4
#source_model.ampl = 1e4
source_model.ampl=1.0
#source_model.fwhm.freeze() 
# Adding this constant background components the fit works with cash statistics as well
#spatial_model_bkg = Const2D('spatial-model-bkg')
#spectral_model_bkg = PowLaw1D('spectral-model-bkg')
#bkg_model = CombinedModel3D(spatial_model=spatial_model_bkg, spectral_model=spectral_model_bkg)

#omega=0.0004

#omega = TableModel('omega')
#omega.load(None, omega.ravel())
#model = bkg+1E-11 * exposure * (source_model) # 1E-11 flux factor
model = bkg+1E-11 * exposure * (source_model) # 1E-9 flux factor
#model = bkg+1E-11 * omega*exposure * (source_model) 
#model = bkg+1E-7 * exposure * (source_model) # 1E-9 flux factor

# Fit
# For now only Chi2 statistics seems to work, using Cash, the optimizer doesn't run at all,
# maybe because of missing background model?
fit = Fit(data=cube, model=model, stat=Cash(), method=LevMar())
#fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead())
#fit = Fit(data=cube, model=model, stat="cstat", method=LevMar())
#fit = Fit(data=cube, model=model, stat=Chi2ConstVar(), method=LevMar())
result = fit.fit()
print(result)
