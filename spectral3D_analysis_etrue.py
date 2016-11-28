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
from plexpcutoff import MyPLExpCutoff

from sherpa.stats import Cash, Chi2ConstVar
from sherpa.optmethods import LevMar, NelderMead
from sherpa.estmethods import Confidence, Covariance
from sherpa.fit import Fit
from gammapy.cube.sherpa_ import Data3D, CombinedModel3D, CombinedModel3DInt
from NormGauss2d import NormGauss2DInt
from gammapy.spectrum.utils import LogEnergyAxis
from gammapy.irf import EnergyDispersion
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
use_cube=input_param["general"]["use_cube"]

#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_centers=energy_bins.log_centers
energy_true_bins = EnergyBounds.equal_log_spacing(input_param["energy true binning"]["Emin"], input_param["energy true binning"]["Emax"], input_param["energy true binning"]["nbin"], 'TeV')
energy_centers_true=energy_bins.log_centers

#outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux, use_cube)
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux,use_cube)


if "dec" in input_param["general"]["sourde_name_skycoord"]:
        source_center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg")
else:
        source_center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])  
"""
Source model paramaters initial
"""
#Dans HGPS, c est une gaussienne de 0.05deg en sigma donc *2.35 pour fwhm
#avec HESS meme une source pontuelle ne fera jamais en dessous de 0.03-0.05 degre,
counts_3D=SkyCube.read(outdir_data+"/counts_cube.fits")
bkg_3D=SkyCube.read(outdir_data+"/bkg_cube.fits")
exposure_3D=SkyCube.read(outdir_data+"/exposure_cube.fits")
psf_3D=SkyCube.read(outdir_data+"/mean_psf_cube_"+source_name+".fits", format="fermi-counts")
rmf=EnergyDispersion.read(outdir_data+"/mean_rmf.fits")

wcs=counts_3D.sky_image_ref.wcs
omega=counts_3D.sky_image_ref.solid_angle().to("deg2").value
center_maps=counts_3D.sky_image_ref.center

    
fwhm_init=None
ampl_init=None
xpos_init,ypos_init=skycoord_to_pixel(source_center, wcs)
fwhm_frozen=False
ampl_frozen=False
xpos_frozen=input_param["param_fit"]["gauss_configuration"]["xpos_frozen"]
ypos_frozen=input_param["param_fit"]["gauss_configuration"]["ypos_frozen"]
source_model=source_punctual_model(source_name, fwhm_init,fwhm_frozen, ampl_init, ampl_frozen, xpos_init, xpos_frozen, ypos_init, ypos_frozen)
imax=-2
#imax=12


 
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
#counts_3D=SkyCube(name="counts3D",data=Quantity(counts," "),wcs=wcs,energy=energy_bins[0:imax+1])
#counts_3D=SkyCube(name="counts3D",data=Quantity(counts," "),wcs=wcs,energy=energy_bins[0:imax+1])
#logenergy_axis=LogEnergyAxis(energy_bins[0:imax+1],mode='edges')
#counts_3D=SkyCube(name="counts3D",data=Quantity(counts," "),wcs=wcs,energy_axis=logenergy_axis)
logenergy_true_axis=LogEnergyAxis(energy_true_bins,mode='edges')
cube=counts_3D.to_sherpa_data3d(dstype='Data3DInt', use_true_energy=True,  energy_true_axis=logenergy_true_axis)
#cube=counts_3D.to_sherpa_data3d()
exposure = TableModel('exposure')
exposure.load(None, exposure_3D.data.ravel()*1e4)
# Freeze exposure amplitude
exposure.ampl.freeze()
bkg = TableModel('bkg')
bkg.load(None, bkg_3D.data.ravel())
# Freeze bkg amplitude
bkg.ampl=1
bkg.ampl.freeze()

psf = TableModel('psf')
psf.load(None, psf_3D.data.ravel())


# Setup combined spatial and spectral model
#spatial_model = NormGauss2D('spatial-model')
spatial_model = NormGauss2DInt('spatial-model')
#spatial_model.set_wcs(wcs)
spectral_model = PowLaw1D('spectral-model')
#spectral_model = MyPLExpCutoff('spectral-model')
source_model = CombinedModel3DInt(use_psf=True,exposure=exposure_3D,psf=psf_3D.data,spatial_model=spatial_model, spectral_model=spectral_model, edisp=rmf.data)
#source_model = CombinedModel3D(use_psf=False,spatial_model=spatial_model, spectral_model=spectral_model)

# Set starting values
if "dec" in input_param["general"]["sourde_name_skycoord"]:
        center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg").galactic
else:
        center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"]).galactic
#center=SkyCoord.from_name("Crab").galactic
source_model.gamma = 2.2
source_model.xpos = center.l.value
source_model.ypos = center.b.value
#source_model.xpos.freeze()
#source_model.ypos.freeze()
source_model.fwhm = 0.12
source_model.ampl=1.0
#source_model.fwhm.freeze() 
# Adding this constant background components the fit works with cash statistics as well
#spatial_model_bkg = Const2D('spatial-model-bkg')
#spectral_model_bkg = PowLaw1D('spectral-model-bkg')
#bkg_model = CombinedModel3D(spatial_model=spatial_model_bkg, spectral_model=spectral_model_bkg)


model = bkg+1E-11 * (source_model)

# Fit
# For now only Chi2 statistics seems to work, using Cash, the optimizer doesn't run at all,
# maybe because of missing background model?
fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
result = fit.fit()
err=fit.est_errors()
print(err)

