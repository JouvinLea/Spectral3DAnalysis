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
from gammapy.cube.sherpa_ import Data3D, CombinedModel3D, CombinedModel3DInt, CombinedModel3DIntConvolveEdisp
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

#outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux, use_cube)
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux,use_cube)


"""
Source model paramaters initial
"""
#Dans HGPS, c est une gaussienne de 0.05deg en sigma donc *2.35 pour fwhm
#avec HESS meme une source pontuelle ne fera jamais en dessous de 0.03-0.05 degre,
counts_3D=SkyCube.read(outdir_data+"/counts_cube.fits")
cube=counts_3D.to_sherpa_data3d(dstype='Data3DInt')
bkg_3D=SkyCube.read(outdir_data+"/bkg_cube.fits")
exposure_3D=SkyCube.read(outdir_data+"/exposure_cube.fits")
i_nan=np.where(np.isnan(exposure_3D.data))
exposure_3D.data[i_nan]=0
exposure_3D.data=exposure_3D.data*1e4
psf_3D=SkyCube.read(outdir_data+"/mean_psf_cube_"+source_name+".fits", format="fermi-counts")
rmf=EnergyDispersion.read(outdir_data+"/mean_rmf.fits")

# Setup combined spatial and spectral model
spatial_model = NormGauss2DInt('spatial-model')
spectral_model = PowLaw1D('spectral-model')
#spectral_model = MyPLExpCutoff('spectral-model')
dimensions=[exposure_3D.data.shape[1],exposure_3D.data.shape[2],rmf.data.shape[1],exposure_3D.data.shape[0]]
source_model = CombinedModel3DIntConvolveEdisp(dimensions=dimensions,use_psf=True,exposure=exposure_3D,psf=psf_3D,spatial_model=spatial_model, spectral_model=spectral_model, edisp=rmf.data)


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

bkg = TableModel('bkg')
bkg.load(None, bkg_3D.data.value.ravel())
# Freeze bkg amplitude
bkg.ampl=1
bkg.ampl.freeze()
model = bkg+1E-11 * (source_model)

# Fit
# For now only Chi2 statistics seems to work, using Cash, the optimizer doesn't run at all,
# maybe because of missing background model?
fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
result = fit.fit()
err=fit.est_errors()
print(err)


def PWL(E,phi_0,gamma):
    return phi_0*E**gamma
def EXP(E,phi_0,gamma,beta):
    return phi_0*E**gamma*np.exp(-beta*E)
coord=exposure_3D.sky_image_ref.coordinates(mode="edges")
d = coord.separation(center)
pix_size=exposure_3D.wcs.to_header()["CDELT2"]
i=np.where(d<pix_size*u.deg)
#i permet de faire la moyenne exposure autour de pixel autour de la source
mean_exposure=list()
for ie in range(len(exposure_3D.energies())):
    mean_exposure.append(exposure_3D.data[ie,i[0],i[1]].value.mean()*1e4)
etrue=EnergyBounds(exposure_3D.energies("edges")).log_centers
etrue_band=EnergyBounds(exposure_3D.energies("edges")).bands
dic_result_fit=dict()
if "spectral-model.beta" in err.parnames:
    for name,val in zip(err.parnames,err.parvals):
        dic_result_fit[name]=val
    spectre=EXP(etrue.value,dic_result_fit["spatial-model.ampl"]*1e-11,dic_result_fit["spectral-model.gamma"],dic_result_fit["spectral-model.beta"])
else:   
    for name,val in zip(err.parnames,err.parvals):
        dic_result_fit[name]=val
    spectre=PWL(etrue.value,dic_result_fit["spatial-model.ampl"]*1e-11,dic_result_fit["spectral-model.gamma"])
covolve_edisp=np.zeros((rmf.data.shape[1],exposure_3D.data.shape[0]))
for ireco in range(rmf.data.shape[1]):
    covolve_edisp[ireco,:]=spectre*np.asarray(mean_exposure)*rmf.data[:,ireco]*etrue_band
npred=np.sum(covolve_edisp,axis=1)
    

nobs=np.sum(counts_3D.data-bkg_3D.data,axis=(1,2))
    
