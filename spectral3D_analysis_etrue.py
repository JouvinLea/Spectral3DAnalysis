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
use_etrue=input_param["general"]["use_etrue"]
if not use_etrue:
    print "With this script normally use_etrue=True and you put it at False..."
#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')

#outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux, use_cube,use_etrue)
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux,use_cube,use_etrue)


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
    return phi_0*E**(-gamma)
def EXP(E,phi_0,gamma,beta):
    return phi_0*E**(-gamma)*np.exp(-beta*E)
coord=exposure_3D.sky_image_ref.coordinates(mode="edges")
d = coord.separation(center)
pix_size=exposure_3D.wcs.to_header()["CDELT2"]
i=np.where(d<pix_size*u.deg)
#i permet de faire la moyenne exposure autour de pixel autour de la source
mean_exposure=list()
for ie in range(len(exposure_3D.energies())):
    mean_exposure.append(exposure_3D.data[ie,i[0],i[1]].value.mean())
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
err_npred=np.sqrt(npred)    
nobs=np.sum(counts_3D.data-bkg_3D.data,axis=(1,2))
#err_nobs=np.sqrt(nobs)
err_nobs=np.sqrt(np.sum(counts_3D.data,axis=(1,2)))
residus=(nobs.value-npred)/npred


#Excess from the 2D fit energy band by energy band
#TODO: for the moment je met a la main  aller chercher les cartes en 2D avec 250pix car les cubes sont a 50....

outdir_result_image = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name,250,for_integral_flux=False)
name_2D="_region_"+str(extraction_region)+"pix"
if freeze_bkg:
    name_2D+="_bkg_fix"
else:
    name_2D+="_bkg_free"   

if input_param["param_fit"]["use_EM_model"]:
    name_2D+="_emission_galactic_True"
#store the fit result for the model of the source
filename_table_result=outdir_result_image+"/morphology_fit_result"+name_2D+".txt"
filename_covar_result=outdir_result_image+"/morphology_fit_covar_result"+name_2D+".txt"
table_models= Table.read(filename_table_result, format="ascii")
table_covar=Table.read(filename_covar_result, format="ascii")
#imax: until which energy bin we want to plot
imax=input_param["param_plot"]["imax"]
E=table_models["energy"][0:imax]
excess=table_models[source_name+".ampl"][0:imax]
err_excess_min=table_covar[source_name+".ampl_min"][0:imax]
err_excess_max=table_covar[source_name+".ampl_max"][0:imax]
residus_2D=(excess-npred[0:imax])/npred[0:imax]
iok=np.where(err_excess_min!=0)[0]
#iok=[0,1,2,3,4,5,6,7,8,9]
import matplotlib.gridspec as gridspec
fig =pt.figure(1)
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[:3,:])
#pt.subplot(2,1,1)
pt.plot(energy_bins.log_centers, npred, label="npred")
#pt.errorbar(energy_bins.log_centers.value, npred,yerr=err_npred, label="npred")
pt.errorbar(energy_bins.log_centers.value, nobs.value,yerr=err_nobs.value, label="nobs=counts-bkg",linestyle='None', marker="*")
pt.errorbar(E[iok], excess[iok], yerr=[err_excess_min[iok], err_excess_max[iok]], linestyle="None", label="fit 2D")
pt.ylabel("counts")
pt.xscale("log")
pt.legend()
ax2 = fig.add_subplot(gs[3,:],sharex=ax1) 
#pt.subplot(2,1,2)
pt.errorbar(energy_bins.log_centers.value, residus,yerr=err_nobs.value/npred,linestyle='None', marker="*", label="obs=counts-bkg")
pt.errorbar(E[iok], residus_2D[iok],yerr=[err_excess_min[iok]/npred[0:imax][iok], err_excess_max[iok]/npred[0:imax][iok]],linestyle='None', marker="*", label="fit 2D")
pt.xlabel("Energy (TeV)")
pt.ylabel("(nobs-npred)/npred")
pt.xscale("log")
pt.axhline(y=0, color='red',linewidth=2)
pt.legend(fontsize = 'x-small')
pt.subplots_adjust(hspace=0.1)
#pt.savefig("npred_nobs_PA_"+source_name+".png")
pt.savefig("npred_nobs_HAPFR_"+source_name+".png")
fig =pt.figure(2)
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[:3,:])
pt.plot(energy_bins.log_centers, npred, label="npred")
#pt.errorbar(energy_bins.log_centers.value, npred,yerr=err_npred, label="npred")
pt.errorbar(energy_bins.log_centers.value, nobs.value,yerr=err_nobs.value, label="nobs",linestyle='None', marker="*")
pt.errorbar(E[iok], excess[iok], yerr=[err_excess_min[iok], err_excess_max[iok]], linestyle="None", label="fit 2D")
pt.ylabel("counts")
pt.xscale("log")
pt.yscale("log")
pt.legend()
ax2 = fig.add_subplot(gs[3,:],sharex=ax1) 
pt.errorbar(energy_bins.log_centers.value, residus,yerr=err_nobs.value/npred,linestyle='None', marker="*", label="obs=counts-bkg")
pt.errorbar(E[iok], residus_2D[iok],yerr=[err_excess_min[iok]/npred[0:imax][iok], err_excess_max[iok]/npred[0:imax][iok]],linestyle='None', marker="*", label="fit 2D")
pt.xlabel("Energy (TeV)")
pt.ylabel("(nobs-npred)/npred")
pt.xscale("log")
pt.legend(fontsize = 'x-small')
pt.axhline(y=0, color='red',linewidth=2)
#pt.savefig("npred_nobs_PA_log_"+source_name+".png")
pt.savefig("npred_nobs_HAPFR_log_"+source_name+".png")




