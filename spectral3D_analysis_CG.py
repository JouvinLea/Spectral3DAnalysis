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
from gammapy.utils.energy import EnergyBounds,Energy
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
%run spectral3D_analysis_etrue.py config_crab_4runs_HAPFR.yaml 
Estimation de la morphologie de la source (notamment de la fwhm de la gaussienne qui modelise la source ponctuelle)
 a partir de la psf et de l exposure: on=bkg+psf(model*exposure)
"""
def make_empty_cube(image_size, energy,center, data_unit=""):
    """
    Parameters
    ----------
    image_size:int, Total number of pixel of the 2D map
    energy: Energybounds
    center: SkyCoord of the source
    unit : str, Data unit.
    """
    def_image=dict()
    def_image["nxpix"]=image_size
    def_image["nypix"]=image_size
    def_image["binsz"]=0.02
    def_image["xref"]=center.galactic.l.deg
    def_image["yref"]=center.galactic.b.deg
    def_image["proj"]='TAN'
    def_image["coordsys"]='GAL'
    def_image["unit"]=data_unit
    e_min=energy[0]
    e_max=energy[-1]
    nbins=len(energy)-1
    empty_cube=SkyCube.empty(emin=e_min.value, emax=e_max.value, enumbins=nbins, eunit=e_min.unit, mode='edges', **def_image)
    return empty_cube

def make_skymaskcube(ereco,image_size,  center, exclusion_region):
    """
    Compute a SkyCube mask for the region we want to exclude in the fit.

    Parameters
    ----------
    ereco: Tuple for the reconstructed energy axis: (Emin,Emax,nbins)
    image_size: size in pixel of the image
    center: SkyCoord of image center
    exclusion_region: CircleSkyRegion containing the center and the radius of the position to exclude

    """
    sky_mask_cube = make_empty_cube(image_size=image_size, energy=ereco, center=center)
    energies = sky_mask_cube.energies(mode='edges').to("TeV")
    coord_center_pix = sky_mask_cube.sky_image_ref.coordinates(mode="center").icrs
    lon = np.tile(coord_center_pix.data.lon.degree, (len(energies) - 1, 1, 1))
    lat = np.tile(coord_center_pix.data.lat.degree, (len(energies) - 1, 1, 1))
    coord_3d_center_pix = SkyCoord(lon, lat, unit="deg")
    index_excluded_region = np.where(
        (exclusion_region.center).separation(coord_3d_center_pix) < exclusion_region.radius)
    sky_mask_cube.data[:] = 1
    sky_mask_cube.data[index_excluded_region] = 0

    
input_param=yaml.load(open(sys.argv[1]))
#Input param fit and source configuration
image_size= input_param["general"]["image_size"]
extraction_region=input_param["param_fit_3D"]["extraction_region"]
freeze_bkg=input_param["param_fit_3D"]["freeze_bkg"]
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

#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_bins_true = EnergyBounds.equal_log_spacing(input_param["energy true binning"]["Emin"], input_param["energy true binning"]["Emax"], input_param["energy true binning"]["nbin"], 'TeV')
energy_reco=[Energy(input_param["energy binning"]["Emin"],"TeV"),Energy(input_param["energy binning"]["Emax"],"TeV"), input_param["energy binning"]["nbin"]]
energy_true=[Energy(input_param["energy true binning"]["Emin"],"TeV"),Energy(input_param["energy true binning"]["Emax"],"TeV"), input_param["energy true binning"]["nbin"]]

#outdir data and result
config_name = input_param["general"]["config_name"]
outdir_data = make_outdir_data(source_name, name_method_fond,config_name,image_size,for_integral_flux=True, ereco=energy_reco,etrue=energy_true,use_cube=True,use_etrue=True)
outdir_result = make_outdir_filesresult(source_name, name_method_fond,config_name,image_size,for_integral_flux=True,ereco=energy_reco,etrue=energy_true,use_cube=True,use_etrue=True)

"""
Source model paramaters initial
"""
#Dans HGPS, c est une gaussienne de 0.05deg en sigma donc *2.35 pour fwhm
#avec HESS meme une source pontuelle ne fera jamais en dessous de 0.03-0.05 degre,

if "dec" in input_param["general"]["sourde_name_skycoord"]:
        center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"], input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg").galactic
else:
        center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"]).galactic
extraction_size=input_param["param_fit_3D"]["extraction_region"]
empty_cube_reco=make_empty_cube(extraction_size, energy_bins ,center, data_unit="")
empty_cube_true=make_empty_cube(extraction_size, energy_bins_true ,center, data_unit="")

"""
Define SkyCube
"""
cube_mask=SkyCube.read("skycube_mask_CG_binE_"+str(input_param["energy binning"]["nbin"])+".fits")
index_region_selected_3d = np.where(cube_mask.data.value == 1)

counts_3D=SkyCube.read(outdir_data+"/counts_cube.fits").cutout(center,extraction_size)
coord=counts_3D.sky_image_ref.coordinates(mode="edges")
energies=counts_3D.energies(mode='edges').to("TeV")
cube=counts_3D.to_sherpa_data3d(dstype='Data3DInt')
#apply the cube_mask
cube.mask = cube_mask.data.value.ravel()

bkg_3D=SkyCube.read(outdir_data+"/bkg_cube.fits").cutout(center,extraction_size)
exposure_3D=SkyCube.read(outdir_data+"/exposure_cube.fits").cutout(center,extraction_size)
i_nan=np.where(np.isnan(exposure_3D.data))
exposure_3D.data[i_nan]=0
exposure_3D.data=exposure_3D.data*1e4
psf_SgrA=SkyCube.read(outdir_data+"/mean_psf_cube_GC.fits", format="fermi-counts").cutout(center,extraction_size)
psf_G0p9=SkyCube.read(outdir_data+"/mean_psf_cube_G0.9.fits", format="fermi-counts").cutout(center,extraction_size)
rmf=EnergyDispersion.read(outdir_data+"/mean_rmf.fits")

# Setup combined spatial and spectral model
spatial_model_SgrA = NormGauss2DInt('spatial-model_SgrA')
spectral_model_SgrA = PowLaw1D('spectral-model_SgrA')
#spectral_model_SgrA = MyPLExpCutoff('spectral-model_SgrA')
source_model_SgrA = CombinedModel3DIntConvolveEdisp(coord=coord,energies=energies,use_psf=True,exposure=exposure_3D,psf=psf_SgrA,spatial_model=spatial_model_SgrA, spectral_model=spectral_model_SgrA, edisp=rmf.data.data,select_region=True, index_selected_region=index_region_selected_3d)


# Set starting values SgrA
 
if "dec" in input_param["param_SgrA"]["sourde_name_skycoord"]:
    source_center_SgrA = SkyCoord(input_param["param_SgrA"]["sourde_name_skycoord"]["ra"], input_param["param_SgrA"]["sourde_name_skycoord"]["dec"], unit="deg").galactic
else:
    source_center_SgrA = SkyCoord.from_name(input_param["param_SgrA"]["sourde_name_skycoord"]).galactic

#center=SkyCoord.from_name("Crab").galactic
source_model_SgrA.gamma = 2.2
source_model_SgrA.xpos =  source_center_SgrA.l.value
source_model_SgrA.ypos =  source_center_SgrA.b.value
source_model_SgrA.xpos.freeze()
source_model_SgrA.ypos.freeze()
source_model_SgrA.fwhm = 0.12
source_model_SgrA.fwhm.freeze()
source_model_SgrA.ampl=1.0


bkg = TableModel('bkg')
bkg.load(None, bkg_3D.data[index_region_selected_3d].value.ravel())
# Freeze bkg amplitude
bkg.ampl=1
bkg.ampl.freeze()
model = bkg+1E-11 * (source_model_SgrA)

# Fit
# For now only Chi2 statistics seems to work, using Cash, the optimizer doesn't run at all,
# maybe because of missing background model?
fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
result = fit.fit()
err=fit.est_errors()
print(err)

# Set starting values G0p9
if "dec" in input_param["param_G0p9"]["sourde_name_skycoord"]:
        source_center_G0p9 = SkyCoord(input_param["param_G0p9"]["sourde_name_skycoord"]["ra"], input_param["param_G0p9"]["sourde_name_skycoord"]["dec"], unit="deg").galactic
else:
        source_center_G0p9 = SkyCoord.from_name(input_param["param_G0p9"]["sourde_name_skycoord"]).galactic                                      
spatial_model_G0p9 = NormGauss2DInt('spatial-model_G0p9')
spectral_model_G0p9 = PowLaw1D('spectral-model_G0p9')
#spectral_model_G0p9 = MyPLExpCutoff('spectral-model_G0p9')
source_model_G0p9 = CombinedModel3DIntConvolveEdisp(coord=coord,energies=energies,use_psf=True,exposure=exposure_3D,psf=psf_G0p9,spatial_model=spatial_model_G0p9, spectral_model=spectral_model_G0p9, edisp=rmf.data.data,select_region=True, index_selected_region=index_region_selected_3d)

source_model_G0p9.gamma = 2.2
source_model_G0p9.xpos =  source_center_G0p9.l.value
source_model_G0p9.ypos =  source_center_G0p9.b.value
source_model_G0p9.xpos.freeze()
source_model_G0p9.ypos.freeze()
source_model_G0p9.fwhm = 0.12
source_model_G0p9.fwhm.freeze()
source_model_G0p9.ampl=1.0

model = bkg+1E-11 * (source_model_SgrA) + 1E-11 * (source_model_SgrA)
fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
result = fit.fit()
err=fit.est_errors()
print(err)

# Set starting values Gauss*CS
spatial_model_central_gauss  = NormGauss2DInt('spatial-model_central_Gauss_CS')
spectral_model_central_gauss  = PowLaw1D('spectral-model_Gauss_CS')
CS_cube=SkyCube.empty_like(exposure_3D)
CS_map = SkyImage.read("CStot.fits")
if 'COMMENT' in CS_map.meta:
    del CS_map.meta['COMMENT']
cs_reproj = (CS_map.reproject(CS_cube.sky_image_ref)).cutout(center,extraction_size)
cs_reproj.data[np.where(np.isnan(cs_reproj.data))] = 0
cs_reproj.data[np.where(cs_reproj.data < input_param["param_fit_3D"]["CS"]["threshold_map"])] = 0
cs_reproj.data = cs_reproj.data / cs_reproj.data.sum()
for i in range(len(energy_bins_true)):
    CS_cube[iE,:,:]=cs_reproj.data
CS=TableModel('CS')
CS.load(None, CS_cube.data[index_region_selected_3d].value.ravel())
CS.ampl=input_param["param_fit_3D"]["CS"]["ampl_init"]


#spectral_model_central_gauss  = MyPLExpCutoff('spectral-model_Gauss_CS')
source_model_central_gauss  = CombinedModel3DIntConvolveEdisp(coord=coord,energies=energies,use_psf=True,exposure=exposure_3D,psf=psf_SgrA,spatial_model=spatial_model_Gauss_CS , spectral_model=spectral_model_Gauss_CS , edisp=rmf.data.data,select_region=True, index_selected_region=index_region_selected_3d)


#center=SkyCoord.from_name("Crab").galactic
source_model_Gauss_CS.gamma = 2.2
source_model_Gauss_CS.xpos =  source_center_SgrA.l.value
source_model_Gauss_CS.ypos =  source_center_SgrA.b.value
source_model_Gauss_CS.xpos.freeze()
source_model_Gauss_CS.ypos.freeze()
source_model_Gauss_CS.fwhm = 1
source_model_Gauss_CS.fwhm.freeze()
source_model_Gauss_CS.ampl=1.0
source_model_Gauss_CS.ampl.freeze()



# Set starting values Central component
spatial_model_central_gauss  = NormGauss2DInt('spatial-model_central_gauss')
spectral_model_central_gauss  = PowLaw1D('spectral-model_central_gauss')
#spectral_model_central_gauss  = MyPLExpCutoff('spectral-model_central_gauss')
source_model_central_gauss  = CombinedModel3DIntConvolveEdisp(coord=coord,energies=energies,use_psf=True,exposure=exposure_3D,psf=psf_SgrA,spatial_model=spatial_model_central_gauss , spectral_model=spectral_model_central_gauss , edisp=rmf.data.data,select_region=True, index_selected_region=index_region_selected_3d)


#center=SkyCoord.from_name("Crab").galactic
source_model_central_gauss.gamma = 2.2
source_model_central_gauss.xpos =  source_center_SgrA.l.value
source_model_central_gauss.ypos =  source_center_SgrA.b.value
source_model_central_gauss.xpos.freeze()
source_model_central_gauss.ypos.freeze()
source_model_central_gauss.fwhm = 0.24
source_model_central_gauss.fwhm.freeze()
source_model_central_gauss.ampl=1.0
model = bkg+1E-11 * (source_model_SgrA) + 1E-11 * (source_model_SgrA)+ 1E-11 * (source_model_central_gauss)
fit = Fit(data=cube, model=model, stat=Cash(), method=NelderMead(), estmethod=Covariance())
result = fit.fit()
err=fit.est_errors()
print(err)
"""
def PWL(E,phi_0,gamma):
    return phi_0*E**(-gamma)
def EXP(E,phi_0,gamma,beta):
    return phi_0*E**(-gamma)*np.exp(-beta*E)
etrue=EnergyBounds(exposure_3D.energies("edges")).log_centers
etrue_band=EnergyBounds(exposure_3D.energies("edges")).bands
table_result=Table()
if "spectral-model.beta" in err.parnames:
    for name,val,err_val_min,err_val_max in zip(err.parnames,err.parvals,err.parmins,err.parmaxes):
        cc = Column(np.array([val]), name=name)
        if err_val_min==None:
            cc_min = Column(np.array([0]), name=name+"_min")
        else:
            cc_min = Column(np.array([err_val_min]), name=name+"_min")
        if err_val_min==None:
            cc_max = Column(np.array([0]), name=name+"_max")
        else:
            cc_max = Column(np.array([err_val_max]), name=name+"_max")
        table_result.add_columns([cc,cc_min,cc_max])
    if "spatial-model.xpos" in table_result.colnames:
        table_result.write(outdir_result+"/table_result_EXP.fits",overwrite=True)
    else:
        table_result.write(outdir_result+"/table_result_pos_freeze_EXP.fits",overwrite=True)
    spectre=EXP(etrue.value,table_result["spatial-model.ampl"]*1e-11,table_result["spectral-model.gamma"],table_result["spectral-model.beta"])
else:
    for name,val,err_val_min,err_val_max in zip(err.parnames,err.parvals,err.parmins,err.parmaxes):
        cc = Column(np.array([val]), name=name)
        if err_val_min==None:
            cc_min = Column(np.array([0]), name=name+"_min")
        else:
            cc_min = Column(np.array([err_val_min]), name=name+"_min")
        if err_val_min==None:
            cc_max = Column(np.array([0]), name=name+"_max")
        else:
            cc_max = Column(np.array([err_val_max]), name=name+"_max")
        table_result.add_columns([cc,cc_min,cc_max])
    if "spatial-model.xpos" in table_result.colnames:
        table_result.write(outdir_result+"/table_result_PWL.fits",overwrite=True)
    else:
        table_result.write(outdir_result+"/table_result_pos_freeze_PWL.fits",overwrite=True)
    
    spectre=PWL(etrue.value, table_result["spatial-model.ampl"]*1e-11, table_result["spectral-model.gamma"])


    
 
coord=exposure_3D.sky_image_ref.coordinates(mode="edges")
d = coord.separation(center)
pix_size=exposure_3D.wcs.to_header()["CDELT2"]
i=np.where(d<pix_size*u.deg)
#i permet de faire la moyenne exposure autour de pixel autour de la source
mean_exposure=list()
for ie in range(len(exposure_3D.energies())):
    mean_exposure.append(exposure_3D.data[ie,i[0],i[1]].value.mean())
covolve_edisp=np.zeros((rmf.data.data.shape[1],exposure_3D.data.shape[0]))
for ireco in range(rmf.data.data.shape[1]):
    covolve_edisp[ireco,:]=spectre*np.asarray(mean_exposure)*rmf.data.data[:,ireco]*etrue_band
npred=np.sum(covolve_edisp,axis=1)
err_npred=np.sqrt(npred)    
nobs=np.sum(counts_3D.data-bkg_3D.data,axis=(1,2))
#err_nobs=np.sqrt(nobs)
err_nobs=np.sqrt(np.sum(counts_3D.data,axis=(1,2)))
residus=(nobs.value-npred)/npred


#Excess from the 2D fit energy band by energy band
#TODO: for the moment je met a la main  aller chercher les cartes en 2D avec 250pix car les cubes sont a 50....

outdir_result_image = make_outdir_filesresult(source_name, name_method_fond,config_name,250,for_integral_flux=False,ereco=energy_reco)
name_2D="_region_"+str(extraction_region)+"pix"
if freeze_bkg:
    name_2D+="_bkg_fix"
else:
    name_2D+="_bkg_free"   

if input_param["param_fit_3D"]["use_EM_model"]:
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
if "spectral-model.beta" in err.parnames:
    pt.savefig("npred_nobs_HAPFR_EXP_"+source_name+".png")
else:
    pt.savefig("npred_nobs_HAPFR_PWL_"+source_name+".png")
fig =pt.figure(2)
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[:3,:])
pt.plot(energy_bins.log_centers, npred, label="npred")
#pt.errorbar(energy_bins.log_centers.value, npred,yerr=err_npred, label="npred")
pt.errorbar(energy_bins.log_centers.value, nobs.value,yerr=err_nobs.value, label="nobs=counts-bkg",linestyle='None', marker="*")
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
if "spectral-model.beta" in err.parnames:
    pt.savefig("npred_nobs_HAPFR_log_EXP_"+source_name+".png")
else:
    pt.savefig("npred_nobs_HAPFR_log_PWL_"+source_name+".png")
"""



