#! /usr/bin/env python
# encoding: UTF-8
from astropy.table import Table
import numpy as np
import pylab as pt
from matplotlib.backends.backend_pdf import PdfPages
from gammapy.utils.energy import EnergyBounds
from method_fit import *
from method_plot import *
import yaml
import sys
pt.ion()
"""
./plot_spectra.py "config_crab.yaml"
plot le flux des differentes composantes utilisees pour fitter le on quand on estime le flux dans la source
"""

input_param=yaml.load(open(sys.argv[1]))
#Input param fit and source configuration
extraction_region=input_param["param_fit"]["extraction_region"]
freeze_bkg=input_param["param_fit"]["freeze_bkg"]
source_name=input_param["general"]["source_name"]
name_method_fond = input_param["general"]["name_method_fond"]
name="_region_"+str(extraction_region)+"pix"
if freeze_bkg:
    name+="_bkg_free"
else:
    name+="_bkg_fix"
fwhm_frozen=input_param["param_fit"]["gauss_configuration"]["fwhm_frozen"]
name+="_fwhm_gauss"+str(fwhm_frozen)
if fwhm_frozen:
    name+="_value"+str(input_param["param_fit"]["gauss_configuration"]["fwhm_init"]*2.35)
if input_param["param_fit"]["use_EM_model"]:
    name+="_emission_galactic_True"
#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_centers=energy_bins.log_centers

#outdir result and plot
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins))
outdir_plot = make_outdir_plot(source_name, name_method_fond, len(energy_bins))

#store the fit result for the model of the source
filename_table_result=outdir_result+"/flux_fit_result"+name+".txt"
filename_covar_result=outdir_result+"/flux_covar_result"+name+".txt"
table_models= Table.read(filename_table_result, format="ascii")
table_covar=Table.read(filename_covar_result, format="ascii")

#imax: until which energy bin we want to plot
imax=input_param["param_plot"]["imax"]
E=table_models["energy"][0:imax]
flux=table_models["g2.ampl"][0:imax]*1e-4 #Pour l'avoir en cm2
sup=table_covar["g2.ampl_max"][0:imax]*1e-4
inf=table_covar["g2.ampl_min"][0:imax]*1e-4
#Eth: energy binninf to reprent the PWL or EXP fitted with HESS
Eth=EnergyBounds.equal_log_spacing(E[0], E[-1], 100, 'TeV').value

pdf=PdfPages(outdir_plot+"/spectra_"+name+".pdf")
#plot le flux de la gaussienne
plot_spectra_source(pdf, E, flux, inf, sup, Eth, input_param["param_fit_HESS"])   
#If "bkg.ampl" is in table_models.colnames that means we let the bkg free and we want to plot the normalisation for the different energy bin
#plot l amplitude du fond si il a ete laisse libre dans le fit
if "bkg.ampl" in table_models.colnames:
    plot_bkg_norm(pdf, E, table_models["bkg.ampl"][0:imax], table_covar["bkg.ampl_min"][0:imax], table_covar["bkg.ampl_max"][0:imax])
#Si jamais on fite d'autre composante genre le fond galactique pour J1813...    
if len(table_models.colnames)>2:
    for name in table_models.colnames[2:]:
        if ((name=="energy") |(name=="g2.ampl") |(name=="dof") | (name=="statval")):
                continue
        elif (name=="g2.fwhm"):
            plot_param(pdf,E,name,table_models[name][0:imax],table_covar[name+"_min"][0:imax], table_covar[name+"_max"][0:imax])
        
        else:    
            plot_flux_component(pdf,E,name,table_models[name][0:imax]*1e-4,table_covar[name+"_min"][0:imax]*1e-4, table_covar[name+"_max"][0:imax]*1e-4)
        
    
