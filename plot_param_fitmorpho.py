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
plot la valeur des differentes composantes utilisees pour le fit morpho
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

if input_param["param_fit"]["use_EM_model"]:
    name+="_emission_galactic_True"
#Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"], input_param["energy binning"]["Emax"], input_param["energy binning"]["nbin"], 'TeV')
energy_centers=energy_bins.log_centers

#outdir result and plot
config_name = input_param["general"]["config_name"]
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux)
outdir_plot = make_outdir_plot(source_name, name_method_fond, len(energy_bins),config_name,image_size,for_integral_flux)

#store the fit result for the model of the source
filename_table_result=outdir_result+"/morphology_fit_result"+name+".txt"
filename_covar_result=outdir_result+"/morphology_fit_covar_result"+name+".txt"
table_models= Table.read(filename_table_result, format="ascii")
table_covar=Table.read(filename_covar_result, format="ascii")

#imax: until which energy bin we want to plot
imax=input_param["param_plot"]["imax"]
E=table_models["energy"][0:imax]

pdf=PdfPages(outdir_plot+"/param_fitmorpho_"+name+".pdf")
#plot le count de la gaussienne
#Si jamais on fite d'autre composantif len(table_models.colnames)>2:
for name in table_models.colnames:
    if ((name=="energy") |(name=="dof") | (name=="statval")):
        continue
    else:
        plot_param(pdf,E,name,table_models[name][0:imax],table_covar[name+"_min"][0:imax], table_covar[name+"_max"][0:imax])
pdf.close()        
