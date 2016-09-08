from astropy.table import Table
import numpy as np
import pylab as pt
from matplotlib.backends.backend_pdf import PdfPages
from gammapy.utils.energy import EnergyBounds
pt.ion()



def plot_spectra_source(pdf, E, flux, err_inf, err_sup, Eth, fit_HESS):
    """

    Parameters
    ----------
    pdf: PdfPages
        to store on the same pdf
    E: array
        energy array of the bins used to estimate the flux
    flux: array
        array of the source flux for the different energy bin
    err_inf: array
        array of the err_min on the source flux for the different energy bin
    err_sup: array
        array of the err_max on the source flux for the different energy bin
    Eth: array
        energy theoric used to plit the spectrum fitted with HESS, more bin than in E genrally but same boundaries
    fit_HESS: list
        containt the parameters of the law fit with HESS for the source

    Returns
    -------

    """
    pt.figure()
    pt.loglog(E, flux*E**2, "o", label="fit sherpa", color="blue")
    pt.errorbar(E, flux*E**2, yerr=[-err_inf*E**2, err_sup*E**2], label=None,color="blue", linestyle="None")
    color=["cyan", "red", "green"]
    for i,name_law in enumerate(fit_HESS["law"]):
        pt.loglog(Eth, make_law(Eth,fit_HESS[name_law])*Eth**2, label="H.E.S.S.: "+name_law, color=color[i])
    pt.ylabel("flux*E2 (Tev cm-2 s-1)")
    pt.xlabel("Energy (TeV)")
    pt.legend()
    pdf.savefig()

def plot_bkg_norm(pdf, E, bkg_norm, err_inf, err_sup):
    """

    Parameters
    ----------
    pdf: PdfPages
        to store on the same pdf
    E: array
        energy array of the bins used to estimate the flux
    bkg_norm: array
        array of the bkg norm for the different energy bin
    err_inf: array
        array of the err_min on the bkg norm for the different energy bin
    err_sup: array
        array of the err_max on the bkg norm for the different energy bin
    Returns
    -------

    """
    pt.figure()
    pt.semilogx(E, bkg_norm, "o", color="blue")
    pt.errorbar(E, bkg_norm, yerr=[err_inf, err_sup], color="blue", linestyle="None")
    pt.xlabel("Energy (TeV)")
    pt.ylabel("bkg ampl")
    pt.title("bgk norm fitte pour different bbins en energie")
    pdf.savefig()

def plot_flux_component(pdf, E, name, component_flux, err_inf, err_sup):
    """

    Parameters
    ----------
    pdf: PdfPages
        to store on the same pdf
    E: array
        energy array of the bins used to estimate the flux
    component_flux: array
        array of the flux of the additional component for the different energy bin
    err_inf: array
        array of the err_min on the additional component  for the different energy bin
    err_sup: array: array
        array of the err_max on the additional component  for the different energy bin

    Returns
    -------

    """
    pt.figure()
    pt.loglog(E, component_flux*E**2, "o", color="blue")
    pt.errorbar(E, component_flux*E**2, yerr=[err_inf*E**2, err_sup*E**2], color="blue", linestyle="None")
    pt.xlabel("Energy (TeV)")
    pt.ylabel(name)
    pt.title(name+" flux fitte pour different bbins en energie")
    pdf.savefig()

def plot_param(pdf, E, name, param, err_inf, err_sup):
    """

    Parameters
    ----------
    pdf: PdfPages
        to store on the same pdf
    E: array
        energy array of the bins used to estimate the flux
    param: array
        array of the value of the param of a component for the different energy bin
    err_inf: array
        array of the err_min on the param of a component  for the different energy bin
    err_sup: array: array
        array of the err_max on the param of a component  for the different energy bin

    Returns
    -------

    """
    pt.figure()
    pt.semilogx(E, param, "o", color="blue")
    pt.errorbar(E, param, yerr=[err_inf, err_sup], color="blue", linestyle="None")
    pt.xlabel("Energy (TeV)")
    pt.ylabel(name)
    pt.title(name+" fitte pour different bbins en energie")
    pdf.savefig()
    
def make_law(E,law_param):
    """

    Parameters
    ----------
    E: array
        array of energy bins to plot the spectrum fitted with HESS
    law_param: list
        list of the parameter needed to plot the law fitted by HESS

    Returns
    -------

    """
    if len(law_param)==2:
        law=law_param["phi0"]*E**(-law_param["gamma"])
    elif len(law_param)==3:
        law=law_param["phi0"]*E**(-law_param["gamma"])*np.exp(-E/law_param["E_c"])
    elif len(law_param)==4:
        spectre=np.zeros(len(E))
        i1=np.where(E<law_param["E_b"])
        i2=np.where(E>=law_param["E_b"])
        spectre[i1]=law_param["phi0"]*E**(-law_param["gamma1"])
        spectre[i2]=law_param["phi0"]*E**(-law_param["gamma2"])*law_param["E_b"]**(law_param["gamma2"]-law_param["gamma1"])
        law=spectre

    return law

