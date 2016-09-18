#! /usr/bin/env python
from sherpa.astro.ui import *
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
import astropy.units as u
from IPython.core.display import Image
from gammapy.image import SkyImageCollection, SkyImage
from gammapy.utils.energy import EnergyBounds, Energy
import astropy.units as u
import pylab as pt
from gammapy.background import fill_acceptance_image
from gammapy.utils.energy import EnergyBounds
from astropy.coordinates import Angle
from astropy.units import Quantity
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from method_fit import *
from matplotlib.backends.backend_pdf import PdfPages
from gammapy.detect import compute_ts_map
from astropy.convolution import Gaussian2DKernel
import yaml
import sys

pt.ion()

"""
./estimation_sourceflux.py "config_crab.yaml"
Estimation du flux du source model a partir de la psf et de l exposure: on=bkg+psf(model*exposure)
"""

input_param = yaml.load(open(sys.argv[1]))
# Input param fit and source configuration
# Sur quelle taille de la carte on fait le fit
freeze_bkg = input_param["param_fit_morpho"]["freeze_bkg"]
source_name = input_param["general"]["source_name"]
name_method_fond = input_param["general"]["name_method_fond"]
if freeze_bkg:
    name = "_bkg_fix"
else:
    name = "_bkg_free"
# Energy binning
energy_bins = EnergyBounds.equal_log_spacing(input_param["energy binning"]["Emin"],
                                             input_param["energy binning"]["Emax"],
                                             input_param["energy binning"]["nbin"], 'TeV')
energy_centers = energy_bins.log_centers

# outdir data and result
outdir_data = make_outdir_data(source_name, name_method_fond, len(energy_bins))
outdir_result = make_outdir_filesresult(source_name, name_method_fond, len(energy_bins))
outdir_plot = make_outdir_plot(source_name, name_method_fond, len(energy_bins))
outdir_profiles = make_outdir_profile(source_name, name_method_fond, len(energy_bins))

# Pour pouvoir definir la gaussienne centre sur la source au centre des cartes en general
E1 = energy_bins[0].value
E2 = energy_bins[1].value
on = SkyImageCollection.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["counts"]

if "dec" in input_param["general"]["sourde_name_skycoord"]:
    source_center = SkyCoord(input_param["general"]["sourde_name_skycoord"]["ra"],
                             input_param["general"]["sourde_name_skycoord"]["dec"], unit="deg")
else:
    source_center = SkyCoord.from_name(input_param["general"]["sourde_name_skycoord"])

param_fit = input_param["param_fit_morpho"]
if param_fit["gauss_SgrA"]["fit"]:
    name += "_SgrA"
if param_fit["gauss_G0p9"]["fit"]:
    name += "_G0p9"
# Si on inverse LS et CS alors c est qu il y a les deux!
if param_fit["invert_CS_LS"]:
    name += "_CS__LS"
else:
    if param_fit["Large scale"]["fit"]:
        name += "_LS"
    if param_fit["Gauss_to_CS"]["fit"]:
        name += "_CS"
if param_fit["central_gauss"]["fit"]:
    name += "_central_gauss"
if param_fit["arc source"]["fit"]:
    name += "_arcsource"
if param_fit["SgrB2"]["fit"]:
    name += "_SgrB2"
for i_E, E in enumerate(energy_bins[0:-1]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E + 1].value
    energy_band = Energy([E1 , E2], energy_bins.unit)
    print "energy band: ", E1, " TeV- ", E2, "TeV"
    # load Data
    on = SkyImageCollection.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")["counts"]
    on.write(outdir_data + "/on_maps" + str(E1) + "_" + str(E2) + "_TeV.fits", clobber=True)
    data = fits.open(outdir_data + "/on_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")
    load_image(1, data)
    # load exposure model
    exposure = make_exposure_model(outdir_data, E1, E2)
    # load bkg model
    bkg = make_bkg_model(outdir_data, E1, E2, freeze_bkg)
    # load psf model
    psf_SgrA = make_psf_model(outdir_data, E1, E2, on, "GC")
    psf_G0p9 = make_psf_model(outdir_data, E1, E2, on, "G0p9")
    # load CS model
    CS = make_CS_model(outdir_data, on, None, param_fit["CS"]["ampl_frozen"],
                       param_fit["CS"]["threshold_map"])
    # modele gauss pour sgrA centre sur SgrA
    source_center_SgrA = SkyCoord.from_name(input_param["param_SgrA"]["sourde_name_skycoord"])
    xpos_SgrA, ypos_SgrA = skycoord_to_pixel(source_center_SgrA, on.wcs)
    xpos_GC, ypos_GC = skycoord_to_pixel(source_center_SgrA, on.wcs)
    xpos_SgrA += 0.5
    ypos_SgrA += 0.5
    mygaus_SgrA = source_punctual_model(param_fit["gauss_SgrA"]["name"], param_fit["gauss_SgrA"]["fwhm_init"],
                                        param_fit["gauss_SgrA"]["fwhm_frozen"], None,
                                        param_fit["gauss_SgrA"]["ampl_frozen"], xpos_SgrA,
                                        param_fit["gauss_SgrA"]["xpos_frozen"],
                                        ypos_SgrA, param_fit["gauss_SgrA"]["ypos_frozen"])
    # modele gauss pour G0p9 centre sur G0p9
    source_center_G0p9 = SkyCoord(input_param["param_G0p9"]["sourde_name_skycoord"]["l_gal"],
                                  input_param["param_G0p9"]["sourde_name_skycoord"]["b_gal"], unit='deg',
                                  frame="galactic")
    xpos_G0p9, ypos_G0p9 = skycoord_to_pixel(source_center_G0p9, on.wcs)
    xpos_G0p9 += 0.5
    ypos_G0p9 += 0.5
    mygaus_G0p9 = source_punctual_model(param_fit["gauss_G0p9"]["name"], param_fit["gauss_G0p9"]["fwhm_init"],
                                        param_fit["gauss_G0p9"]["fwhm_frozen"], None,
                                        param_fit["gauss_G0p9"]["ampl_frozen"], xpos_G0p9,
                                        param_fit["gauss_G0p9"]["xpos_frozen"],
                                        ypos_G0p9, param_fit["gauss_G0p9"]["ypos_frozen"])

    # modele asymetric large scale gauss centre sur SgrA
    Large_Scale = source_NormGauss2D(param_fit["Large scale"]["name"], None,
                              param_fit["Large scale"]["fwhm_frozen"], None,
                              param_fit["Large scale"]["ampl_frozen"], xpos_GC,
                              param_fit["Large scale"]["xpos_frozen"],
                              ypos_GC, param_fit["Large scale"]["ypos_frozen"],ellep_fit=True,
                              ellep_init=param_fit["Large scale"]["ellip_init"],
                              ellep_frozen=param_fit["Large scale"]["ellip_frozen"])

    # Modele large gaussienne  multiplie avec CS centre sur SgrA
    gaus_CS = source_Gauss2D(param_fit["Gauss_to_CS"]["name"], None,
                          param_fit["Gauss_to_CS"]["fwhm_frozen"], param_fit["Gauss_to_CS"]["ampl_init"],
                          param_fit["Gauss_to_CS"]["ampl_frozen"], xpos_GC, param_fit["Gauss_to_CS"]["xpos_frozen"],
                          ypos_GC, param_fit["Gauss_to_CS"]["ypos_frozen"])

    # Modele symetric central gauss centre sur SgrA
    central_gauss = source_NormGauss2D(param_fit["central_gauss"]["name"], None,
                                param_fit["central_gauss"]["fwhm_frozen"], None,
                                param_fit["central_gauss"]["ampl_frozen"], xpos_GC,
                                param_fit["central_gauss"]["xpos_frozen"],
                                ypos_GC, param_fit["central_gauss"]["ypos_frozen"])
    """
    central_gauss = NormGauss2D("central_gauss")
    central_gauss.xpos, central_gauss.ypos = skycoord_to_pixel(source_center_SgrA, on.wcs)
    freeze(central_gauss.xpos)
    freeze(central_gauss.ypos)
    set_par(central_gauss.ampl, val=None, min=0, max=None, frozen=None)
    """
    #Arc_source
    source_center_arcsource = SkyCoord(param_fit["arc source"]["l"],
                       param_fit["arc source"]["b"], unit='deg', frame="galactic")
    xpos_arcsource, ypos_arcsource = skycoord_to_pixel(source_center_arcsource, on.wcs)
    arc_source=source_NormGauss2D(param_fit["arc source"]["name"], param_fit["arc source"]["fwhm_init"],
                                       param_fit["arc source"]["fwhm_frozen"], None,
                                       param_fit["arc source"]["ampl_frozen"], xpos_arcsource, param_fit["arc source"]["xpos_frozen"],
                          ypos_arcsource, param_fit["arc source"]["ypos_frozen"])
    #Gauss SgrB2
    source_center_sgrB2 = SkyCoord(param_fit["SgrB2"]["l"],
                       param_fit["SgrB2"]["b"], unit='deg', frame="galactic")
    xpos_sgrB2, ypos_sgrB2 = skycoord_to_pixel(source_center_sgrB2, on.wcs)
    sgrB2=source_NormGauss2D(param_fit["SgrB2"]["name"], param_fit["SgrB2"]["fwhm_init"],
                                       param_fit["SgrB2"]["fwhm_frozen"], None,
                                       param_fit["SgrB2"]["ampl_frozen"], xpos_sgrB2, param_fit["SgrB2"]["xpos_frozen"],
                          ypos_sgrB2, param_fit["SgrB2"]["ypos_frozen"])
    #region of inerest
    pix_deg = on.to_image_hdu().header["CDELT2"]
    lat=1.6/ pix_deg#Pour aller a plus et -0.8 as did Anne
    lon=4 / pix_deg#Pour aller a plus ou moins 2deg as did Anne
    x_pix_SgrA=skycoord_to_pixel(source_center_SgrA, on.wcs)[0]
    y_pix_SgrA=skycoord_to_pixel(source_center_SgrA, on.wcs)[1]
    name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(lon) + "," + str(lat) +")"
    #name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(150) + "," + str(50) +")"
    #name_interest = "box(" + str(x_pix_SgrA) + "," + str(y_pix_SgrA) + "," + str(10) + "," + str(10) +")"
    notice2d(name_interest)

    #ignore region in a box that mask J1734-303
    source_J1745_303 = SkyCoord(358.76, -0.6, unit='deg', frame="galactic")
    source_J1745_303_xpix, source_J1745_303_ypix = skycoord_to_pixel(source_J1745_303, on.wcs)
    width=100
    height=80
    name_region = "box(" + str(source_J1745_303_xpix+20) + "," + str(source_J1745_303_ypix-20) + "," + str(width) + "," + str(height) +")"
    ignore2d(name_region)

    set_stat("cstat")
    set_method("neldermead")

    list_src = [psf_SgrA(mygaus_SgrA)]
    if param_fit["gauss_G0p9"]["fit"]:
        list_src.append(psf_G0p9(mygaus_G0p9))
    # Si on inverse LS et CS alors c est qu il y a les deux!
    if param_fit["invert_CS_LS"]:
        list_src.append(psf_SgrA(gaus_CS * CS))
        list_src.append(psf_SgrA(Large_Scale))
    else:
        if param_fit["Large scale"]["fit"]:
            list_src.append(psf_SgrA(Large_Scale))
        if param_fit["Gauss_to_CS"]["fit"]:
            list_src.append(psf_SgrA(gaus_CS * CS))
    if param_fit["central_gauss"]["fit"]:
        list_src.append(psf_SgrA(central_gauss))
    if param_fit["arc source"]["fit"]:
        list_src.append(psf_SgrA(arc_source))
    if param_fit["SgrB2"]["fit"]:
        list_src.append(psf_SgrA(sgrB2))

    model = bkg
    set_full_model(model)
    pdf_lat=PdfPages(outdir_profiles+"/profiles_lattitude_"+name+"_" + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.pdf")
    pdf_lon=PdfPages(outdir_profiles+"/profiles_longitude_"+name+"_" + str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.pdf")
    for i_src, src in enumerate(list_src):
        model += src
        set_full_model(model)
        fit()
        result = get_fit_results()
        if i_src==0:
            table_models = result_table_CG(result, int(i_src))
        else:
            table_models = join(table_models.filled(-1000), result_table_CG(result, int(i_src)), join_type='outer')
        covar()
        covar_res = get_covar_results()
        # conf()
        # covar_res= get_conf_results()
        if i_src==0:
            table_covar = covar_table_CG(covar_res, int(i_src))
        else:
            table_covar = join(table_covar.filled(0), covar_table_CG(covar_res, int(i_src)), join_type='outer')

        save_resid(outdir_result + "/residual_morpho_step_" + str(i_src) + "_"+ name + "_"  + str("%.2f" % E1) + "_" + str(
            "%.2f" % E2) + "_TeV.fits", clobber=True)
        # import IPython; IPython.embed()
        # Profil lattitude et longitude
        shape = np.shape(on.data)
        mask = get_data().mask.reshape(shape)
        map_data=SkyImage.empty_like(on)
        model_map =SkyImage.empty_like(on)
        exp_map=SkyImage.empty_like(on)
        map_data.data = get_data().y.reshape(shape) * mask
        model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask
        exp_map.data= np.ones(map_data.data.shape)* mask

        resid = map_data.data - model_map.data
        coord = on.coordinates()
        # Longitude profile
        i_b = np.where((coord.b[:, 0] < on.center.b + 0.15 * u.deg) & (coord.b[:, 0] > on.center.b - 0.15 * u.deg))[0]
        npix_l = np.sum(np.flipud(mask[i_b, :]), axis=0)
        profile_l_model = np.sum(model_map.data[i_b, :], axis=0) / npix_l
        profile_l_on = np.sum(map_data.data[i_b, :], axis=0) / npix_l
        # Ca donne des coups par arcmin2 car on prend en compte qu on ne cumula pas le meme nombre de pixel pour chaque
        # longitude vu qu il y a des regions d exclusions
        profile_l_resid = np.sum(resid[i_b, :], axis=0) / npix_l
        err_l = np.sqrt(profile_l_on / npix_l)
        l = coord.l[0, :]
        l.value[np.where(l > 180 * u.deg)] = l.value[np.where(l > 180 * u.deg)] - 360
        resid_l_rebin, l_rebin, err_l_rebin = rebin_profile(profile_l_resid, l, err_l, nrebin=3)

        # Latitude profile
        l_center = on.center.l
        if l_center > 180 * u.deg:
            l_center = l_center - 360 * u.deg
        i_l = np.where((l < l_center + 1.5 * u.deg) & (l > l_center - 1.5 * u.deg))[0]
        npix_b = np.sum(np.flipud(mask[:, i_l]), axis=1)
        profile_b_model = np.sum(model_map.data[:, i_l], axis=1) / npix_b
        profile_b_on = np.sum(map_data.data[:, i_l], axis=1) / npix_b
        profile_b_resid = np.sum(resid[:, i_l], axis=1) / npix_b
        err_b = np.sqrt(profile_b_on / npix_b)
        resid_b_rebin, b_rebin, err_b_rebin = rebin_profile(profile_b_resid, coord.b[:, 0], err_b, nrebin=3)

        fig = pt.figure()
        ax = fig.add_subplot(2, 1, 1)
        pt.plot(l.value, profile_l_model, label="model")
        pt.plot(l.value, profile_l_on, label="on data")
        pt.xlim(-1.5, 1.5)
        pt.gca().invert_xaxis()
        pt.legend()
        ax = fig.add_subplot(2, 1, 2)
        pt.errorbar(l_rebin.value, resid_l_rebin, yerr=err_l_rebin, linestyle='None', marker="o",
                    label="Step= " + str(i_src))
        pt.axhline(y=0, color='red', linewidth=2)
        pt.legend()
        pt.ylabel("residual")
        pt.xlabel("longitude (degrees)")
        pt.title("longitude profile")
        pt.xlim(-1.5, 1.5)
        pt.gca().invert_xaxis()
        pdf_lon.savefig()

        fig = pt.figure()
        ax = fig.add_subplot(2, 1, 1)
        pt.plot(coord.b[:, 0].value, profile_b_model, label="model")
        pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
        pt.xlim(-1, 1)
        pt.legend()
        ax = fig.add_subplot(2, 1, 2)
        pt.errorbar(b_rebin.value, resid_b_rebin, yerr=err_b_rebin, linestyle='None', marker="o",
                    label="Step= " + str(i_src))
        pt.axhline(y=0, color='red', linewidth=2)
        pt.legend()
        pt.ylabel("residual")
        pt.xlabel("latitude (degrees)")
        pt.title("latitude profile")
        pt.xlim(-1, 1)
        pdf_lat.savefig()

        E_center = EnergyBounds(energy_band).log_centers
        if E_center < 1 * u.TeV:
            pix = 5
        elif ((1 * u.TeV < E_center) & (E_center < 5 * u.TeV)):
            pix = 4
        else:
            pix = 2.5
        kernel = Gaussian2DKernel(pix)
        TS = compute_ts_map(map_data, model_map, exp_map, kernel)
        TS.write(outdir_plot+"/TS_map_step_" + str(i_src) + "_" +name+"_"+ str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits",
                 clobber=True)
        sig = SkyImage.empty(TS["ts"])
        sig.data = np.sqrt(TS["ts"].data)
        sig.name = "sig"
        sig.write(
            outdir_plot+"/significance_map_step_" + str(i_src) + "_" +name+"_" +str("%.2f" % E1) + "_" + str("%.2f" % E2) + "_TeV.fits",
            clobber=True)
        if i_src==len(list_src)-1:
            model = bkg + psf_SgrA(mygaus_SgrA) + psf_G0p9(mygaus_G0p9)
            set_full_model(model)

            # Profil lattitude et longitude
            shape = np.shape(on.data)
            mask = get_data().mask.reshape(shape)
            map_data=SkyImage.empty_like(on)
            model_map =SkyImage.empty_like(on)
            map_data.data = get_data().y.reshape(shape) * mask
            model_map.data = get_model()(get_data().x0, get_data().x1).reshape(shape) * mask

            resid = map_data.data - model_map.data
            coord = on.coordinates()

            # Longitude profile
            i_b = np.where((coord.b[:, 0] < on.center.b + 0.15 * u.deg) & (coord.b[:, 0] > on.center.b - 0.15 * u.deg))[0]
            npix_l = np.sum(np.flipud(mask[i_b, :]), axis=0)
            profile_l_model = np.sum(model_map.data[i_b, :], axis=0) / npix_l
            profile_l_on = np.sum(map_data.data[i_b, :], axis=0) / npix_l
            # Ca donne des coups par arcmin2 car on prend en compte qu on ne cumula pas le meme nombre de pixel pour chaque
            # longitude vu qu il y a des regions d exclusions
            profile_l_resid = np.sum(resid[i_b, :], axis=0) / npix_l
            err_l = np.sqrt(profile_l_on / npix_l)
            l = coord.l[0, :]
            l.value[np.where(l > 180 * u.deg)] = l.value[np.where(l > 180 * u.deg)] - 360
            resid_l_rebin, l_rebin, err_l_rebin = rebin_profile(profile_l_resid, l, err_l, nrebin=3)

            # Latitude profile
            l_center = on.center.l
            if l_center > 180 * u.deg:
                l_center = l_center - 360 * u.deg
            i_l = np.where((l < l_center + 1.5 * u.deg) & (l > l_center - 1.5 * u.deg))[0]
            npix_b = np.sum(np.flipud(mask[:, i_l]), axis=1)
            profile_b_model = np.sum(model_map.data[:, i_l], axis=1) / npix_b
            profile_b_on = np.sum(map_data.data[:, i_l], axis=1) / npix_b
            profile_b_resid = np.sum(resid[:, i_l], axis=1) / npix_b
            err_b = np.sqrt(profile_b_on / npix_b)
            resid_b_rebin, b_rebin, err_b_rebin = rebin_profile(profile_b_resid, coord.b[:, 0], err_b, nrebin=3)

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(l.value, profile_l_model, label="model")
            pt.plot(l.value, profile_l_on, label="on data")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            pt.errorbar(l_rebin.value, resid_l_rebin, yerr=err_l_rebin, linestyle='None', marker="o",
                        label="Step= " + str(i_src))
            pt.axhline(y=0, color='red', linewidth=2)
            pt.legend()
            pt.ylabel("residual")
            pt.xlabel("longitude (degrees)")
            pt.title("longitude profile")
            pt.xlim(-1.5, 1.5)
            pt.gca().invert_xaxis()
            pdf_lon.savefig()

            fig = pt.figure()
            ax = fig.add_subplot(2, 1, 1)
            pt.plot(coord.b[:, 0].value, profile_b_model, label="model")
            pt.plot(coord.b[:, 0].value, profile_b_on, label="on data")
            pt.xlim(-1, 1)
            pt.legend()
            ax = fig.add_subplot(2, 1, 2)
            pt.errorbar(b_rebin.value, resid_b_rebin, yerr=err_b_rebin, linestyle='None', marker="o",
                        label="Step= " + str(i_src))
            pt.axhline(y=0, color='red', linewidth=2)
            pt.legend()
            pt.ylabel("residual")
            pt.xlabel("latitude (degrees)")
            pt.title("latitude profile")
            pt.xlim(-1, 1)
            pdf_lat.savefig()

    pdf_lon.close()
    pdf_lat.close()

    table_models = table_models.filled(-1000)
    table_covar = table_covar.filled(0)
    filename_table_result = outdir_result + "/morphology_fit_result_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table_models.write(filename_table_result, format="ascii")
    filename_covar_result = outdir_result + "/morphology_fit_covar_" + name + "_" + str("%.2f" % E1) + "_" + str(
        "%.2f" % E2) + "_TeV.txt"
    table_covar.write(filename_covar_result, format="ascii")
    table_models = Table()
    table_covar = Table()
