from method_fit import *
import numpy as np
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from gammapy.image import SkyImageList, SkyMask
from gammapy.utils.energy import EnergyBounds
import pylab as pt
import math
import astropy.units as u
from scipy.optimize import curve_fit

pt.ion()
"""
script to plot for different images (source_name, model_fond and energy bins), the histogram of the significance of
the resulting images outside of the source to see if this is well centered on zero
"""

energy_bins = EnergyBounds.equal_log_spacing(0.5, 100, 5, 'TeV')
#energy_bins = EnergyBounds.equal_log_spacing(0.5, 40, 20, 'TeV')
# energy_bins = EnergyBounds.equal_log_spacing(0.3, 100, 5, 'TeV')
# energy_bins = EnergyBounds.equal_log_spacing(0.5, 30, 10, 'TeV')

name_source="Crab"
#name_source = "J1813"
#name_source = "RXJ1713"
# name_source = "G21p5"
# name_source = "2155"

#name_method_fond = "coszenbinning_zen_0_34_49_61_72_eff"
# name_method_fond = "coszenbinning_zen_0_27_39_49_57_65_72_15binE"
name_method_fond="coszenbinning_zen_0_34_49_61_72_sansLMC"

config_name="Mpp_Std"
outdir_data = make_outdir_data(name_source, name_method_fond, len(energy_bins), config_name)
outdir_plot = make_outdir_plot(name_source, name_method_fond, len(energy_bins), config_name)

exclusion_mask = SkyMask.read('tevcat_exclusion_radius_0p5.fits')
#exclusion_mask = SkyMask.read('exclusion_large.fits')
for i_E, E in enumerate(energy_bins[:-2]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E + 1].value
    significance_map = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")[
        "significance"]
    #coord=significance_map.coordinates()
    #center=significance_map.center
    #offset=center.separation(coord)
    #i=np.where(offset>Angle(2,"deg"))
    #significance_map.data[i]=-1000
    refheader = significance_map.to_image_hdu().header
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    pt.figure(i_E)
    n, bins, patches = histo_significance(significance_map, exclusion_mask)
    bin_center = (bins[1:] + bins[0:-1]) / 2
    popt, pcov = curve_fit(norm, bin_center, n)
    perr = np.sqrt(np.diag(pcov))
    pt.plot(bin_center, norm(bin_center, popt[0], popt[1], popt[2]), color="red", linewidth=3,
            label="mean= " + str("%.3f" % popt[1]) + "+/- " + str("%.3f" % perr[1]) + " and sigma= " + str(
                "%.3f" % popt[2]) + " +/- " + str("%.3f" % perr[2]))
    pt.legend()
    pt.title("Energy band: " + str("%.2f" % E1) + "-" + str("%.2f" % E2) + " TeV")
    pt.savefig(outdir_plot + "/histo_sigma_" + str("%.2f" % E1) + "-" + str("%.2f" % E2) + ".jpg")


for i_E, E in enumerate(energy_bins[:-2]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E + 1].value
    significance_map = SkyImageList.read(outdir_data + "/fov_bg_maps" + str(E1) + "_" + str(E2) + "_TeV.fits")[
        "significance"]
    refheader = significance_map.to_image_hdu().header
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    coord=significance_map.coordinates()
    center=significance_map.center
    offset=center.separation(coord)
    i=np.where(exclusion_mask.data==0)
    #i=np.where((exclusion_mask.data==0) | (offset>Angle(2,"deg")))
    significance_map.data[i]=0
    pt.figure(100+i_E)
    pt.imshow(significance_map)
    pt.colorbar()
    significance_map.write("significance_outside_exclusionregion_" + str("%.2f" % E1) + "-" + str("%.2f" % E2) + ".fits", clobber=True)

for i_E, E in enumerate(energy_bins[:-2]):
    E1 = energy_bins[i_E].value
    E2 = energy_bins[i_E + 1].value
    filename_bkg_norm = outdir_data + '/table_bkg_norm_' + str(E1) + '_' + str(E2) + '_TeV.fits'
    table=Table.read(filename_bkg_norm)
    pt.figure(1000+i_E)
    n, bins, patches = pt.hist(table["blg_norm"],bins=5)
    pt.legend()
    pt.title("Energy band: " + str("%.2f" % E1) + "-" + str("%.2f" % E2) + " TeV")
    pt.xlabel("Bkg norm")
    pt.savefig(outdir_plot + "/histo_bkg_normfond_" + str("%.2f" % E1) + "-" + str("%.2f" % E2) + ".jpg")
    
