import sys
import os

# from TEMPy.ShowPlot import Plot
# """
from TEMPy.mapprocess import mapcompare
from TEMPy.mapprocess import Filter
from TEMPy.MapParser import MapParser
from TEMPy.EMMap import Map

"""
from TEMPy.map_process import mapcompare
from TEMPy.map_process import Filter
from TEMPy.maps.map_parser import MapParser
from TEMPy.maps.em_map import Map

"""
# from TEMPy.map_process import MapEdit
from copy import deepcopy

# import mrcfile
import numpy as np

# from ccpem_core.tasks.tempy.difference_map.difference_map import write_mapfile
mrcfile_import = True
try:
    import mrcfile
except ImportError:
    mrfile_import = False


def readmap(mapin):
    """try:
        mrcobject = mrcfile.open(mapin, mode='r')
        if np.any(np.isnan(mrcobject.data)):
            sys.exit('Map has NaN values: {}'.format(mapin))
        emmap = Filter(mrcobject)
        print('no valuerror {0}'.format(emmap.apix))
    except ValueError:"""
    mrcobject = MapParser.readMRC(mapin)
    emmap = Filter(mrcobject)
    # print(emmap.apix)
    emmap.set_apix_as_tuple()
    # print(emmap.apix)

    emmap.fix_origin()
    return emmap


def write_mapfile(mapobj, map_path):
    # TEMPy map
    if not mrcfile_import and mapobj.__class__.__name__ == "Map":
        mapobj.write_to_MRC_file(map_path)
    # mrcfile map
    elif mrcfile_import:
        newmrcobj = mrcfile.new(map_path, overwrite=True)
        mapobj.set_newmap_data_header(newmrcobj)
        newmrcobj.close()
    # tempy mapprocess map
    else:
        newmrcobj = Map(
            np.zeros(mapobj.fullMap.shape), list(mapobj.origin), mapobj.apix, "mapname"
        )
        mapobj.set_newmap_data_header(newmrcobj)
        newmrcobj.update_header()
        newmrcobj.write_to_MRC_file(map_path)


def get_diffmap12(
    emmap1,
    emmap2,
    res1,
    res2,
    plot_spectra=False,
    debug=False,
    lpfiltb=False,
    refsc=False,
    randsize=0.1,
    flag_dust=False,
    cyc=0,
    verbose=0,
):
    if emmap2.__class__.__name__ == "Map":
        print("filtermap")
        # emmap2 = MapEdit(emmap2)
        emmap2 = Filter(emmap2)
        emmap2.set_apix_as_tuple()
        # emmap2.fix_origin()
    elif emmap2.__class__.__name__ == "Filter":
        if not isinstance(emmap1.apix, tuple):
            emmap2.set_apix_as_tuple()
    print(emmap1.apix)
    if emmap1.__class__.__name__ == "Map":
        print("filtermap")
        emmap1 = Filter(emmap1)
        emmap1.set_apix_as_tuple()
        # emmap1.fix_origin()
    elif emmap1.__class__.__name__ == "Filter":
        if not isinstance(emmap1.apix, tuple):
            emmap1.set_apix_as_tuple()
    print(f"{emmap1.apix}, {emmap2.apix}")
    # rite_mapfile(emmap2, '{0}/pdbin_emmap2.map'.format(outdir))
    # print('in map scaling, {0}'.format(str(emmap2.fullMap.dtype)))
    if res1 > 4.0:
        t1 = 2.0
    else:
        t1 = 2.5
    if res2 > 4.0:
        t2 = 2.0
    else:
        t2 = 2.5
    c1 = emmap1.calculate_map_contour(sigma_factor=t1)
    c2 = emmap2.calculate_map_contour(sigma_factor=t2)
    # print(emmap1.box_size(), emmap1.apix, c1)
    # print(emmap2.box_size(), emmap2.apix, c2)

    # mapin2 = "/home/swh514/Projects/testing_ground/test_sf_mapgridpositions_gridtree_rad2p5_11_mapgridtocoord.map"
    # emmap2 = readmap(mapin2)
    # input resolution of both maps

    # low pass filter flag
    # flag_filt = False
    # use second map (model map ) as reference
    # refsc = False
    # dust filter after difference
    # randsize = 0.1
    # flag_dust = False
    # calculate contour
    # check grid dimension
    samegrid = False
    try:
        print("comparing maps.")
        # print(emmap1.fullMap.shape, emmap2.fullMap.shape)
        # print(emmap1.origin, emmap2.origin)
        mapcompare.compare_grid(emmap1, emmap2)
        # print("Map dimension are the same.")
        samegrid = True
    except AssertionError:
        samegrid = False

    # check spacing along all axes
    if emmap1.apix[0] != emmap1.apix[1] or emmap1.apix[1] != emmap1.apix[2]:
        samegrid = False
    if emmap2.apix[0] != emmap2.apix[1] or emmap2.apix[1] != emmap2.apix[2]:
        samegrid = False

    if debug:
        print(str(emmap1.fullMap.dtype), str(emmap2.fullMap.dtype))

    # global scaling of amplitudes
    # assuming same grid
    if samegrid:
        diff1, diff2, dict_plot = mapcompare.amplitude_match(
            emmap1, emmap2, reso=max(res1, res2), lpfiltb=lpfiltb, ref=refsc
        )

        # if plot_spectra:
        #    pl = Plot()
        #    pl.lineplot(dict_plot, 'spectra.png')

        # min of minimas of two scaled maps
        min1 = diff1.min()
        min2 = diff2.min()

        min_scaled_maps = min(min1, min2)
        # shift to posittive values
        if min_scaled_maps < 0.0:
            # make values non zero
            min_scaled_maps = min_scaled_maps + 0.05 * min_scaled_maps
            diff1.shift_density(-min_scaled_maps, inplace=True)
            diff2.shift_density(-min_scaled_maps, inplace=True)
        print("calculating difference")

        # store scaled map1
        scaledmap1 = diff1.copy(deep=True)
        # store scaled map2
        scaledmap2 = diff2.copy(deep=True)
        # calculate difference map 1
        diff1.fullMap = diff1.fullMap - diff2.fullMap

        # dust filter
        if flag_dust:
            # mask1 = scaledmap1.copy(deep=True)
            # print("Using threshold {} for fractional difference".format(0.3))
            # mask1.fullMap[:] = mask1.fullMap > 0.3
            print("Dusting the differences")
            diff1.remove_dust_by_size(0.0, prob=randsize, inplace=True)
            diff1.apply_mask(diff1.fullMap > 0.0, inplace=True)
            diff2.remove_dust_by_size(0.0, prob=randsize, inplace=True)
            diff2.apply_mask(diff2.fullMap > 0.0, inplace=True)

        # write difference maps
        diff1ingrid = diff1.copy(deep=False)
        diff2ingrid = diff2.copy(deep=False)
        if verbose >= 6:
            mapout1 = "test_diff1_{0}.map".format(cyc)  # .format(outdir)
            write_mapfile(diff1ingrid, mapout1)

        # mapout2 = 'test_diff1_2.map' #.format(outdir)
        # write_mapfile(diff2ingrid, mapout2)
        return scaledmap1, scaledmap2, diff1ingrid
    else:
        print("different spacing")


"""
def get_diffmap(mapin1, pdbin, res1, res2, plot=False, debug=False):
    emmap1 = readmap(mapin1)
    inmap1 = MapParser.readMRC(mapin1)
    emmap2 = esf.main(inmap1, pdbin)
    #write_mapfile(emmap2, '{0}/pdbin_emmap2.map'.format(outdir))
    
    if emmap2.__class__.__name__ == 'Map':
        print('filtermap')
        emmap2 = Filter(emmap2)
        emmap2.set_apix_as_tuple()
        emmap2.fix_origin()

    write_mapfile(emmap2, '{0}/pdbin_emmap2.map'.format(outdir))
    print('in map scaling, {0}'.format(str(emmap2.fullMap.dtype)))
    c1 = emmap1.calculate_map_contour(sigma_factor=2.0)
    c2 = emmap2.calculate_map_contour(sigma_factor=2.0)
    print(emmap1.box_size(), emmap1.apix, c1)
    print(emmap2.box_size(), emmap2.apix, c2)
    
   

    #mapin2 = "/home/swh514/Projects/testing_ground/test_sf_mapgridpositions_gridtree_rad2p5_11_mapgridtocoord.map"
    #emmap2 = readmap(mapin2)
    # input resolution of both maps
    r1 = 3.2
    r2 = 3.2
    # low pass filter flag
    flag_filt = False
    # use second map (model map ) as reference
    refsc = False
    # dust filter after difference
    randsize = 0.1
    flag_dust = False
    # calculate contour
    # check grid dimension
    samegrid = False
    try:
        mapcompare.compare_grid(emmap1,emmap2)
        print("Map dimension are the same.")
        samegrid = True
    except AssertionError: samegrid = False

    # check spacing along all axes
    if emmap1.apix[0] != emmap1.apix[1] or emmap1.apix[1] != emmap1.apix[2]:
        samegrid = False
    if emmap2.apix[0] != emmap2.apix[1] or emmap2.apix[1] != emmap2.apix[2]:
        samegrid = False

    if debug:
        print(str(emmap1.fullMap.dtype), str(emmap2.fullMap.dtype))

    #global scaling of amplitudes
    #assuming same grid
    if samegrid:
        diff1, diff2, dict_plot = mapcompare.amplitude_match(emmap1, emmap2,
                                reso=max(r1,r2), lpfiltb=flag_filt, ref=refsc)
    
        if plot_spectra:
            pl = Plot()
            pl.lineplot(dict_plot, 'spectra.png')
        
        #min of minimas of two scaled maps
        min1 = diff1.min()
        min2 = diff2.min()

        min_scaled_maps = min(min1, min2)
        # shift to posittive values
        if (min_scaled_maps < 0.):
            # make values non zero
            min_scaled_maps = min_scaled_maps + 0.05*min_scaled_maps
            diff1.shift_density(-min_scaled_maps, inplace=True)
            diff2.shift_density(-min_scaled_maps, inplace=True)
        print("calculating difference")

        # store scaled map1
        scaledmap1 = diff1.copy(deep=True)
        #calculate difference map 1
        diff1.fullMap = (diff1.fullMap - diff2.fullMap)

        #dust filter
        if flag_dust:
            #mask1 = scaledmap1.copy(deep=True)
            #print("Using threshold {} for fractional difference".format(0.3))
            #mask1.fullMap[:] = mask1.fullMap > 0.3
            print("Dusting the differences")
            diff1.remove_dust_by_size(0.0, prob=randsize, inplace=True)
            diff1.apply_mask(diff1.fullMap>0.0,inplace=True)
            diff2.remove_dust_by_size(0.0, prob=randsize, inplace=True)
            diff2.apply_mask(diff2.fullMap>0.0,inplace=True)
            
        #write difference maps
        diff1ingrid = diff1.copy(deep=False)
        diff2ingrid = diff2.copy(deep=False)
        #mapout1 = '{0}/test_diff1_1.map'.format(outdir)
        #mapout2 = '{0}/test_diff1_2.map'.format(outdir)
        #write_mapfile(diff1ingrid, mapout1)
        #write_mapfile(diff2ingrid, mapout2)
        return diff1ingrid
    else:
        print("different spacing")

"""


if __name__ == "__main__":
    from TEMPy.StructureParser import PDBParser
    from TEMPy.mapprocess import array_utils

    # sys.path.append("/home/swh514/Projects/testing_ground")
    # import electron_scattering_factor as esf
    # import esf_map_calc as esf
    # outdir = '/home/swh514/Projects/testing_ground/scale_map/test4'
    mapin1 = "/home/swh514/Projects/data/EMD-3488/map/emd_3488.map"
    mapin1 = "/home/swh514/Projects/sheetbend_python_git/AF_mr_refine_kakashi/AF-Q86YT5-F1-model_v1/emd_22457.map"
    pdbin = "/home/swh514/Projects/sheetbend_python_git/AF_mr_refine_kakashi/AF-Q86YT5-F1-model_v1/FULL/molrep.pdb"
    res1 = 6.0
    res2 = 6.0
    cyc = 2
    plot_spectra = True
    debug = True
    # emmap1 = readmap(mapin1)
    inmap1 = MapParser.readMRC(mapin1)
    structure = PDBParser.read_PDB_file("test", pdbin, hetatm=False, water=False)
    emmap2 = inmap1.copy()
    emmap2.fullMap = emmap2.fullMap * 0
    emmap2 = structure.calculate_rho(2.5, inmap1, emmap2)
    # emmap2 = esf.main(inmap1, pdbin)
    # write_mapfile(emmap2, '{0}/pdbin_emmap2.map'.format(outdir))
    fltrmap = Filter(inmap1)
    # frequency from 0:0.5, 0.1 =10Angs, 0.5 = 2Angs ? or apix dependent?
    # 1/Angs = freq or apix/reso = freq?
    print(f"filtermap dtype : {fltrmap.fullMap.dtype}")
    ftfilter = array_utils.tanh_lowpass(
        fltrmap.fullMap.shape, inmap1.apix / res1, fall=0.5
    )
    lp_map = fltrmap.fourier_filter(ftfilter=ftfilter, inplace=False)
    lp_map.set_apix_tempy()
    fltr_cmap = Filter(emmap2)
    ftfilter = array_utils.tanh_lowpass(
        fltr_cmap.fullMap.shape, inmap1.apix / res1, fall=0.5
    )
    lp_cmap = fltr_cmap.fourier_filter(ftfilter=ftfilter, inplace=False)

    diffmap = get_diffmap12(
        inmap1,
        emmap2,
        res1,
        res2,
        plot_spectra=False,
        debug=False,
        flag_filt=False,
        refsc=False,
        randsize=0.1,
        flag_dust=False,
        cyc=cyc,
    )
    mapout1 = "test_diff1_{0}.map".format(cyc)  # .format(outdir)
    write_mapfile(diffmap, mapout1)
    """if emmap2.__class__.__name__ == 'Map':
        print('filtermap')
        emmap2 = Filter(emmap2)
        emmap2.set_apix_as_tuple()
        emmap2.fix_origin()

    write_mapfile(emmap2, '{0}/pdbin_emmap2.map'.format(outdir))
    print('in map scaling, {0}'.format(str(emmap2.fullMap.dtype)))
    c1 = emmap1.calculate_map_contour(sigma_factor=2.0)
    c2 = emmap2.calculate_map_contour(sigma_factor=2.0)
    print(emmap1.box_size(), emmap1.apix, c1)
    print(emmap2.box_size(), emmap2.apix, c2)
    
   

    #mapin2 = "/home/swh514/Projects/testing_ground/test_sf_mapgridpositions_gridtree_rad2p5_11_mapgridtocoord.map"
    #emmap2 = readmap(mapin2)
    # input resolution of both maps
    r1 = 3.2
    r2 = 3.2
    # low pass filter flag
    flag_filt = False
    # use second map (model map ) as reference
    refsc = False
    # dust filter after difference
    randsize = 0.1
    flag_dust = False
    # calculate contour
    # check grid dimension
    samegrid = False
    try:
        mapcompare.compare_grid(emmap1,emmap2)
        print("Map dimension are the same.")
        samegrid = True
    except AssertionError: samegrid = False

    # check spacing along all axes
    if emmap1.apix[0] != emmap1.apix[1] or emmap1.apix[1] != emmap1.apix[2]:
        samegrid = False
    if emmap2.apix[0] != emmap2.apix[1] or emmap2.apix[1] != emmap2.apix[2]:
        samegrid = False

    if debug:
        print(str(emmap1.fullMap.dtype), str(emmap2.fullMap.dtype))

    #global scaling of amplitudes
    #assuming same grid
    if samegrid :
        diff1, diff2, dict_plot = mapcompare.amplitude_match(emmap1, emmap2,
                                reso=max(r1,r2), lpfiltb=flag_filt, ref=refsc)
    
        #if plot_spectra:
        #    pl = Plot()
        #    pl.lineplot(dict_plot, 'spectra.png')
        
        #min of minimas of two scaled maps
        min1 = diff1.min()
        min2 = diff2.min()

        min_scaled_maps = min(min1, min2)
        # shift to posittive values
        if (min_scaled_maps < 0.):
            # make values non zero
            min_scaled_maps = min_scaled_maps + 0.05*min_scaled_maps
            diff1.shift_density(-min_scaled_maps, inplace=True)
            diff2.shift_density(-min_scaled_maps, inplace=True)
        print("calculating difference")

        # store scaled map1
        scaledmap1 = diff1.copy(deep=True)
        #calculate difference map 1
        diff1.fullMap = (diff1.fullMap - diff2.fullMap)

        #dust filter
        if flag_dust:
            #mask1 = scaledmap1.copy(deep=True)
            #print("Using threshold {} for fractional difference".format(0.3))
            #mask1.fullMap[:] = mask1.fullMap > 0.3
            print("Dusting the differences")
            diff1.remove_dust_by_size(0.0, prob=randsize, inplace=True)
            diff1.apply_mask(diff1.fullMap>0.0,inplace=True)
            diff2.remove_dust_by_size(0.0, prob=randsize, inplace=True)
            diff2.apply_mask(diff2.fullMap>0.0,inplace=True)
            
        #write difference maps
        diff1ingrid = diff1.copy(deep=False)
        diff2ingrid = diff2.copy(deep=False)
        #mapout1 = '{0}/test_diff1_1.map'.format(outdir)
        #mapout2 = '{0}/test_diff1_2.map'.format(outdir)
        #write_mapfile(diff1ingrid, mapout1)
        #write_mapfile(diff2ingrid, mapout2)
    else:
        print("different spacing")
    """
