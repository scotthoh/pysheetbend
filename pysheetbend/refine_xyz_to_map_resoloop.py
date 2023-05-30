"""
Python implementation of csheetbend to perform shift field refinement
Copyright 2018 Kevin Cowtan & University of York all rights reserved
Author: Soon Wen Hoh, University of York 2020

License: GNU LESSER GENERAL PUBLIC LICENSE v2.1
"""

from __future__ import print_function, absolute_import
from argparse import ArgumentError
import sys
from os import getcwd
from os.path import splitext, basename
import gemmi
import numpy as np
from pysheetbend.utils.dencalc import calculate_density_with_boxsize, calculate_density
from pysheetbend.shiftfield import shift_field_coord, shift_field_uiso
import pysheetbend.shiftfield_util as sf_util
import pysheetbend.pseudoregularizer_gemmi as pseudoregularizer
from pysheetbend.utils import fileio, map_funcs, cell

# from memory_profiler import profile


# @profile
def main(args):
    """
    Run main
    """

    # Set variables from parsed arguments
    ippdb = args.pdbin
    ipmap = args.mapin
    if ippdb is None or ipmap is None:
        raise ArgumentError(
            "Please provide both --pdbin and --mapin to refine model against "
            "a map.\n Program terminated...\n"
        )

    nomask = args.nomask
    ipmask = args.maskin
    oppdb = args.pdbout  # shiftfield.pdb
    res = args.res
    resbycyc = args.res_by_cyc  # None
    ncyc = args.cycle  # 1
    refxyz = args.refxyz  # False
    refuiso = args.refuiso  # False
    # postrefuiso = args.postrefuiso  # False
    pseudoreg = args.pseudoreg  # no, yes, or postref
    rad = args.rad  # -1.0
    radscl = args.radscl  # 4.0
    coor_tol = args.coor_tol  # 0.01
    output_intermediate = args.intermediate  # False
    verbose = args.verbose  # 0
    ncycrr = args.cycle_regularise  # refine-regularise-cycle, macro cycles
    function_type = "quadratic"  # quadratic function filter
    hetatm_present = False  # for file writeout in case no hetatm present
    biso_range = args.biso_range
    keep_ligand = not args.no_ligands
    keep_water = not args.no_water
    selection = args.selection
    alternative_clustering = False
    write_cubic_map = args.cubicmap
    # for profiling, if verbose >= 2, writes out time
    # for each profiled section at the end
    timelog = sf_util.Profile()
    pdbin_ext = None
    offset_origin = [0.0, 0.0, 0.0]
    # job_location = getcwd()
    # if both --coord and --usio are not specified
    if not refxyz and not refuiso:
        refxyz = True
        print("Setting --coord to True for coordinates refinement.")

    if resbycyc is None:
        if res > 0.0:
            resbycyc = [res]
        else:
            # need to change this python3 can use , end=''
            raise ArgumentError(
                "Please specify the resolution of the map "
                "or resolution by cycle.\n"
                "Program terminated ...\n"
            )
    elif res > 0.0:
        if res > resbycyc[-1]:
            resbycyc[-1] = res

    if refuiso:
        print(f"B-factor bounds : {biso_range[0]} < B < {biso_range[1]}")
    if oppdb is not None:
        pathfname = oppdb  # user specified file
        pdbin_ext = splitext(basename(ippdb))[1]

    else:
        pathfname = splitext(basename(ippdb))[0]
        pdbin_ext = splitext(basename(ippdb))[1]

    # initialise results list, will contain all results
    results = []
    # read map
    mapin, grid_info = fileio.read_map(ipmap)
    # read model
    structure, hetatm_present = fileio.get_structure(
        ippdb,
        keep_waters=keep_water,
        keep_hetatom=keep_ligand,
        verbose=verbose,
    )
    # check if map is cubic, make cubic map
    if (
        grid_info.grid_shape[0] != grid_info.grid_shape[1]
        or grid_info.grid_shape[1] != grid_info.grid_shape[2]
    ):
        mapin, grid_info = map_funcs.make_map_cubic(mapin, grid_info, write_map=write_cubic_map)

    fullMap = mapin.grid
    # TODO: check map box size and reduce if too big?
    apix0 = grid_info.voxel_size
    ori0 = grid_info.origin

    # set unit cell from map to structure if (1.0 1.0 1.0)
    if not structure.cell.is_crystal():
        structure.cell = fullMap.unit_cell
    else:
        sf_util.match_model_map_unitcell(structure, fullMap)

    # if origin not 0,0,0, offset structure in advance
    # and keep the offset
    if np.any(grid_info.origin):  # not zero
        print("Fix origin")
        offset_origin = grid_info.origin
        tr = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*-offset_origin))
        structure[0].transform_pos_and_adp(tr)
        grid_info.origin = np.array([0.0, 0.0, 0.0])
        structure.write_minimal_pdb("starting_input_originat0.pdb")
    # flag selections
    if selection:
        sf_util.mark_selected_residues(structure, selection)
    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)
    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)  # so that the grid size matches original
    else:
        samp_rate = 1.5  # default; CLIPPER's oversampling parameter
        res = 3.0 * min_d

    if pseudoreg != "no":
        print(f"Setting model for Pseudoregularization: {pseudoreg}")
        preg = pseudoregularizer.Pseudoregularize(structure.clone())

    # read in mask if provided as input
    if ipmask is not None:
        maskin, maskin_grid_info = fileio.read_map(ipmask)
    fft_obj0, ifft_obj0 = sf_util.plan_fft_ifft(
        gridinfo=grid_info, fft_in_dtype=np.float32, fft_out_dtype=np.complex64
    )
    # mask
    timelog.start("MakeMask")
    if nomask:
        mmap = np.ma.make_mask_none(grid_info.grid_shape)
    elif ipmask is None:
        # calculate model map density
        timelog.start("MapDensity")
        cmap_grid = calculate_density(
            structure=structure,
            reso=res,
            rate=1.5,
        )
        timelog.end("MapDensity")
        mask_map, mask_grid = map_funcs.make_mask_from_maps(
            [fullMap, cmap_grid],
            grid_info,
            res,
            lpfilt_pre=True,
            radcyc=30.0,
            verbose=verbose,
        )
        # mmap = np.ma.make_mask(np.logical_not(mask_grid))
        mmap = mask_grid.mask
        # if verbose >= 5:
        #    temp_ma = combined_map.copy()
        #    temp_ma[mmap] = 0.0
        #    fileio.write_map_as_MRC(
        #        temp_ma, fullMap.unit_cell, outpath=f"calculated_mask.mrc"
        #    )
    else:
        mask_map = map_funcs.lowpass_map(
            maskin.grid,
            grid_info.voxel_size[0] / res,
            fftobj=fft_obj0,
            ifftobj=ifft_obj0,
        )
        mask_map = map_funcs.resample_data_by_boxsize(mask_map, grid_info.grid_shape)
        filt_data_r, f000 = map_funcs.make_filter_edge_centered(
            grid_info.grid_shape, filter_radius=30.0, function=function_type
        )
        mask_map = map_funcs.fft_convolution_filter(
            mask_map, filt_data_r, 1.0 / f000, grid_info.grid_shape
        )
        mmap = np.ma.masked_less(mask_map, mask_map.mean()).mask
        # mmap = np.ma.make_mask(np.logical_not(mask_grid))
    print(f"mmap size {mmap.size}, non zero {np.count_nonzero(mmap)}")
    timelog.end("MakeMask")
    if verbose >= 5:
        fileio.write_mask_as_MRC(mmap, fullMap.unit_cell, outpath="calculated_mask.mrc")

    # CASE 1: Refine model against EM data
    print("\nRefine model against EM map")
    res_list = (np.linspace(res, resbycyc[0], int(ncycrr)))[::-1]
    # less fragment at low res
    eps_list = np.linspace(5.2, 5.45, int(ncycrr))[::-1]
    # eps_list = np.linspace(3.7, 4.0, int(ncycrr))

    print(f"Running refinement on resolutions: {res_list}")
    for rcyc in range(0, len(res_list)):  # resolution cycle; macro-cycle
        resolution = res_list[rcyc]
        if pseudoreg != "no":
            print(
                "\n Refine-pseudo regularise cycle, Resolution: {0} A\n".format(
                    resolution
                )
            )
            if resolution >= 4.5 and alternative_clustering:
                preg.get_frags_clusters(
                    atom_selection="centre",  # "one_per_residue"
                    dbscan_eps=eps_list[rcyc],
                    attr_name=f"cluster_rescyc{str(rcyc)}",
                    outfile_suffix=splitext(basename(ippdb))[0],
                )
                dbscan_cluster = True  # False
            else:
                dbscan_cluster = False
        else:
            print("\n Refine cycle, Resolution: {0} A\n".format(resolution))
        result = []
        # set radius if not user specified
        radcyc = rad
        if radcyc <= 0.0:
            if resolution < 6.0 and (radscl >= 5.0 or radscl < 4.0):
                radscl = 4.5
            radcyc = radscl * resolution
        print("\n Cycle: {0}   Radius: {1}\n".format(rcyc + 1, radcyc))
        # downsample maps
        # changes 22-25 Oct 21 downsampling maps (coarser grids),
        # larger pixel size/spacing used. res/(2*samprate) = min spacing,
        # samp rate default = 1.5 ; took gemmi code
        # faster overall; but should be able to optimize more by calculating
        # better gridshapes.
        # don't have to use lowpass for now
        # 17 Feb - should low pass filter before resample to
        # remove artifact of high reso
        spacing = resolution / (2 * samp_rate)
        downsamp_shape, downsamp_apix = sf_util.calc_best_grid_apix(
            spacing,
            (fullMap.unit_cell.a, fullMap.unit_cell.b, fullMap.unit_cell.c),
        )
        gridshape = cell.GridInfo(
            downsamp_shape,
            grid_info.grid_start,
            downsamp_shape,
            downsamp_apix,
            grid_info.origin,
        )
        if verbose >= 2:
            print(f" INFO: Calculated spacing : {spacing}")
            print(repr(gridshape))

        timelog.start("FFTPlan")
        fft_obj, ifft_obj = sf_util.plan_fft_ifft(
            gridinfo=gridshape, fft_in_dtype=np.float32, fft_out_dtype=np.complex64
        )
        timelog.end("FFTPlan")
        timelog.start("Resample")

        downsamp_map = map_funcs.lowpass_map(
            fullMap,
            cutoff=grid_info.voxel_size[0] / resolution,
            fftobj=fft_obj0,
            ifftobj=ifft_obj0,
        )
        downsamp_map = map_funcs.resample_data_by_boxsize(downsamp_map, downsamp_shape)
        timelog.end("Resample")
        # resample calculate model map density to match downsamp_map grid shape
        cmap_grid = calculate_density(
            structure=structure, reso=resolution, rate=samp_rate
        )
        lp_cutoff = (structure.cell.parameters[0] / cmap_grid.shape[0]) / resolution
        cmap_grid, cmap_fftobj, cmap_ifftobj = map_funcs.lowpass_map(
            cmap_grid,
            cutoff=lp_cutoff,
            grid_shape=cmap_grid.shape,
            grid_reci=np.array(
                [cmap_grid.shape[0], cmap_grid.shape[1], cmap_grid.shape[2] // 2 + 1]
            ),
            return_fftobj=True,
        )
        cmap_grid = map_funcs.resample_data_by_boxsize(cmap_grid, downsamp_shape)
        # cmap_grid = calculate_density_with_boxsize(
        #    structure=structure,
        #    reso=resolution,
        #    rate=samp_rate,
        #    grid_shape=downsamp_shape,
        # )
        timelog.end("MapDensity")
        timelog.start("Resample")
        # cmap_grid = map_funcs.lowpass_map(cmap_grid, cutoff=grid_info.voxel_size[0]/resolution, gridinfo=grid_info) # noqa=E501
        # cmap_grid = map_funcs.resample_data_by_boxsize(cmap_grid, downsamp_shape)
        # resample the mask
        # mmap_downsamp = np.ma.resize(mmap, downsamp_shape)
        # mmap_downsamp = map_funcs.downsample_mask(mmap.astype(int), downsamp_shape)
        if nomask:
            mmap_downsamp = np.ma.make_mask_none(downsamp_shape)
        elif not np.all(grid_info.grid_shape == downsamp_shape):
            combinedmap_downsamp = map_funcs.resample_data_by_boxsize(
                mask_map, downsamp_shape
            )
            print(
                f"{combinedmap_downsamp.min()}, {combinedmap_downsamp.max()}, "
                f"{combinedmap_downsamp.mean()}, {combinedmap_downsamp.std()}"
            )
            mmap_downsamp = np.ma.masked_less(
                combinedmap_downsamp, combinedmap_downsamp.mean()
            ).mask
        else:
            mmap_downsamp = mmap
        if verbose >= 5:
            print(
                f"DEBUG:mmap size {mmap_downsamp.size}, "
                f"non zero {np.count_nonzero(mmap_downsamp)}"
            )
        timelog.end("Resample")

        # downsamp_apix = downsamp_map.apix
        # need to update this for mask if true

        if verbose >= 5:
            fileio.write_map_as_MRC(
                downsamp_map, fullMap.unit_cell, outpath=f"downsamp_map_{rcyc+1}.mrc"
            )
            fileio.write_map_as_MRC(
                cmap_grid, fullMap.unit_cell, outpath=f"cmapgrid_{rcyc+1}.mrc"
            )

            # temp_ma = downsamp_map + cmap_grid
            # temp_ma[mmap_downsamp] = 0.0
            # fileio.write_map_as_MRC(
            #    temp_ma, fullMap.unit_cell, outpath=f"calculated_mask_{rcyc+1}.mrc"
            # )
            fileio.write_mask_as_MRC(
                mmap_downsamp,
                fullMap.unit_cell,
                outpath=f"calculated_mask_{rcyc+1}.mrc",
            )
        # calculate difference map
        # truncate resolution - low pass filter; lpfiltb = True
        # spherical tophat function fall=0.01 tophat
        # in terms of b-factor shifts # recheck
        # refsc > dust,refsc=False > refsc, dust > dust
        # in terms of model better to worst # recheck
        # dust,refsc=False > refsc > dust > refsc, dust
        if refxyz:
            timelog.start("DiffMap")
            scl_map, scl_cmap, dmap = map_funcs.calc_diffmap(
                downsamp_map,
                cmap_grid,
                resolution,
                resolution,
                gridshape,
                gridshape,
                lpfilt_pre=False,
                lpfilt_post=False,
                refscl=True,
                randsize=0.1,
                flag_dust=False,
                verbose=verbose,
                fft_obj=fft_obj,
                ifft_obj=ifft_obj,
            )
            timelog.end("DiffMap")
            if verbose >= 4:
                fileio.write_map_as_MRC(
                    scl_map, fullMap.unit_cell, outpath=f"scl_mapin_{rcyc+1}.mrc"
                )
                fileio.write_map_as_MRC(
                    scl_cmap, fullMap.unit_cell, outpath=f"scl_cmap_{rcyc+1}.mrc"
                )
        converge = False
        cyc = 1  # internal cycle index
        diffmap_lpfilter = True
        while not converge:
            # for RMSD calculation at the end
            model_current = structure[0].clone()

            if refxyz:
                print(" REFINE XYZ")
                timelog.start("Shiftfield")
                x1m, x2m, x3m = shift_field_coord(
                    scl_cmap,
                    dmap,
                    mmap_downsamp,
                    radcyc,
                    function_type,
                    gridshape,
                    fft_obj,
                    ifft_obj,
                    cyc,
                    verbose=verbose,
                    timelog=timelog,
                )
                timelog.end("Shiftfield")

                # Use gemmi interpolate and update positions
                # convert numpy array to FloatGrid first for x1m, x2m, x3m
                timelog.start("Numpy2Grid")
                grid_dx = map_funcs.numpy_to_gemmi_grid(
                    x1m, fullMap.unit_cell, spacegroup="P1"
                )
                grid_dy = map_funcs.numpy_to_gemmi_grid(
                    x2m, fullMap.unit_cell, spacegroup="P1"
                )
                grid_dz = map_funcs.numpy_to_gemmi_grid(
                    x3m, fullMap.unit_cell, spacegroup="P1"
                )
                timelog.end("Numpy2Grid")
                timelog.start("UpdateModel")
                map_funcs.update_atoms_position(
                    grid_dx,
                    grid_dy,
                    grid_dz,
                    structure,
                    mode="tricubic",
                    selection=selection,
                )
                timelog.end("UpdateModel")

            # run pseudo-regularisation end of every shift-field iteration
            # not recommended as the convergence rate is slower
            # i.e. shifts might get cancelled or reduced from pseudo-regularisation
            if pseudoreg == "yes" and resolution >= 6.0:
                timelog.start("PseudoReg")
                preg.regularize_frag(structure, dbscan_cluster=dbscan_cluster)
                timelog.end("PseudoReg")

            # calculate model map density after refinement
            cmap_grid = calculate_density_with_boxsize(
                structure=structure,
                reso=resolution,
                rate=samp_rate,
                grid_shape=downsamp_shape,
                fft_obj=cmap_fftobj,
                ifft_obj=cmap_ifftobj,
            )
            timelog.start("DiffMap")
            scl_map, scl_cmap, dmap = map_funcs.calc_diffmap(
                downsamp_map,
                cmap_grid,
                resolution,
                resolution,
                gridshape,
                gridshape,
                lpfilt_pre=diffmap_lpfilter,
                lpfilt_post=False,
                refscl=True,
                randsize=0.1,
                flag_dust=False,
                verbose=verbose,
                fft_obj=fft_obj,
                ifft_obj=ifft_obj,
            )
            timelog.end("DiffMap")
            if verbose >= 5:
                fileio.write_map_as_MRC(
                    scl_map, fullMap.unit_cell, outpath=f"scl_mapin_{rcyc+1}_{cyc}.mrc"
                )
                fileio.write_map_as_MRC(
                    scl_cmap, fullMap.unit_cell, outpath=f"scl_cmap_{rcyc+1}_{cyc}.mrc"
                )
            timelog.start("Scoring")
            # calculate map contour
            mapcurreso_t = map_funcs.calculate_map_threshold(scl_map)
            fltrcmap_t = map_funcs.calculate_map_threshold(scl_cmap)
            m = " Calculated map and model threshold are "
            m += f"{mapcurreso_t:.2f} and {fltrcmap_t:.2f} Angstroms\n"
            print(m)
            ovl_map, ovl_mdl = map_funcs.calculate_overlap_scores(
                scl_map, scl_cmap, mapcurreso_t, fltrcmap_t
            )
            if ovl_map >= 0.5:
                diffmap_lpfilter = False
            else:
                diffmap_lpfilter = True
            timelog.end("Scoring")
            m = " TEMPy scores :\n"
            m += f"  Fraction of map overlapping with model : {ovl_map:.3f}\n"
            m += f"  Fraction of model overlapping with map : {ovl_mdl:.3f}\n"
            print(m)
            # check rmsd convergence
            coor_sum_sq = 0.0
            count = 0
            for ch in range(0, len(structure[0])):
                for r in range(0, len(structure[0][ch])):
                    for a in range(0, len(structure[0][ch][r])):
                        atom2 = structure[0][ch][r][a]
                        coor_sum_sq += model_current[ch][r][a].pos.dist(atom2.pos) ** 2
                        count += 1
            coor_rmsd = np.sqrt(coor_sum_sq / count)
            if verbose >= 2:
                print(f" RMSD : {coor_rmsd:.4f}")
            if rcyc == 0:
                tolerance = (coor_tol / 2) * resolution
            else:
                tolerance = coor_tol * resolution
            if coor_rmsd <= tolerance:
                converge = True
            elif ovl_map >= 0.8 and ovl_mdl >= 0.8:
                converge = True

            # Save results for every iteration
            temp_result = sf_util.ResultsByCycle(
                rcyc,
                cyc,
                resolution,
                radcyc,
                ovl_map,
                ovl_mdl,
                0.0,  # for FSCavg future devel
            )
            result.append(temp_result)
            # output intermediates of every iteration
            if output_intermediate:
                outfname = "{0}_intermediate_rescyc{1}_cyc{2}.pdb".format(
                    pathfname, rcyc + 1, cyc
                )
                structure.write_minimal_pdb(f"{outfname}")
            # write out shifts
            # if len(shift_vars) != 0 and verbose >= 3:
            #    outcsv = "shiftvars1_linalg_{0}.csv".format(cyc + 1)
            #    fopen = open(outcsv, "w")
            #    for j in range(0, len(shift_vars)):
            #        fopen.write("{0}, {1}\n".format(j, shift_vars[j]))
            #    fopen.close()
            sys.stdout.flush()
            # break using specified number of ncyc
            if cyc >= ncyc:
                break
            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #    print(stat)
            """
            if len(shift_u) != 0 and verbose >= 2:
                outusiocsv = "shiftuiso_u2b_{0}.csv".format(cyc + 1)
                fuiso = open(outusiocsv, "w")
                for j in range(0, len(shift_u)):
                    fuiso.write("{0}, {1}\n".format(j, shift_u[j]))
                fuiso.close()
            """
            cyc += 1
            # end of convergence cycle loop

        # after every resolution if pseudoreg is set to postref
        if pseudoreg == "postref" or pseudoreg == "yes":
            timelog.start("PseudoReg")
            preg.regularize_frag(structure, dbscan_cluster=dbscan_cluster)
            timelog.end("PseudoReg")
            # logger.info(f"End of refine-regularise cycle {cycrr+1}")
            # logger.info("TEMPys scores :")
            # logger.info(
            #    "Fraction of map overlapping with model: {0:.3f}".format(ovl_mapf)
            # )
            # logger.info(
            #    "Fraction of model overlapping with map: {0:.3f}".format(ovl_mdlf)
            # )
        # U-isotropic refinement
        # better to run at the end of each Regularise-Refine cycle (macro-cycle)
        uiso_refined = False

        if refuiso:  # and (rcyc in [0, len(res_list)//2, len(res_list)-1]):
            uiso_converged = False
            ucyc = 0
            print(" REFINE UISO")
            if verbose >= 2:
                outfname = "{0}_intermediate_rescyc{1}_cyc{2}_pre_refuiso.pdb".format(
                    pathfname, rcyc + 1, cyc
                )
                structure.write_minimal_pdb(f"{outfname}")
            while not uiso_converged:
                # calculate model map density
                model_current = structure[0].clone()
                # test refine uiso with user given map resolution
                timelog.start("MapDensity")
                cmap_grid = calculate_density_with_boxsize(
                    structure=structure,
                    reso=resolution,
                    rate=samp_rate,
                    grid_shape=downsamp_shape,
                    fft_obj=cmap_fftobj,
                    ifft_obj=cmap_ifftobj,
                    # lpfilt=False
                )
                if verbose > 6:
                    fileio.write_map_as_MRC(
                        cmap_grid,
                        fullMap.unit_cell,
                        spacegroup="P1",
                        outpath=f"cmap_uiso{rcyc}.mrc",
                    )
                timelog.end("MapDensity")
                # calculate difference map
                timelog.start("DiffMap")
                scl_map, scl_cmap, dmap = map_funcs.calc_diffmap(
                    downsamp_map,
                    cmap_grid,
                    resolution,
                    resolution,
                    gridshape,
                    gridshape,
                    lpfilt_pre=False,
                    lpfilt_post=False,
                    refscl=True,
                    randsize=0.1,
                    flag_dust=True,
                    verbose=verbose,
                    fft_obj=fft_obj,
                    ifft_obj=ifft_obj,
                )
                if verbose >= 5:
                    fileio.write_map_as_MRC(
                        scl_cmap,
                        fullMap.unit_cell,
                        outpath=f"scl_cmap_{rcyc+1}_{cyc}_{ucyc}.mrc",
                    )

                timelog.end("DiffMap")
                # check uiso convergence by model to map overlap score
                mapin_t = map_funcs.calculate_map_threshold(scl_map)
                cmap_t = map_funcs.calculate_map_threshold(scl_cmap)
                ovl_map, ovl_mdl = map_funcs.calculate_overlap_scores(
                    scl_map, scl_cmap, mapin_t, cmap_t
                )
                m = " Calculated map & model threshold are "
                m += "{0:.2f} and {1:.2f} \n".format(
                    mapin_t,
                    cmap_t,
                )
                m += " Fraction of map overlapping with model: "
                m += "{0:.3f} \n".format(ovl_map)
                m += " Fraction of model overlapping with map: "
                m += "{0:.3f}\n".format(ovl_mdl)
                print(m)
                if ovl_map >= 0.90 and ovl_mdl >= 0.90:
                    uiso_converged = True
                    if rcyc == (len(res_list) - 1):
                        break
                # start refine u-iso
                timelog.start("UISO")
                x1m = shift_field_uiso(
                    scl_cmap,  # cmap_grid,  #
                    dmap,
                    mmap_downsamp,
                    radcyc,
                    function_type,
                    gridshape,
                    fullMap.unit_cell.parameters,
                    fft_obj=fft_obj,
                    ifft_obj=ifft_obj,
                    verbose=verbose,
                    timelog=timelog,
                )
                timelog.end("UISO")
                timelog.start("Numpy2Grid")
                grid_du = map_funcs.numpy_to_gemmi_grid(
                    x1m, fullMap.unit_cell, spacegroup="P1"
                )
                timelog.end("Numpy2Grid")
                timelog.start("UpdateModel")
                map_funcs.update_uiso_values(
                    grid_du,
                    structure,
                    biso_range,
                    mode="tricubic",
                    cycle=rcyc + 1,
                    verbose=verbose,
                    ucyc=ucyc + 1,
                )
                timelog.end("UpdateModel")  # end of u-iso refine
                # only do convergence check/loop at the final macro cycle
                # before final cycle just run once for the b-factor
                uiso_converged = True

                if rcyc != (len(res_list) - 1):
                    uiso_converged = True
                else:
                    outfname = f"{pathfname}_uiso_rescyc{rcyc+1}_ucyc_{ucyc+1}.pdb"
                    structure.write_minimal_pdb(f"{outfname}")

                ucyc += 1
            uiso_refined = True

        outfname = "{0}_refined_rescyc{1}".format(pathfname, rcyc + 1)
        if pseudoreg != "no":
            outfname += "_pseudoreg"
        elif uiso_refined:
            outfname += "_refuiso"
        outfname += ".pdb"
        structure.write_minimal_pdb(f"{outfname}")
        results.append(result)
        rcyc += 1
        # end of resolution cycle loop (macro cycle)

    # Final overlap scores
    timelog.start("MapDensity")
    cmap_grid = calculate_density_with_boxsize(
        structure=structure,
        reso=res,
        rate=samp_rate,
        grid_shape=gridshape.grid_shape,
    )
    timelog.start("MapDensity")
    timelog.start("Scale")
    scl_map, scl_cmap = map_funcs.global_scale_maps(
        fullMap,
        cmap_grid,
        grid_info,
        grid_info,
        res,
        res,
        lpfilt_pre=True,
        ref_scale=True,
    )
    timelog.end("Scale")
    timelog.start("Scoring")
    if verbose > 5:
        fileio.write_map_as_MRC(
            grid_data=scl_cmap,
            unitcell=fullMap.unit_cell.parameters,
            spacegroup="P1",
            outpath="final_scl_cmap.mrc",
        )
        fileio.write_map_as_MRC(
            grid_data=scl_map,
            unitcell=fullMap.unit_cell.parameters,
            spacegroup="P1",
            outpath="final_scl_map.mrc",
        )
    mapin_t = map_funcs.calculate_map_threshold(scl_map)
    cmap_t = map_funcs.calculate_map_threshold(scl_cmap)
    ovl_map, ovl_mdl = map_funcs.calculate_overlap_scores(
        scl_map, scl_cmap, mapin_t, cmap_t
    )
    timelog.end("Scoring")
    m = "Calculated map & model threshold are "
    m += "{0:.2f} and {1:.2f} \n".format(
        mapin_t,
        cmap_t,
    )
    m += "Final TEMPy scores :\n"
    m += " Fraction of map overlapping with model: "
    m += "{0:.3f} \n".format(ovl_map)
    m += " Fraction of model overlapping with map: "
    m += "{0:.3f}\n".format(ovl_mdl)
    print(m)
    # write final pdb
    outfname = "{0}_refined_final.pdb".format(pathfname)
    structure.write_minimal_pdb(f"{outfname}")
    # shift refined model back to input map origin if they have been offset
    if not np.array_equal(ori0, grid_info.origin):
        tr = gemmi.Transform(gemmi.Mat33(), gemmi.Vec3(*ori0))
        structure[0].transform_pos_and_adp(tr)
        outfname = "{0}_refined_final_maporigin_match.pdb".format(pathfname)
        structure.write_minimal_pdb(f"{outfname}")
    # TODO: restructure XML results or even JSON
    # write xml results
    # if xmlout is not None:
    #    f = open(xmlout, "w")
    #    for m in range(0, len(res_list)):
    #        for i in range(0, ncyc):
    #            if m == 0 and i == 0:
    #                results[m][i].write_xml_results_header(f, outfname, ippdb)
    #            if m > 0 and i == 0:
    #                results[m][i].write_xml_results_start(f)
    #            results[m][i].write_xml_results_cyc(f)
    #            if i == ncyc - 1:
    #                results[m][i].write_xml_results_end_macrocyc(f)
    #        if m == len(res_list) - 1:
    #            results[m][i].write_xml_results_final(f)
    #    f.close()

    # logger.info(f"Ended at {datetime.datetime.now()}")
    if verbose >= 2:
        timelog.profile_log()


if __name__ == "__main__":
    # parser = SheetbendParser()
    # parser.get_args()
    # main(parser.args)
    from sheetbend_cmdln_parser import SheetbendParser

    sb_parser = SheetbendParser()  # args=args)
    sb_parser.parse_args(sys.argv[1:])
    main(sb_parser.args)
