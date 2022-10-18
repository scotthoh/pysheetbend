from __future__ import absolute_import, print_function
from logging import raiseExceptions
import gemmi
from matplotlib.streamplot import Grid
import numpy as np
from enum import Enum
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from regex import F
from scipy.signal import resample
from scipy.ndimage import measurements
from pysheetbend.shiftfield import effective_radius, fltr
import pysheetbend.shiftfield_util as sf_util
from math import fabs
from pysheetbend.utils import dencalc
from pysheetbend.utils.cell import GridInfo


class functionType(Enum):
    STEP = 0
    LINEAR = 1
    QUADRATIC = 2


def map_grid_position_array(grid_info, structure, index=True):
    xyz_coordinates = []
    origin = np.array((grid_info.origin[0], grid_info.origin[1], grid_info.origin[2]))
    apix = np.array(
        (grid_info.voxel_size[0], grid_info.voxel_size[1], grid_info.voxel_size[2])
    )
    # print(origin, apix)
    out = np.zeros(3, dtype=np.int16)
    for cra in structure[0].all():
        xyz = np.array((cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z))
        if index:
            xyz = np.around((xyz - origin) / apix, 0, out)
            # xyz = (np.round(((xyz - origin) / apix, 0))).astype(int)
        else:
            xyz = (xyz - origin) / apix
        xyz_coordinates.append([xyz[0], xyz[1], xyz[2]])
    # print(xyz_coordinates)
    return xyz_coordinates


def fltr(r, radius, function: functionType = functionType(2)):
    '''
    Radial map filter, returns value from filter function,
    outside radius=0.0
    Arguments
        r: distance
        radius: radius cutoff
        function: modes 0 : step, 1 : linear, 2 : quadratic
    Return
        Radius value from filter function
    '''
    if r < radius:
        if function == functionType(2):
            return pow(1.0 - r / radius, 2)
        elif function == functionType(1):
            return 1.0 - r / radius
        elif function == functionType(0):
            return 1.0
    else:
        return 0.0


def effective_radius(func, radius):
    '''
    Gets effective radius from function and radius given
    Arguments
        func: function modes; 0 : step, 1 : linear, 2 : quadratic
        radius: radius cutoff
    Return
        effective radius value
    '''
    nrad = 1000
    drad = 0.25
    sum_r = np.zeros(nrad)
    for i in range(0, nrad):
        r = drad * (float(i) + 0.5)
        sum_r[i] = r * r * fabs(fltr(r, radius, func))
        if i >= 1:
            sum_r[i] += sum_r[i - 1]

    for i in range(0, nrad):
        if sum_r[i] > 0.99 * sum_r[nrad - 1]:
            break

    return drad * (float(i) + 1.0)


def prepare_mask_filter(apix: np.ndarray, fltr_radius=15.0, pad=5):
    '''
    Prepare filter kernel to be used with scipy.fftconvolve
    Arguments
        apix: pixel size [x,y,z]
        fltr_radius: radius for filter
        pad: padding for filter kernel
    Return
        Filter 3D array
    '''
    rad = effective_radius(functionType(2), fltr_radius)
    win_points = int(fltr_radius * 2) + 1 + (pad * 2)
    start = (fltr_radius + pad) * -1
    end = fltr_radius + pad
    rad_x = np.linspace(start, end, num=win_points)
    rad_y = np.linspace(start, end, num=win_points)
    rad_z = np.linspace(start, end, num=win_points)
    rad_x = rad_x * apix[0]
    rad_y = rad_y * apix[1]
    rad_z = rad_z * apix[2]
    rad_x = rad_x ** 2
    rad_y = rad_y ** 2
    rad_z = rad_z ** 2
    dist = np.sqrt(rad_x[:, None, None] + rad_y[:, None] + rad_x)
    dist_ind = zip(*np.nonzero(dist < rad))
    fltr = np.zeros(dist.shape)
    count = 0
    # f000 = 0.0
    # fill the radial function map
    for i in dist_ind:
        rf = fltr(dist[i], fltr_radius, 2)
        count += 1
        fltr[i] = rf

    return fltr


def make_filter_edge_centered(grid_info, filter_radius=15.0, function=functionType(2)):
    eff_rad = effective_radius(function, filter_radius)
    fltr_data_r = np.zeros(grid_info.grid_shape, dtype=np.float32)
    nx, ny, nz = np.indices(grid_info.grid_shape)
    indi = np.vstack([nx.ravel(), ny.ravel(), nz.ravel()]).T
    c = indi + grid_info.grid_half
    c_bool = c < 0
    c[c_bool[:, 0], 0] += grid_info.grid_shape[0]
    c[c_bool[:, 1], 1] += grid_info.grid_shape[1]
    c[c_bool[:, 2], 2] += grid_info.grid_shape[2]
    c -= grid_info.grid_half
    pos = c[:] * grid_info.voxel_size + grid_info.grid_start
    dist = np.sqrt(np.sum(np.square(pos), axis=1)).reshape(grid_info.grid_shape)
    dist_bool = np.logical_and((dist < eff_rad), (dist < filter_radius))
    f000 = 0.0
    if function == functionType(2):  # quadratic
        rf = pow(1.0 - dist[dist_bool] / filter_radius, 2)
    elif function == functionType(1):  # linear
        rf = 1.0 - dist[dist_bool] / filter_radius
    elif function == functionType(0):  # step
        rf = 1.0
    else:
        raise ValueError('Choose from function mode 2:Quadratic, 1:Linear, 0:Step.')
    f000 = np.sum(rf)
    fltr_data_r[dist_bool] = rf[:]
    print(f'f000 = {f000:.4f}')

    return fltr_data_r, f000


def fft_convolution_filter(
    data_arr: np.ndarray,
    filter: np.ndarray,
    scale,
    fft_obj,
    ifft_obj,
    grid_info: GridInfo,
):
    '''
    Returns filtered/covolved data
    Arguments:
        data_arr: numpy array of data
        filter: numpy array of filter
        scale: scale value for the fft convolution
        fft_obj: planned fft object
        ifft_obj: planned ifft object
        grid_info: GridInfo class containing grid info
    '''
    data_r = np.zeros(grid_info.grid_shape, dtype=np.float32)
    data_r = data_arr.copy()
    fltr_input = np.zeros(grid_info.grid_shape, dtype=np.float32)
    fltr_input = filter.copy()

    fltr_data_c = np.zeros(grid_info.grid_reci, dtype=np.complex64)
    data_c = np.zeros(grid_info.grid_reci, dtype=np.complex64)
    fltr_data_c = fft_obj(fltr_input, fltr_data_c)
    fltr_data_c = fltr_data_c.conjugate().copy()

    data_c = fft_obj(data_r, data_c)
    data_c = data_c.conjugate().copy()

    data_c[:, :, :] = scale * data_c[:, :, :] * fltr_data_c[:, :, :]
    data_c = data_c.conjugate().copy()
    data_r = ifft_obj(data_c, data_r)

    return data_r


def find_background_peak(map_array, iter=2):
    """
    rewritten from ccpem TEMPy (from Agnel)
    Arguments:
        mapdata: data array
        iter: iteration cycles (default=2)
    Return:
        peak, average
    """
    lbin = np.amin(map_array)
    rbin = np.amax(map_array)
    avg = np.nanmean(map_array)  # .mean()
    sigma = np.nanstd(map_array)  # .std()
    for i in range(iter):
        if i == 0:
            data = map_array
        else:
            data = map_array[(map_array >= lbin) & (map_array <= rbin)]

        freq, bins = np.histogram(data, 100)
        ind = np.nonzero(freq == np.amax(freq))[0]
        peak = None
        for j in ind:
            val = (bins[j] + bins[j + 1]) / 2.0
            if val < float(avg) + float(sigma):
                peak = val
                lbin = bins[j]
                rbin = bins[j + 1]
            if peak is None:
                break

    return peak, avg


def get_peak_and_sigma(map_array):
    peak, avg = find_background_peak(map_array)

    if peak is None:
        peak = avg
    sigma = None
    if peak is not None:
        mask_array = map_array[map_array > peak]
        sigma = np.sqrt(np.mean(np.square(mask_array - peak)))

    return peak, avg, sigma


def calculate_map_threshold(map_data, sigma_factor: float = 2.0):
    if isinstance(map_data, gemmi.FloatGrid):
        maparray = np.array(map_data, copy=False, dtype=np.float32)
    else:
        maparray = map_data
    try:
        peak, avg, sigma = get_peak_and_sigma(maparray)
        vol_threshold = float(avg) + (sigma_factor * float(sigma))
    except:
        try:
            mean = np.nanmean(maparray)
            sigma = np.nanstd(maparray)
            vol_threshold = float(mean) + (1.5 * float(sigma))
        # peak, avg, sigma = get_peak_and_sigma(map_data)
        except:
            vol_threshold = 0.0

    return vol_threshold


def calculate_map_contour(map_data, sigma_factor: float = 1.5):
    if isinstance(map_data, gemmi.FloatGrid):
        maparray = np.array(map_data, copy=False, dtype=np.float32)
        peak, avg, sigma = get_peak_and_sigma(maparray)
    else:
        peak, avg, sigma = get_peak_and_sigma(map_data)
    if peak is not None:
        contour = float(peak) + (sigma_factor * float(sigma))
    else:
        contour = 0.0

    return contour


def calculate_overlap_scores(
    map_data,
    map_data2,
    map_threshold,
    map_threshold2,
):
    binmap1 = map_data > float(map_threshold)
    binmap2 = map_data2 > float(map_threshold2)
    mask_array = (binmap1 * binmap2) > 0

    size1 = np.sum(binmap1)
    size2 = np.sum(binmap2)
    return float(np.sum(mask_array)) / size1, float(np.sum(mask_array)) / size2


def calculate_pixel_size(cell_size, grid_shape):

    x = cell_size.a / float(grid_shape[0])  # x
    y = cell_size.b / float(grid_shape[1])  # y
    z = cell_size.c / float(grid_shape[2])  # z

    return np.array((x, y, z), dtype=np.float32)


def resample_data_by_boxsize(map_data, new_shape):
    if isinstance(map_data, gemmi.FloatGrid):
        grid_data = np.array(map_data, copy=False)  # , dtype=np.float32)
        grid_data = resample(grid_data, new_shape[0], axis=0)
        grid_data = resample(grid_data, new_shape[1], axis=1)
        grid_data = resample(grid_data, new_shape[2], axis=2)
        # map_data = gemmi.FloatGrid(grid_data)
        return grid_data
    else:
        if isinstance(map_data, np.ndarray):
            map_data = resample(map_data, new_shape[0], axis=0)
            map_data = resample(map_data, new_shape[1], axis=1)
            map_data = resample(map_data, new_shape[2], axis=2)
            return map_data
        else:
            return map_data


def compare_tuple(tuple1, tuple2):
    return bool(np.asarray(tuple1 == tuple2).all())


def make_mask_from_maps(
    maps,
    grid_info,
    resolution=2.0,
    func: functionType = functionType(2),
    lpfilt_pre=False,
    lpfilt_post=False,
    ref_scale=False,
):
    # amplitude match
    fft_obj = sf_util.plan_fft(grid_info, input_dtype=np.float32)
    ifft_obj = sf_util.plan_ifft(grid_info, input_dtype=np.complex64)

    scaled_maps = amplitude_match(
        maps[0],
        maps[1],
        grid_info,
        grid_info,
        reso=resolution,
        lpfilt_pre=lpfilt_pre,
        lpfilt_post=lpfilt_post,
        ref_scale=ref_scale,
        fft_obj=fft_obj,
        ifft_obj=ifft_obj,
    )

    combined_map = scaled_maps[0] + scaled_maps[1]
    filt_data_r, f000 = make_filter_edge_centered(grid_info, function=func)
    mmap = fft_convolution_filter(
        combined_map,
        filt_data_r,
        1.0 / f000,
        fft_obj,
        ifft_obj,
        grid_info,
    )
    mmapt = calculate_map_threshold(mmap)
    mmap = np.ma.masked_less(mmap, mmapt)

    return mmap


def make_fourier_shell(map_shape, keep_shape=False, fftshift=True, normalise=False):
    """
    Make grid sampling frequencies as distance from centre for a given grid
    Return:
        Grid with sampling frequencies
    """
    if keep_shape:
        rad_0 = np.arange(
            np.floor(map_shape[0] / 2.0) * -1, np.ceil(map_shape[0] / 2.0)
        ) / float(np.floor(map_shape[0]))
        rad_1 = np.arange(
            np.floor(map_shape[1] / 2.0) * -1, np.ceil(map_shape[1] / 2.0)
        ) / float(np.floor(map_shape[1]))
        rad_2 = np.arange(
            np.floor(map_shape[2] / 2.0) * -1, np.ceil(map_shape[2] / 2.0)
        ) / float(np.floor(map_shape[2]))
        if not fftshift:
            rad_2 = ifftshift(rad_2)
    else:
        rad_0 = np.arange(
            np.floor(map_shape[0] / 2.0) * -1, np.ceil(map_shape[0] / 2.0)
        ) / float(np.floor(map_shape[0]))
        rad_1 = np.arange(
            np.floor(map_shape[1] / 2.0) * -1, np.ceil(map_shape[1] / 2.0)
        ) / float(np.floor(map_shape[1]))
        rad_2 = np.arange(0, np.floor(map_shape[2] / 2.0) + 1) / float(
            np.floor(map_shape[2])
        )
    if not fftshift:
        rad_0 = fftshift(rad_0)
        rad_1 = fftshift(rad_1)
    if normalise:
        rad_0 = rad_0 / float(np.floor(map_shape[0]))
        rad_1 = rad_1 / float(np.floor(map_shape[1]))
        rad_2 = rad_2 / float(np.floor(map_shape[2]))
    rad_2 = rad_2 ** 2
    rad_1 = rad_1 ** 2
    rad_0 = rad_0 ** 2
    dist = np.sqrt(rad_0[:, None, None] + rad_1[:, None] + rad_2)
    return dist


def tanh_lowpass(map_shape, cutoff, fall=0.3, keep_shape=False):
    if fall == 0.0:
        fall = 0.01
    drop = np.pi / (2 * float(cutoff) * float(fall))
    cutoff = min(float(cutoff), 0.5)
    dist = make_fourier_shell(map_shape, keep_shape=keep_shape)
    dist1 = dist + cutoff
    dist1[:] = drop * dist1
    dist1[:] = np.tanh(dist1)
    dist[:] = dist - cutoff
    dist[:] = drop * dist
    dist[:] = np.tanh(dist)
    dist[:] = dist1 - dist
    dist = 0.5 * dist
    del dist1
    return dist


def amplitude_match(
    emmap1,
    emmap2,
    grid1_info,
    grid2_info,
    step=0.0005,
    reso=None,
    lpfilt_pre=False,
    lpfilt_post=False,
    ref_scale=False,
    fft_obj=None,
    ifft_obj=None,
):
    # Taken from Agnel's code in TEMPy library / CCPEM
    # assuming maps used are always cubic (same dimension x,y,z)
    ft1 = fft_obj(emmap1)
    ft1 = fftshift(ft1, axes=(0, 1))
    ft2 = fft_obj(emmap2)
    ft2 = fftshift(ft2, axes=(0, 1))
    # ft1 = fftshift(fftn(emmap1))
    # ft2 = fftshift(fftn(emmap2))
    # pre scaling lowpass filter
    if reso is not None:
        cutoff1 = grid1_info.voxel_size[0] / float(reso)
        cutoff2 = grid2_info.voxel_size[0] / float(reso)
        if lpfilt_post and not lpfilt_pre:
            ftfltr1 = tanh_lowpass(
                grid1_info.grid_shape, cutoff1, fall=0.2, keep_shape=False
            )
            ftfltr2 = tanh_lowpass(
                grid2_info.grid_shape, cutoff2, fall=0.2, keep_shape=False
            )
            ft1[:] = ft1 * ftfltr1
            ft2[:] = ft2 * ftfltr2
            del ftfltr1, ftfltr2
    # min dimension
    size1 = np.min(grid1_info.grid_shape)
    size2 = np.min(grid2_info.grid_shape)
    if step is None:
        step = 1.0 / min(
            size1 * grid1_info.voxel_size[0], size2 * grid2_info.voxel_size[0]
        )
    else:
        step = max(
            1.0
            / min(
                size1 * grid1_info.voxel_size[0],
                size2 * grid2_info.voxel_size[0],
            ),
            step,
        )
    dist1 = (
        make_fourier_shell(grid1_info.grid_shape, fftshift=True)
        / grid1_info.voxel_size[0]
    )
    dist2 = (
        make_fourier_shell(grid2_info.grid_shape, fftshift=True)
        / grid2_info.voxel_size[0]
    )
    print(dist1.shape, dist2.shape)
    # ft1_avg = []
    # ft2_avg = []
    # ft1_avg_new = []
    lfreq = []
    # select max spatial frequency to iterate to low resolution map
    maxlvl = 0.5 / min(grid1_info.voxel_size[0], grid2_info.voxel_size[0])
    # loop over freq shells, shellwidth = 0.005
    # scale amplitudes
    nc = 0
    x = 0.0
    highlvl = x + step
    while x < maxlvl:
        # indices between upper and lower shell bound
        fshells1 = (dist1 < min(maxlvl, highlvl)) & (dist1 >= x)
        # radial average
        shellvec1 = ft1[fshells1]
        # indices between upper and lower shell bound
        fshells2 = (dist2 < min(maxlvl, highlvl)) & (dist2 >= x)
        # radial average
        shellvec2 = ft2[fshells2]
        abs1 = abs(shellvec1)
        abs2 = abs(shellvec2)
        ns1 = len(np.nonzero(abs1)[0])
        ns2 = len(np.nonzero(abs2)[0])
        ###
        if (ns1 < 5 or ns2 < 5) and nc < 3:
            nc += 1
            highlvl = min(maxlvl, x + (nc + 1) * step)
            if highlvl < maxlvl:
                continue
            # x = max(0.0, x - nc * step)
            # continue
        else:
            nc = 0
        ft1_m = np.mean(np.square(abs1))
        ft2_m = np.mean(np.square(abs2))
        if ft1_m == 0.0 and ft2_m == 0.0:
            x = highlvl
            highlvl = x + step
            if highlvl < maxlvl:
                continue

        # sq of radial avg amplitude
        # ft1_avg.append(np.log10(np.mean(np.square(abs1))))
        # ft2_avg.append(np.log10(np.mean(np.square(abs2))))

        # scale to amplitudes of reference map, map1 if reference
        if ref_scale:
            if ft1_m == 0.0:
                continue
            ft1[fshells1] = shellvec1 * np.sqrt(ft2_m / ft1_m)
        else:
            # replace with avg amplitudes for the two maps
            ft1[fshells1] = shellvec1 * np.sqrt((ft2_m + ft1_m) / (2.0 * ft1_m))
            ft2[fshells2] = shellvec2 * np.sqrt((ft2_m + ft1_m) / (2.0 * ft2_m))
        lfreq.append(highlvl)
        x = highlvl
        highlvl = x + step

        # sampling_frq = highlvl
        # cutoff_freq = min((1.0 / reso) + 0.25, maxlvl)
        del fshells1, fshells2, shellvec1, shellvec2
        # scale the rest and break after relevant frequencies
        # if sampling_frq > cutoff_freq:
        #    fshells1 = dist1 >= highlvl
        #    shellvec1 = ft1[fshells1]
        #    ft1_m = np.mean(abs(shellvec1))
        #    fshells2 = dist2 >= highlvl
        #    shellvec2 = ft2[fshells2]
        #    ft2_m = np.mean(abs(shellvec2))
        #    if ft1_m == 0.0 and ft2_m == 0.0:
        #        break
        #    ft1_avg.append(np.log10(np.mean(np.square(abs(shellvec1)))))
        #    ft2_avg.append(np.log10(np.mean(np.square(abs(shellvec2)))))

        #   if ref_scale:
        #        if ft1_m == 0.0:
        #            break
        #        ft1[fshells1] = shellvec1 * (ft2_m / ft1_m)
        #    else:
        #        ft1[fshells1] = shellvec1 * (ft2_m + ft1_m) / (2 * ft1_m)
        #        ft2[fshells2] = shellvec2 * (ft2_m + ft1_m) / (2 * ft2_m)

        # ft1_m = np.mean(abs(ft1[fshells1]))
        # ft1_avg_new.
        #    lfreq.append((highlvl + step / 2))
        #    break
        # x = highlvl
        # highlvl = x + step

        # post scaling low pass filter
    if reso is not None:
        cutoff1 = grid1_info.voxel_size[0] / float(reso)
        cutoff2 = grid2_info.voxel_size[0] / float(reso)
        if lpfilt_pre and not lpfilt_post:
            ftfltr1 = tanh_lowpass(
                grid1_info.grid_shape, cutoff1, fall=0.2, keep_shape=False
            )
            ftfltr2 = tanh_lowpass(
                grid2_info.grid_shape, cutoff2, fall=0.2, keep_shape=False
            )
            ft1[:] = ft1 * ftfltr1
            ft2[:] = ft2 * ftfltr2
            del ftfltr1, ftfltr2
    # print('in amplitude scale {0}'.format(ft1.dtype))
    # print('in amplitude scale {0}'.format(ft2.dtype))
    map1_scaled = np.zeros(emmap1.shape, dtype=np.float32)
    map2_scaled = np.zeros(emmap2.shape, dtype=np.float32)
    map1_scaled = ifft_obj(ifftshift(ft1, axes=(0, 1)), map1_scaled)
    map2_scaled = ifft_obj(ifftshift(ft2, axes=(0, 1)), map2_scaled)
    # print('in amplitude scale {0}'.format(map1_scaled.dtype))
    # print('in amplitude scale {0}'.format(map2_scaled.dtype))
    return map1_scaled, map2_scaled


def grid_footprint():
    m = np.zeros((3, 3, 3))
    m[1, 1, 1] = 1
    m[0, 1, 1] = 1
    m[1, 0, 1] = 1
    m[1, 1, 0] = 1
    m[2, 1, 1] = 1
    m[1, 2, 1] = 1
    m[1, 1, 2] = 1
    return m


def label_patches(mapdata, map_threshold, prob=0.1, inplace=False):
    # Taken from Agnel's code in TEMPy library / CCPEM
    ftpt = grid_footprint
    binmap = mapdata > float(map_threshold)
    label_array, labels = measurements.label(mapdata * binmap, structure=ftpt)
    sizes = measurements.sum(binmap, label_array, range(labels + 1))

    if labels <= 10:
        m_array = sizes < 0.05 * sizes.max()
        ct_remove = np.sum(m_array)
        remove_points = m_array[label_array]
        label_array[remove_points] = 0
        if inplace:
            mapdata[:] = (label_array > 0) * (mapdata * binmap)
        else:
            newmap = mapdata.copy()
            newmap = (label_array > 0) * (mapdata * binmap)
            return newmap, labels - ct_remove
        return labels - ct_remove + 1

    means = measurements.mean(mapdata * binmap, label_array, range(labels + 1))
    freq, bins = np.histogram(sizes[1:], 20)

    m_array = np.zeros(len(sizes))
    ct_remove = 0
    for i in range(len(freq)):
        f = freq[i]
        s2 = bins[i + 1]
        s1 = bins[i]
        p_size = float(f) / float(np.sum(freq))
        if p_size > prob:
            m_array = m_array + (
                (sizes >= s1)
                & (sizes < s2)
                & (
                    means
                    < (
                        float(map_threshold)
                        + 0.35 * (np.amax(mapdata) - float(map_threshold))
                    )
                )
            )
            ct_remove += 1
    m_array = m_array > 0
    remove_points = m_array[label_array]

    label_array[remove_points] = 0
    if inplace:
        mapdata[:] = (label_array > 0) * (mapdata * binmap)
    else:
        newmap = mapdata.copy()
        newmap = (label_array > 0) * (mapdata * binmap)
        return newmap, labels - ct_remove
    return labels - ct_remove


def calc_diffmap(
    emmap1,
    emmap2,
    res1,
    res2,
    grid1_info,
    grid2_info,
    lpfilt_pre=False,
    lpfilt_post=False,
    refscl=False,
    randsize=0.1,
    flag_dust=False,
    cyc=0,
    verbose=0,
    fft_obj=None,
    ifft_obj=None,
):
    """
    Calculate difference map, rewritten from Agnel's difference map calculation in CCPEM TEMpy

    """
    t1 = 2.0 if res1 > 4.0 else 2.5
    t2 = 2.0 if res2 > 4.0 else 2.5
    # convert data to np.array if necesary
    # if isinstance(emmap_1, gemmi.FloatGrid):
    #    emmap1 = np.array(emmap_1, copy=False, dtype=np.float32)
    # if isinstance(emmap_2, gemmi.FloatGrid):
    #    emmap2 = np.array(emmap_2, copy=False, dtype=np.float32)

    # get contours
    c1 = calculate_map_contour(emmap1, sigma_factor=t1)
    c2 = calculate_map_contour(emmap2, sigma_factor=t2)
    samegrid = False
    # check same grid dimensions
    try:
        assert np.broadcast(emmap1.shape, emmap2.shape)
        assert compare_tuple(grid1_info.grid_start, grid2_info.grid_start)
        assert compare_tuple(grid1_info.voxel_size, grid2_info.voxel_size)
        samegrid = True
    except AssertionError:
        samegrid = False

    # check spacing along all axes
    if (
        grid1_info.voxel_size[0] != grid1_info.voxel_size[1]
        or grid1_info.voxel_size[1] != grid1_info.voxel_size[2]
    ):
        samegrid = False
    if (
        grid2_info.voxel_size[0] != grid2_info.voxel_size[1]
        or grid2_info.voxel_size[1] != grid2_info.voxel_size[2]
    ):
        samegrid = False

    # global scaling of amplitudes
    if samegrid:
        diff1, diff2 = amplitude_match(
            emmap1,
            emmap2,
            grid1_info,
            grid2_info,
            reso=max(res1, res2),
            lpfilt_pre=lpfilt_pre,
            lpfilt_post=lpfilt_post,
            ref_scale=refscl,
            fft_obj=fft_obj,
            ifft_obj=ifft_obj,
        )
        # min of minimas of two scaled maps
        min1 = diff1.min()
        min2 = diff2.min()
        min_scaled_maps = min(min1, min2)
        # shift to positive values
        if min_scaled_maps < 0.0:
            # make values non zero
            min_scaled_maps = min_scaled_maps + 0.05 * min_scaled_maps
            diff1[:] = diff1 + float(-min_scaled_maps)
            diff2[:] = diff2 + float(-min_scaled_maps)

        # calculate difference map
        print("Calculating difference map")
        # store scaled maps
        scaledmap1 = diff1.copy()
        scaledmap2 = diff2.copy()
        # difference map Fo - Fc (map1 - experiment, map2 - calculated from model)
        diff1 = diff1 - diff2
        diff2 = diff2 - diff1

        # dust filter
        if flag_dust:
            label_patches(diff1, 0.0, prob=randsize, inplace=True)
            label_patches(diff2, 0.0, prob=randsize, inplace=True)
            diff1[:] = diff1 * (diff1 > 0.0)
            diff2[:] = diff2 * (diff2 > 0.0)

        return scaledmap1, scaledmap2, diff1


def make_map_cubic(mapin, grid_info):
    max_len = -1
    max_ind = -1
    for i in range(0, 3):
        if grid_info.grid_shape[i] > max_len:
            max_len = grid_info.grid_shape[i]
            max_ind = i

    max_celldim = mapin.grid.unit_cell.parameters[max_ind]
    new_shape = np.array([max_len, max_len, max_len])
    map_data = resample_data_by_boxsize(mapin.grid, [max_len, max_len, max_len])
    apix = float(max_celldim / max_len)
    new_voxel_size = np.array([apix, apix, apix])
    new_gridinfo = GridInfo(
        new_shape, grid_info.grid_start, new_shape, new_voxel_size, grid_info.origin
    )
    newmap = gemmi.Ccp4Map()
    newmap.grid = gemmi.FloatGrid(map_data)
    newmap.grid.unit_cell.set(
        max_celldim,
        max_celldim,
        max_celldim,
        mapin.grid.unit_cell.alpha,
        mapin.grid.unit_cell.beta,
        mapin.grid.unit_cell.gamma,
    )
    newmap.grid.spacegroup = gemmi.SpaceGroup('P1')
    return newmap, new_gridinfo


if __name__ == '__main__':
    from pysheetbend.utils import fileio
    from pysheetbend.utils.dencalc import calculate_density_with_boxsize

    mapin = '/home/swh514/Projects/data/EMD-3488/map/emd_3488.map'
    # pdbin = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1_cryst1.ent'
    pdbin = '/home/swh514/Projects/work_and_examples/shiftfield/EMD-3488/4angs_lowres/translate_4angxyz.pdb'
    m, gridinfo = fileio.read_map(mapin)
    model, hetatm = fileio.get_structure(pdbin)
    apix0 = gridinfo.voxel_size
    nyquist_res = np.amax(2.0 * apix0)
    min_d = np.amax(apix0)
    res = 3.2
    if res > 0.0 and res > nyquist_res:
        samp_rate = res / (2 * min_d)  # so that the grid size matches original
    else:
        samp_rate = 1.5  # default; CLIPPER's oversampling parameter
    cmap = calculate_density_with_boxsize(model, res, samp_rate, gridinfo.grid_shape)
    fftobj = sf_util.plan_fft(gridinfo)
    ifftobj = sf_util.plan_ifft(gridinfo)

    scm1, scm2, dmap = calc_diffmap(
        m.grid,
        cmap,
        res,
        res,
        gridinfo,
        gridinfo,
        lpfilt_pre=True,
        lpfilt_post=False,
        refscl=False,
        randsize=0.1,
        flag_dust=False,
        verbose=6,
        fft_obj=fftobj,
        ifft_obj=ifftobj,
    )

    fileio.write_map_as_MRC(
        grid_data=scm1,
        unitcell=m.grid.unit_cell.parameters,
        outpath='scale_emd3488.mrc',
    )
    fileio.write_map_as_MRC(
        grid_data=scm2,
        unitcell=m.grid.unit_cell.parameters,
        outpath='scale_calcmap.mrc',
    )
    fileio.write_map_as_MRC(
        grid_data=dmap,
        unitcell=m.grid.unit_cell.parameters,
        outpath='diffmap.mrc',
    )
