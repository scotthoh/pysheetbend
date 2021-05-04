# Utily functions for shiftfield code
# S.W.Hoh, University of York, 2020
#%%
from scipy import signal
import math
import numpy as np
from mpl_toolkits import mplot3d
from timeit import default_timer as timer
'''from TEMPy.math.vector import Vector
from TEMPy.maps.map_parser import MapParser as mp
from TEMPy.protein.structure_blurrer import StructureBlurrer
'''

from TEMPy.MapParser import MapParser as mp
from TEMPy.Vector import Vector as Vector
from TEMPy.StructureBlurrer import StructureBlurrer
from TEMPy.EMMap import Map

try:
    import pyfftw
    pyfftw_flag = True
except ImportError:
    pyfftw_flag = False
import sys
if sys.version_info[0] > 2:
    from builtins import isinstance


def filter_winsize_1Dto3D(w):
    """
    Convert a 1D filtering kernel to 3D
    Arguments:
    *window function*
      a window function; e.g signal.windows.hann(window_size); window_size is an Int
    Returns:
      filtering kernel in 3D array
    """
    w_size = w.shape[0]
    #print('wsize', w_size)
    w1 = np.outer(np.ravel(w), np.ravel(w))
    #print('w1', w1)
    win1 = np.tile(w1, np.hstack([w_size, 1, 1]))
    #print('win1', win1)
    w2 = np.outer(np.ravel(w), np.ones([1, w_size]))
    #print('w2', w2)
    win2 = np.tile(w2, np.hstack([w_size, 1, 1]))
    #print('win2', win2)
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    #print('win2_2', win2)
    win3 = np.multiply(win1, win2)
    #print('win3', win3)
    return win3


#radius = 10
#window_size = math.floor((radius*2) / apix)
#if not window_size % 2: # make window size an odd int
#    window_size = window_size + 1
#
#win = winsize_1Dto3D(signal.windows.hann(window_size))
#print(win)

#%%

def apply_filter_hann(fullmap, radius, apix):
    """
    Apply hann filter to map data and return filtered data.
    Similar to the radial filter in CLIPPER
    Arguments:
    *fullmap*
      map data
    *radius*
      radius cutoff
    *apix*
      pixel size in angstrom/pixel
      can be single float or array of float pixel size in [x,y,z] direction
    """
    #rad = 7  # angstrom (4x the current refinement reso, cycles progressive increas 6 to 3)
    #24 - 12 angstrom at the end
    #6A data 24 radis , step 3A, 12A radius
    window_size = math.floor((radius*2) / apix)
    if not window_size % 2: # make window size an odd int
        window_size = window_size + 1

    win = filter_winsize_1Dto3D(signal.windows.hann(window_size))
    sum_n = np.sum(win)

    filtmap = signal.convolve(fullmap, win, mode='same', method='fft') / sum_n

    return filtmap


#%%
def mapGridPositions_radius(densMap, atom, gridtree, radius):
    """
    Returns the indices of the nearest pixels within the radius
    specified to an atom as a list.
    Adapted from TEMPy mapGridPositions
    Arguments:Map
    *densMap*
      Map instance the atom is to be placed on.
    *atom*
      Atom instance.
    *gridtree*
      KDTree of the map coordinates (absolute cartesian)
    *radius*
      radius from the atom coords for nearest pixels
    """
    origin = densMap.origin
    apix = densMap.apix
    x_pos = int(round((atom.x - origin[0]) / apix, 0))
    y_pos = int(round((atom.y - origin[1]) / apix, 0))
    z_pos = int(round((atom.z - origin[2]) / apix, 0))
    if((densMap.x_size() >= x_pos >= 0) and (densMap.y_size() >= y_pos >= 0)
       and (densMap.z_size() >= z_pos >= 0)):
        # search all points within radius of atom
        list_points = gridtree.query_ball_point([
          atom.x, atom.y, atom.z], radius)
        return list_points  # ,(x_pos, y_pos, z_pos)
    else:
        print('Warning, atom out of map box')
        return []

def make_atom_overlay_map1_rad(densMap, prot, gridtree, rad):
    """
    Returns a Map instance with atom locations recorded on
    the voxel within radius with a value of 1
    """
    densMap = densMap.copy()
    densMap.fullMap = densMap.fullMap * 0.0
    # use mapgridpositions to get the points within radius. faster and efficient
    # this works. resample map before assigning density
    for atm in prot:
        # get list of nearest points of an atom
        points = mapGridPositions_radius(densMap, atm, gridtree[0], rad)
        for ind in points:
            pos = gridtree[1][ind]  # real coordinates of the index
            # initialise AtomShapeFn object with atom
            # get 3D coordinates from map grid position
            # coord_pos = mapgridpos_to_coord(pos, newMap, False)
            # calculate electron density from 3D coordinates and
            # set to the map grid position
            # p_z=int(pos[2]-(newMap.apix/2.0))
            # p_y=int(pos[1]-(newMap.apix/2.0))
            # p_x=int(pos[0]-(newMap.apix/2.0))
            p_z = pos[2]
            p_y = pos[1]
            p_x = pos[0]

            densMap.fullMap[p_z, p_y, p_x] = 1.0
    return densMap


def mapgridpos_to_coord(pos, densMap, ccpem_tempy=False):
    """
    Returns the 3D coordinates of the map grid position given
    Argument:
    *pos*
      map grid position
    *densMap*
      Map instance the atom is placed on
    """

    origin = densMap.origin
    apix = densMap.apix
    #midapix = apix/2.0
    
    #if not ccpem_tempy:
    x_coord = (pos[0] * apix) + origin[0]
    y_coord = (pos[1] * apix) + origin[1]
    z_coord = (pos[2] * apix) + origin[2]
#else:
    #    x_coord = (pos[0] - midapix) * apix + origin[0]
    #    y_coord = (pos[1] - midapix) * apix + origin[1]
    #    z_coord = (pos[2] - midapix) * apix + origin[2]

    return (x_coord, y_coord, z_coord)


class grid_dim:
    def __init__(self, densMap):
        self.grid_sam = densMap.fullMap.shape
        self.g_reci = (densMap.z_size(), densMap.y_size(), densMap.x_size()//2+1)
        self.g_real = (self.g_reci[0], self.g_reci[1], int(self.g_reci[2]-1)*2)  #int((g_reci_[2]-1)*2))
        self.g_half = (densMap.z_size()//2, densMap.y_size()//2, densMap.x_size()//2+1)


def plan_fft(gridshape):
    output_shape = gridshape.g_reci
    #.shape[:len(data_r.shape)-1] + \
    #                            (data_r.shape[-1]//2 + 1,)

    try:
        if not pyfftw_flag:
            raise ImportError
        #start = timer()
        input_arr = pyfftw.empty_aligned(gridshape.grid_sam,
                                            dtype='float64', n=16)
        #end = timer()
        #print(end-start)
        #start = timer()
        output_arr = pyfftw.empty_aligned(output_shape,
                                            dtype='complex128', n=16)
        #end = timer()
        #print(end-start)
        # fft planning,
        fft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_FORWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
    except:
        print('Not running')
    
    return fft


def plan_ifft(gridshape):
    output_shape = gridshape.g_real
    #data_c.shape[:len(data_c.shape)-1] + \
    #                            ((data_c.shape[-1] - 1)*2,)

    try:
        if not pyfftw_flag:
            raise ImportError
        #start = timer()
        input_arr = pyfftw.empty_aligned(gridshape.g_reci,
                                         dtype='complex128', n=16)
        #end = timer()
        #print(end-start)
        #start = timer()
        output_arr = pyfftw.empty_aligned(output_shape,
                                          dtype='float64', n=16)
        #end = timer()
        #print(end-start)
        # fft planning,
        ifft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_BACKWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
    except:
        print('Not running')
    
    return ifft


def fourier_transform(data_r, scale, conj=False):
    
    #grid_size = data_r.size
    output_shape = data_r.shape[:len(data_r.shape)-1] + \
                                (data_r.shape[-1]//2 + 1,)
    
    try:
        if not pyfftw_flag:
            raise ImportError
        #start = timer()
        input_arr = pyfftw.empty_aligned(data_r.shape,
                                         dtype='float64', n=16)
        #end = timer()
        #print(end-start)
        #start = timer()
        output_arr = pyfftw.empty_aligned(output_shape,
                                          dtype='complex128', n=16)
        #end = timer()
        #print(end-start)
        # fft planning,
        fft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_FORWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
        input_arr[:, :, :] = data_r[:, :, :]
        fft.update_arrays(input_arr, output_arr)
        #start = timer()
        fft()
        #end = timer()
        #print(end-start)
    except:
        print("not running")
    # scale?
    #s = float(scale) / grid_size
    #print('fft scale, ', s)
    if conj:
        output_arr[:,:,:] = (output_arr[:,:,:]).conjugate()
    '''if conj:
        for oz in range(0, output_shape[0]):
            for oy in range(0, output_shape[1]):
                for ox in range(0, output_shape[2]):
                    output_arr[oz, oy, ox] = (output_arr[oz, oy, ox]).conjugate()
    '''
    #n = int(np.prod(output_shape))
    #for j in range(0, n):
    #    output_arr[j] = (s * output_arr[j]).conjugate()

    return output_arr


def inv_fourier_transform(data_c, scale, grid_reci, conj=False):
    # grid_size = data_c.map_size()
    #s = float(scale)
    #n = int(np.prod(grid_reci))
    if conj:
        data_c[:,:,:] = (data_c[:,:,:]).conjugate()
        '''
        for z in range(0, grid_reci[0]):
            for y in range(0, grid_reci[1]):
                for x in range(0, grid_reci[2]):
                    data_c[z, y, x] = (data_c[z, y, x]).conjugate()
        '''
    output_shape = data_c.shape[:len(data_c.shape)-1] + \
                                ((data_c.shape[-1] - 1)*2,)

    try:
        if not pyfftw_flag:
            raise ImportError
        #start = timer()
        input_arr = pyfftw.empty_aligned(data_c.shape,
                                         dtype='complex128', n=16)
        #end = timer()
        #print(end-start)
        #start = timer()
        output_arr = pyfftw.empty_aligned(output_shape,
                                          dtype='float64', n=16)
        #end = timer()
        #print(end-start)
        # fft planning,
        ifft = pyfftw.FFTW(input_arr, output_arr, direction='FFTW_BACKWARD', axes=(0,1,2), flags=['FFTW_ESTIMATE'])
        input_arr[:, :, :] = data_c[:, :, :]
        ifft.update_arrays(input_arr, output_arr)
        #start = timer()
        ifft()
        #end = timer()
        #print(end-start)
    except:
        print('not running')

    return output_arr


def cor_mod(a, b):
    c = np.fmod(a, b)
    if (c < 0):
        c += b
    return int(c)


def hkl_c(c, ch, g):
    # z, y, x; return vector(x, y ,z)
    cv = Vector(int(c[2]), int(c[1]), int(c[0]))
    chv = Vector(int(ch[2]), int(ch[1]), int(ch[0]))
    # cv = Vector(int(c[0]), int(c[1]), int(c[2]))
    # chv = Vector(int(ch[0]), int(ch[1]), int(ch[2]))
    v1 = cv + chv
    m1 = Vector(cor_mod(v1.x, g[2]),
                cor_mod(v1.y, g[1]),
                cor_mod(v1.z, g[0]))

    return m1 - chv


def eightpi2():
    """
    Returns 8*pi*pi
    """
    return (8.0 * np.pi * np.pi)

def u2b(u_iso):
    """
    Returns the isotropic B-value
    Argument:
    *u_iso*
      isotropic U-value
    """
    return u_iso * eightpi2()

def map_box_vol(densMap):
    return densMap.x_size() * densMap.y_size() * densMap.z_size()


class Cell:
    """
    Cell object
    """
    def __init__(self, a, b, c, alpha, beta, gamma):
        """
        Arguments:
        a,b,c : Cell dimensions in angstroms
        alpha, beta, gamma : Cell angles in radians or degrees
        (will convert automatically to radians during initialisation)
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if self.alpha > np.pi:
            self.alpha = np.deg2rad(alpha)
            print('deg2rad')
        if self.beta > np.pi:
            self.beta = np.deg2rad(beta)
        if self.gamma > np.pi:
            self.gamma = np.deg2rad(gamma)
        
        self.vol = a*b*c*math.sqrt(2.0*math.cos(alpha)*math.cos(beta)*math.cos(gamma)
                                    -math.cos(alpha)*math.cos(alpha)
                                    -math.cos(beta)*math.cos(beta)
                                    -math.cos(gamma)*math.cos(gamma)+1.0)

        # deal with null
        if self.vol <= 0.0:
            return

        # orthogonalisation + fractionisation matrices
        self.orthmat = np.identity(3)
        self.orthmat[0, 0] = a
        self.orthmat[0, 1] = b*np.cos(gamma)
        self.orthmat[0, 2] = c*np.cos(beta)
        self.orthmat[1, 1] = b*np.sin(gamma)
        self.orthmat[1, 2] = -c*np.sin(beta)*np.cos(self.alpha_star())
        self.orthmat[2, 2] = c*np.sin(beta)*np.sin(self.alpha_star())
        self.fracmat = np.linalg.inv(self.orthmat)

        # calculate metric_tensor
        self.realmetric = self.metric_tensor(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
        self.recimetric = self.metric_tensor(self.a_star(), self.b_star(), self.c_star(), 
                                             self.alpha_star(), self.beta_star(), self.gamma_star())

    def alpha_star(self):
        return np.arccos( (np.cos(self.gamma)*np.cos(self.beta)-np.cos(self.alpha)) /
                         (np.sin(self.beta)*np.sin(self.gamma)) )
    
    def beta_star(self):
        return np.arccos( (np.cos(self.alpha)*np.cos(self.gamma)-np.cos(self.beta)) /
                         (np.sin(self.gamma)*np.sin(self.alpha)) )

    def gamma_star(self):
        return np.arccos( (np.cos(self.beta)*np.cos(self.alpha)-np.cos(self.gamma)) /
                         (np.sin(self.alpha)*np.sin(self.beta)) )

    def a_star(self):
        return self.b*self.c*np.sin(self.alpha)/self.vol

    def b_star(self):
        return self.c*self.a*np.sin(self.beta)/self.vol
    
    def c_star(self):
        return self.a*self.b*np.sin(self.gamma)/self.vol

    def vol(self):
        return self.vol

    def a(self):
        return self.a
    
    def b(self):
        return self.b
    
    def c(self):
        return self.c

    def alpha(self):
        return self.alpha

    def beta(self):
        return self.beta
    
    def gamma(self):
        return self.gamma

    def metric_tensor(self, a, b, c, alp, bet, gam):
        m00 = a*a
        m11 = b*b
        m22 = c*c
        m01 = 2.0*a*b*np.cos(gam)
        m02 = 2.0*a*c*np.cos(bet)
        m12 = 2.0*b*c*np.cos(alp)
        return (m00, m11, m22, m01, m02, m12)

    def metric_reci_lengthsq(self, x, y, z):
        # v[0]*(v[0]*m00 + v[1]*m01 + v[2]*m02)
        # + v[1]*(v[1]*m11 + v[2]*m12) + v[2]*(v[2]*m22)
        return (x*(x*self.recimetric[0] + y*self.recimetric[3] + z*self.recimetric[4]) +
                y*(y*self.recimetric[1] + z*self.recimetric[5]) + z*(z*self.recimetric[2]))
'''
def interp_cubic(densmap, u, v, w):

    calculate value of map coordinate by third order (cubic) interpolation
    based on the surrounding 64 points. Re-coded from CLIPPER C++
    u0 = math.floor(u)
    v0 = math.floor(v)
    w0 = math.floor(w)

    cu1 = float(u - u0)
    cv1 = float(v - v0)
    cw1 = float(w - w0)

    cu0 = 1.0 - cu1
    cv0 = 1.0 - cv1
    cw0 = 1.0 - cw1

    cu = [0.0] * 5
    cu[0] = -0.5*cu1*cu0*cu0  # cubic spline coeffs: u
    cu[1] = cu0*(-1.5*cu1*cu1 + cu1 + 1.0)
    cu[2] = cu1*(-1.5*cu0*cu0 + cu0 + 1.0)
    cu[3] = -0.5*cu18cu1*cu0
    cv[0] = -0.5*cv1*cv0*cv0  # cubic spline coeffs: v
    cv[1] = cv0*(-1.5*cv1*cv1 + cv1 + 1.0)
    cv[2] = cv1*(-1.5*cv0*cv0 + cv0 + 1.0)
    cv[3] = -0.5*cv1*cv1*cv0
    cw[0] = -0.5*cw1*cw0*cw0  # cubic spline coeffs: w
    cw[1] = cw0*(-1.5*cw1*cw1 + cw1 + 1.0)
    cw[2] = cw1*(-1.5*cw0*cw0 + cw0 + 1.0)
    cw[3] = -0.5*cw1*cw1*cw0

    su = 0.0
    for j in range(0,4):
'''

# %%

def crop_mapgrid_points(x0, y0, z0, densmap, k=4):
    #SB=StructureBlurrer()
    #atm_point = SB.mapGridPosition(densmap, atom)
    # make a small box containing density data
    # of surrounding 64 map grid points of atom_point
    boxmap = Map(np.zeros((4, 4, 4)),
            [0, 0, 0],
            densmap.apix,
            'mapname',)

    # problem: should the atom point be at the corner of the box or the centre for interpolation
    # atm mapgridposition at corner
    for iz in range(0, 4):
        for iy in range(0, 4):
            for ix in range(0, 4):
                mx = ix + x0
                my = iy + y0
                mz = iz + z0
                boxmap.fullMap[iz, iy, ix] = densmap.fullMap[mz, my, mx]

    return boxmap  # cropped map, atm pos in cropped map


def get_lowerbound_posinnewbox(x, y, z):
    x0 = x - 1
    y0 = y - 1
    z0 = z - 1
    nbx = x - x0
    nby = y - y0
    nbz = z - z0

    return [x0, y0, z0], [nbx, nby, nbz]

#def xyz_indices(densmap):
#    xg, yg, zg = np.mgrid[0:densmap.x_size(),
#                          0:densmap.y_size(),
#                          0:densmap.z_size()]

def ind_hkl_to_coord(densmap):
    origin = densmap.origin
    apix = densmap.apix
    nz, ny, nx = densmap.fullMap.shape

    zy, yg, xg = np.mgrid[0:nz, 0:ny, 0:nz]
    zg = zg * apix + origin[2]
    yg = yg * apix + origin[1]
    xg = xg * apix + origin[0]

    indi = vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T

def maptree_1(densmap):
    #return gridtree and indices, z,y,x convention
    origin = densmap.origin
    apix = densmap.apix
    nz, ny, nx = densmap.fullMap.shape

    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    zg = zg * apix + origin[2]
    yg = yg * apix + origin[1]
    xg = xg * apix + origin[0]

    indi = np.vstack([zg.ravel(), yg.ravel(), xg.ravel()]).T

    try:
        from scipy.spatial import cKDTree
        gridtree = cKDTree(indi)
    except ImportError:
        try:
            from scipy.spatial import KDTree
            gridtree = KDTree(indi)
        except  ImportError:
            return
    
    zg, yg, xg = np.mgrid[0:nz, 0:ny, 0:nx]
    indi = np.vstack([zg.ravel(), yg.ravel(), xg.ravel()]).T
    return gridtree, indi

class radial_fltr:
    def __init__(self, radcyc, function, densmap):
        self.radius = radcyc
        self.function = function
        '''if function == 'step':
            self.function = 0
        elif function == 'linear':
            self.function = 1
        elif function == 'quadratic':
            self.function = 2
        '''
        # function = step, linear, quadratic
        # determine effective radius of radial function
        self.nrad = 1000
        self.drad = 0.25
        self.sum_r = [0.0]*self.nrad
        self.gridshape = grid_dim(densmap)

        for i in range(0, self.nrad):
            r = self.drad * (float(i) + 0.5)
            self.sum_r[i] = r*r*math.fabs(self.fltr(r))
        
        for i in range(1, self.nrad):
            self.sum_r[i] += self.sum_r[i-1]
        for i in range(0, self.nrad):
            if self.sum_r[i] > 0.99*self.sum_r[self.nrad-1]:
                break
        self.rad = self.drad*(float(i)+1.0)
        self.fltr_data_r = np.zeros(self.gridshape.grid_sam, dtype='float64')
        f000 = 0.0
        # z,y,x convention
        origin = np.array([densmap.origin[2], densmap.origin[1], densmap.origin[0]])
        if isinstance(densmap.apix, tuple):
            apix = np.array([densmap.apix[2], densmap.apix[1], densmap.apix[0]])
        else:    
            apix = np.array([densmap.apix, densmap.apix, densmap.apix])
        
        #g_half = (g_real[0]//2, g_real[1]//2, g_real[0]//2+1)
        #SB = StructureBlurrer()
        #gt = SB.maptree(densmap)
        gt = maptree_1(densmap)
        # filpping indices made things wrong because the 
        # the chronology of indices has changed.
        # is this true? pt1 = 001 and pt1 = 100 different
        #start = timer() # flipping indices takes about 0.4 sec 
        #indi = np.flip(gt[1], 1) # gt[1] indices is x,y,z , flip become z,y,x
        #end = timer()
        #print('flip indices ', end-start)
        
        gh = np.array([self.gridshape.g_half])
        
        start = timer()
        
        c = gt[1] + gh # self.gridshape.g_half
        end = timer()
        print('indi + halfgrid ', end-start)
        
        start = timer()
        c1 = self.cor_mod1(c, self.gridshape.grid_sam) - gh
        end = timer()

        print('cor mod ', end-start)
        
        start = timer()
        pos = c1[:]*apix+origin
        
        r = np.sqrt(np.sum(np.square(pos), axis=1))
        end = timer()
        print('indices get r ', end-start)
        #for i in range(len(r)):
        #    print(pos[i], c1[i], gt[1][i], r[i])
        start = timer()
        print('self.rad ', self.rad)
        r_ind = np.nonzero(r<self.rad)
        print(r_ind)
        #r = np.where(r < self.rad, self.fltr(r), 0.0)
        #r_ind = np.nonzero(r)
        
        end = timer()
        print('nonzero transpose ', end-start)

        start = timer()
        count=0
        for i in r_ind[0]:
            rf = self.fltr(r[i])
            f000 += rf
            #print(gt[1][i][0], gt[1][i][1], gt[1][i][2], rf)
            count+=1
            self.fltr_data_r[gt[1][i][0], gt[1][i][1], gt[1][i][2]] = rf #[i]
        end = timer()
        '''
        count = 0
        print('self.rad ', self.rad)
        for ind in range(0,len(gt[1])):
            xyz = gt[1][ind]
            c = hkl_c((xyz[2], xyz[1], xyz[0]), self.gridshape.g_half, self.gridshape.grid_sam)
            pos = mapgridpos_to_coord((c[0],c[1],c[2]),densmap)
            r = Vector(pos[0],pos[1],pos[2]).mod()
            #r = Vector(cz, cy, cx).mod()
            if r < self.rad:
                print((pos[2],pos[1],pos[0]), (c[2],c[1],c[0]), (xyz[2], xyz[1], xyz[0]), r)
                r = self.fltr(r)
                f000 += r
                print(r)
                #self.fltr_data_r[xyz[2], xyz[1], xyz[0]] = r
                count+=1
        
        end=timer()
        '''
        print('fill radial function map ', end-start)
        print('count ', count)
        
        self.scale = 1.0/f000
        print('scale, ', self.scale, ' f000, ', f000)

    def cor_mod1(self, a, b):
        c = np.fmod(a, b)
        d = np.transpose(np.nonzero(c<0))
        #d, e = np.nonzero(c<0)
        for i in d: #range(len(d)):
            c[i[0], i[1]] += b[i[1]]
            #c[i, j] += b[i]
        return c

    def fltr(self, r):
        if r < self.radius:
            if self.function == 2:
                return pow((1.0-r)/self.radius, 2)
            elif self.function == 1:
                return (1.0-r)/self.radius
            elif self.function == 0:
                return 1.0
        else:
            return 0.0
    

    def mapfilter(self, data_arr, fft_obj, ifft_obj):
        # 2 March
        # something wrong with map filter the center is incorrect
        #g = densmap.box_size()  # ZYX format, size of each direction
        #g_reci = (densmap.z_size(), densmap.y_size(), int(densmap.x_size()//2+1))
        #g_real = (g_reci[0], g_reci[1], (g_reci[2]-1)*2)
        data_r = np.zeros(self.gridshape.grid_sam, dtype='float64')
        data_r[:,:,:] = data_arr[:,:,:]
        fltr_input = np.zeros(self.gridshape.grid_sam, dtype='float64')
        fltr_input[:,:,:] = self.fltr_data_r[:,:,:]
        #boxvol = map_box_vol(densmap)
        #fltr_data_r = np.zeros(data_r.shape, dtype='float64')
        
        
        # determine effective radius of radial function
        '''nrad = 1000
        drad = 0.25
        sum_r = [0.0]*nrad
    
        for i in range(0, nrad):
            r = drad * (float(i) + 0.5)
            sum_r[i] = r*r*math.fabs(self.fltr(r))
        
        for i in range(1, nrad):
            sum_r[i] += sum_r[i-1]
        for i in range(0, nrad):
            if sum_r[i] > 0.99*sum_r[nrad-1]:
                break
        rad = drad*(float(i)+1.0)
        '''
        # fill the radial function map
        
        #g_half = (g[0]//2, g[1]//2, g[2]//2)
        #start = timer()
        #SB = StructureBlurrer()
        #gt = SB.maptree(densmap)
        #end = timer()
        #print('maptree ', end-start)
        
        
        '''
        for cz in range(0, g[0]):    # z
            for cy in range(0, g[1]):     # y
                for cx in range(0, g[2]):    # x
        '''                 
        
        # calc scale factor
        # either 1.0 or 1/f000(default)
        #scale = 1.0 / f000
        #print('scale, ', scale, ' f000, ', f000)
        # fft
        start = timer()
        #fltr_data_c = fourier_transform(fltr_data_r, 1.0, conj=True)
        fltr_data_c = np.zeros(self.gridshape.g_reci, dtype='complex128')
        data_c = np.zeros(self.gridshape.g_reci, dtype='complex128')
        
        fltr_data_c = fft_obj(fltr_input, fltr_data_c)
        fltr_data_c[:,:,:] = (fltr_data_c[:,:,:]).conjugate()
        end = timer()
        #rint('fft_fltdata ', end-start)
        start = timer()
        #data_c = fourier_transform(data_r, 1.0, conj=True)
        data_c = fft_obj(data_r, data_c)
        data_c[:,:,:] = (data_c[:,:,:]).conjugate()
        end = timer()
        #print('fftdata ', end-start)
        # do filter
        #for cz in range(0, g_reci[0]):    # z
        #    for cy in range(0, g_reci[1]):     # y
        #        for cx in range(0, g_reci[2]):    # x
        start = timer()
        data_c[:,:,:] = self.scale*data_c[:,:,:]*fltr_data_c[:,:,:]
        end = timer()
        #print('convolution ', end-start)
    
        # ifft
        #densmap_g_real_size = g_real[0]*g_real[1]*g_real[2]
        start = timer()
        #data_r = inv_fourier_transform(data_c, 1.0, g_reci, conj=True) # densmap_g_real_size/pow(boxvol,2)
        data_c[:,:,:] = (data_c[:,:,:]).conjugate()
        data_r = ifft_obj(data_c, data_r)
        end = timer()
        #print('ifft ', end-start)
        return data_r


def solve_linal(a, b):

    if a.shape[0] != a.shape[1]:
        #print("matrix not square")
        raise ValueError('matrix not square')
        #return(-1)
    if a.shape[0] != b.size:
        #print("matrix/vector mismatch")
        raise ValueError('matrix/vector mismatch')
        #return(-1)
    
    n = a.shape[0] #rows
    #a = copy.deepcopy(mat)
    #b = copy.deepcopy(vec)
    #solve for X by Gaussian elimination
    for i in range(0, n):
        #pick largest pivot
        j = i
        for k in range(i+1, n):
            if np.fabs(a[k, i]) > np.fabs(a[j, i]):
                j = k
        # swap rows
        for k in range(0, n):
            a[i,k], a[j,k] = a[j, k], a[i, k]
        b[i], b[j] = b[j], b[i]

        # perform elimination
        pivot = a[i, i]
        for j in range(0, n):
            if j != i:
                s = a[j, i] / pivot
                for k in range(i+1, n):
                    a[j, k] = a[j, k] - s*a[i, k]
                b[j] = b[j] - s*b[i]
    
    for i in range(0, n):
        b[i] /= a[i,i]
    
    return b