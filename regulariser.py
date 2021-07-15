#%%
from __future__ import print_function  # python 3 proof 
import gemmi
import os
from numpy import array, full, zeros, arccos, arctan2, rad2deg, deg2rad, sin,\
                  tan, fabs, sum as npsum, square as npsq, sqrt as npsqrt
from numpy import linalg, multiply, reshape
import concurrent.futures as CF
from multiprocessing import Process, Pool, Pipe

#from numpy.lib.function_base import gradient

def forces(topo):
    forces_list = []
    #print("chain_infos length : {0}".format(len(topo.chain_infos)))
    for chain_info in topo.chain_infos:
        #print("res_infos length : {0}".format(len(chain_info.res_infos)))
        for res_info in chain_info.res_infos:
            #print("{0} force length : {1}".format(res_info.res, len(res_info.forces)))
            for force in res_info.forces:
                if force.provenance == gemmi.Provenance.PrevLink:
                    forces_list.append(force)
                if force.provenance == gemmi.Provenance.Monomer:
                    forces_list.append(force)
    #print('before extra, len forces_list : {0}'.format(len(forces_list)))
    for link in topo.extras:
        for force in link.forces:
            forces_list.append(force)
    #print('after extra, len forces_list : {0}'.format(len(forces_list)))
    
    return forces_list


def get_rmsz(topo):
    forcelist = forces(topo)
    RMSbond = 0.0
    RMSangle = 0.0
    RMSplane = 0.0
    RMStorsion = 0.0
    sumbond = 0.0
    sumangle = 0.0
    sumplane = 0.0
    sumtorsion = 0.0
    cbond = 0
    cangle = 0
    ctorsion = 0
    cplane = 0
    #count = 0
    for force in forcelist:
        #print(count)
        #print(force.provenance, force.rkind, force.index)
        if force.rkind == gemmi.RKind.Bond:
            #print(force.rkind)
            t = topo.bonds[force.index]
            z = t.calculate_z()
            #print('Bond, {0}, {1}, {2}'.format(force.index, z, z*z))
            sumbond += z * z
            cbond += 1
        if force.rkind == gemmi.RKind.Angle:
            #print(force.rkind)
            t = topo.angles[force.index]
            z = t.calculate_z()
            #print('Angle, {0}, {1}, {2}'.format(force.index, z, z*z))
            sumangle += z * z
            cangle += 1
        if force.rkind == gemmi.RKind.Torsion:
            #print(force.rkind)
            t = topo.torsions[force.index]
            if t.restr.esd > 0.0:
                z = t.calculate_z()
                #print('Torsion, {0}, {1}, {2}'.format(force.index, z, z*z))
                sumtorsion += z * z
                ctorsion += 1
        if force.rkind == gemmi.RKind.Plane:
            #print(force.rkind)
            t = topo.planes[force.index]
            coeff = gemmi.find_best_plane(t.atoms)
            max_z = 0.0
            for atom in t.atoms:
                dist = gemmi.get_distance_from_plane(atom.pos, coeff)
                z = dist / t.restr.esd
                max_z = z if z > max_z else max_z
            #print('Plane, {0}, {1}, {2}'.format(force.index, max_z, max_z*max_z))
            sumplane += max_z * max_z
            cplane += 1
        #else:
        #    continue
        #count += 1
    RMSbond = npsqrt(sumbond/cbond)
    RMSangle = npsqrt(sumangle/cangle)
    RMStorsion = npsqrt(sumtorsion/ctorsion)
    RMSplane = npsqrt(sumplane/cplane)
    print('rmsbond : {0:.3f}, rmsangle : {1:.3f}, rmstorsion : {2:.3f}, rmsplane : {3:.3f}'
          .format(RMSbond, RMSangle, RMStorsion, RMSplane))
    print('nbond : {0}, nangle : {1}, ntorsion : {2}, nplane : {3}'
          .format(cbond, cangle, ctorsion, cplane))
    return (RMSbond, RMSangle, RMStorsion, RMSplane)


class BondRestraint:
    def __init__(self, index1, index2, ideal, sigma):
        self.index1 = index1
        self.index2 = index2
        self.ideal = ideal
        self.sigma = sigma

class AngleRestrains:
    def __init__(self, index1, index2, index3, ideal, sigma):
        self.index1 = index1
        self.index2 = index2
        self.index3 = index3
        self.ideal = ideal
        self.sigma = sigma

class TorsionRestraint:
    def __init__(self, index1, index2, index3, index4, ideal, sigma, period):
        self.index1 = index1
        self.index2 = index2
        self.index3 = index3
        self.index4 = index4
        self.ideal = ideal
        self.sigma = sigma
        self.period = period

class PlaneRestraint:
    def __init__(self, indices, sigma):
        self.indices = indices
        self.sigma = sigma


class TargetFunction():
    def __init__(self, topo, errors):
        self.atom_indices = {}  # key:Gemmi.Atom, value:int(index)
        self.coords = []  # array([], dtype='float64')
        self.n = 0

        forcelist = forces(topo)
        #print('setup target function, force list len {0}'.format(len(forcelist)))
        self.__bond_restr = []
        self.__angle_restr = []
        self.__torsion_restr = []
        self.__plane_restr = []
        for force in forcelist:
            if force.rkind == gemmi.RKind.Bond:
                bond = topo.bonds[force.index]
                index1 = self.atom_index(bond.atoms[0])
                index2 = self.atom_index(bond.atoms[1])
                self.__bond_restr.append(BondRestraint(index1,
                                                       index2,
                                                       bond.restr.value,
                                                       bond.restr.esd))
            if force.rkind == gemmi.RKind.Angle:
                angle = topo.angles[force.index]
                index1 = self.atom_index(angle.atoms[0])
                index2 = self.atom_index(angle.atoms[1])
                index3 = self.atom_index(angle.atoms[2])
                self.__angle_restr.append(AngleRestrains(index1,
                                                         index2,
                                                         index3,
                                                         angle.restr.value,
                                                         angle.restr.esd))
            if force.rkind == gemmi.RKind.Torsion:
                torsion = topo.torsions[force.index]
                index1 = self.atom_index(torsion.atoms[0])
                index2 = self.atom_index(torsion.atoms[1])
                index3 = self.atom_index(torsion.atoms[2])
                index4 = self.atom_index(torsion.atoms[3])
                esd = torsion.restr.esd if torsion.restr.esd > 0 else 1
                prd = torsion.restr.period if torsion.restr.period > 0 else 1
                self.__torsion_restr.append(TorsionRestraint(index1,
                                                             index2,
                                                             index3,
                                                             index4,
                                                             torsion.restr.value,
                                                             esd,
                                                             prd))
            if force.rkind == gemmi.RKind.Plane:
                plane = topo.planes[force.index]
                indices = []
                for atom in plane.atoms:
                    index = self.atom_index(atom)
                    indices.append(index)
                self.__plane_restr.append(PlaneRestraint(indices,
                                                         plane.restr.esd))

        self.coords = array(self.coords)
        self.n = self.coords.size
        self.origin_weights = full(self.n, 0.001, dtype='float64')
        for atom, index in self.atom_indices.items():
            sigma = errors.get(atom)
            if sigma == 0.0:
                self.origin_weights[index] = 1.0
            else:
                #print(f'{atom}, {sigma}')
                self.origin_weights[index] = 1.0 / (sigma * sigma)
            self.origin_weights[index + 1] = self.origin_weights[index]
            self.origin_weights[index + 2] = self.origin_weights[index]

    def __call__(self, points):
        # call function for score and gradient calculation
        # use Process from multiprocess
        score = 0.0
        grad = zeros(self.n, dtype='float64')
        o_rec, o_send = Pipe()
        b_rec, b_send = Pipe()
        a_rec, a_send = Pipe()
        t_rec, t_send = Pipe()
        p_rec, p_send = Pipe()
        origin = Process(target=self.origin_distortion_loop,
                         args=(points, o_send))
        bond = Process(target=self.bond_distortion_loop,
                       args=(points, b_send))
        angle = Process(target=self.angle_distortion_loop,
                        args=(points, a_send))
        torsion = Process(target=self.torsion_distortion_loop,
                          args=(points, t_send))
        plane = Process(target=self.plane_distortion_loop,
                        args=(points, p_send))
        origin.start()
        bond.start()
        angle.start()
        torsion.start()
        plane.start()
        #print(f'score, grad = {c1_rec.recv()}')
        s_o, g_o = o_rec.recv()
        s_b, g_b = b_rec.recv()
        s_a, g_a = a_rec.recv()
        s_t, g_t = t_rec.recv()
        s_p, g_p = p_rec.recv()
        origin.join()
        bond.join()
        angle.join()
        torsion.join()
        plane.join()
        score = s_o + s_b + s_a + s_t + s_p
        grad = g_o + g_b + g_a + g_t + g_p
        return score, grad

    #def origin_distortion_loop(self, points):
    def origin_distortion_loop(self, points, c):
        score = 0.0
        grad = zeros(self.n, dtype='float64')
        for i in range(self.n):
            # use = instead of += to reset grad as origin distortions are
            # calculated first
            s, grad[i] = self.origin_distortion(i, grad, points)
            # print(f'{i}, {s}, {grad[i]}')
            score += s
        c.send([score, grad])
        c.close()
        return score, grad
    
    #def bond_distortion_loop(self, points):
    def bond_distortion_loop(self, points, c):
        score = 0.0
        grad = zeros(self.n, dtype='float64')
        for bond in self.__bond_restr:
            s, grad = self.bond_distortion(bond, grad, points)
            score += s
        c.send([score, grad])
        c.close()
        return score, grad

    #def angle_distortion_loop(self, points):
    def angle_distortion_loop(self, points, c):
        score = 0.0
        grad = zeros(self.n, dtype='float64')
        for angle in self.__angle_restr:
            s, grad = self.angle_distortion(angle, grad, points)
            score += s
        c.send([score, grad])
        c.close()
        return score, grad
    
    #def torsion_distortion_loop(self, points):
    def torsion_distortion_loop(self, points, c):
        score = 0.0
        grad = zeros(self.n, dtype='float64')
        for torsion in self.__torsion_restr:
            s, grad = self.torsion_distortion(torsion, grad, points)
            score += s
        c.send([score, grad])
        c.close()
        return score, grad
    
    #def plane_distortion_loop(self, points):
    def plane_distortion_loop(self, points, c):
        score = 0.0
        grad = zeros(self.n, dtype='float64')
        for plane in self.__plane_restr:
            s, grad = self.plane_distortion(plane, grad, points)
            score += s
        c.send([score, grad])
        c.close()
        return score, grad
    '''
    def __call__(self, points):
        score = 0.0
        #grad = array([], dtype='float64')
        grad = zeros(self.n, dtype='float64')
        #print('origin distortion:')
        for i in range(self.n):
            # use = instead of += to reset grad as origin distortions are
            # calculated first
            s, grad[i] = self.origin_distortion(i, grad, points)
            #print(f'{i}, {s}, {grad[i]}')
            score += s
        #print('bond distortion:')
        for bond in self.__bond_restr:
            s, grad = self.bond_distortion(bond, grad, points)
            score += s
            #print(f'{s:.4f}, ', end='')
            #print(f'{grad[bond.index1]:.4f}, {grad[bond.index1+1]:.4f}, ',
            #      end='')
            #print(f'{grad[bond.index1+2]:.4f}| {grad[bond.index2]:.4f}, ',
            #      end='')
            #print(f'{grad[bond.index2+1]:.4f}, {grad[bond.index2+2]:.4f}')
            
        #print('angle distortion:')
        for angle in self.__angle_restr:
            s, grad = self.angle_distortion(angle, grad, points)
            score += s
            print(f'{s:.4f}, ', end='')
            print(f'{grad[angle.index1]:.4f}, {grad[angle.index1+1]:.4f}, ',
                  end='')
            print(f'{grad[angle.index1+2]:.4f}| {grad[angle.index2]:.4f}, ',
                  end='')
            print(f'{grad[angle.index2+1]:.4f}, {grad[angle.index2+2]:.4f}| ',
                  end='')
            print(f'{grad[angle.index3]:.4f}, {grad[angle.index3+1]:.4f}, ',
                  end='')
            print(f'{grad[angle.index3+2]:.4f}')
            
        #print('torsion distortion:')
        count = 0
        for torsion in self.__torsion_restr:
            s, grad = self.torsion_distortion(torsion, grad, points)
            score += s
            count += 1
            #print(f'{count}')
            print(f'{s:.4f}, ', end='')
            print(f'{grad[torsion.index1]:.4f}, {grad[torsion.index1+1]:.4f}, ',
                  end='')
            print(f'{grad[torsion.index1+2]:.4f}| {grad[torsion.index2]:.4f}, ',
                  end='')
            print(f'{grad[torsion.index2+1]:.4f}, {grad[torsion.index2+2]:.4f}| ',
                  end='')
            print(f'{grad[torsion.index3]:.4f}, {grad[torsion.index3+1]:.4f}, ',
                  end='')
            print(f'{grad[torsion.index3+2]:.4f}| {grad[torsion.index4]:.4f}, ',
                  end='')
            print(f'{grad[torsion.index4+1]:.4f}, {grad[torsion.index4+2]:.4f}')
            
        #print('plane distortion:')
        for plane in self.__plane_restr:
            s, grad = self.plane_distortion(plane, grad, points)
            score += s
            print(f'{s:.4f}, ', end='')
            for i in range(len(plane.indices)):
                print(f'{grad[i]:.4f}, {grad[i+1]:.4f}, {grad[i+2]:.4f}| ',
                      end='')
            print('\n')
        
        return score, grad
    '''
    def atom_index(self, atom):
        if atom not in self.atom_indices:
            self.atom_indices[atom] = len(self.coords)
            self.coords.append(atom.pos.x)
            self.coords.append(atom.pos.y)
            self.coords.append(atom.pos.z)
        return self.atom_indices[atom]
    
    def get_param(self):
        return self.n

    def origin_distortion(self, i, grad, points):
        diff = points[i] - self.coords[i]
        diffsq = diff * diff
        grad[i] = 2.0 * self.origin_weights[i] * diff
        score = self.origin_weights[i] * diffsq
        return score, grad[i]
    
    def bond_distortion(self, bond, grad, points):
        '''
        Calculate bond distortion scores
        #maybe can make the points(array) to shape(n,3) instead of (n,)
        #Score_bond = sum_all_bonds((1/sigma^2)(b_i - b_0)^2 ; b_i = ith angle, b0 ideal angle
        #angle contributed by atom k,l,m

        #gradient = dAi/dXm = 2*(1/sigma^2)[b_i - b0](Xm - Xk)/b_i
        #grad f(x,y,z) = = const * ((delf/delx) + (delf/dely) + (delf/delz))/b_i
        Arguments:
        *bond*
            Gemmi.Bond
        *grad*
            1D array of gradient of size n
        *points*
            1D flatten array of x,y,z coordinates of size n
        '''
        p1 = gemmi.Position(points[bond.index1], points[bond.index1 + 1],
                            points[bond.index1 + 2])
        p2 = gemmi.Position(points[bond.index2], points[bond.index2 + 1],
                            points[bond.index2 + 2])

        length = p1.dist(p2)
        length = 0.1 if length < 0.1 else length
        #length = npsqrt(lengthsq)
        diff = length - bond.ideal
        diffsq = diff * diff
        weight = 1.0 / (bond.sigma * bond.sigma)
        coeff = 2 * weight * diff / length
        grad[bond.index1] += coeff * (p1.x - p2.x)
        grad[bond.index1 + 1] += coeff * (p1.y - p2.y)
        grad[bond.index1 + 2] += coeff * (p1.z - p2.z)
        grad[bond.index2] += coeff * (p2.x - p1.x)
        grad[bond.index2 + 1] += coeff * (p2.y - p1.y)
        grad[bond.index2 + 2] += coeff * (p2.z - p1.z)
        return weight*diffsq, grad


    def angle_distortion(self, angle, grad, points):
        '''
        Calculate angle distortion scores
        
        Arguments:
        *angle*
            Gemmi.angle
        *grad*
            1D array of gradient of size n
        *point*
            1D flatten array of x,y,z coordinates of size n
        '''
        p1 = gemmi.Position(points[angle.index1], points[angle.index1 + 1],
                            points[angle.index1 + 2])
        p2 = gemmi.Position(points[angle.index2], points[angle.index2 + 1],
                            points[angle.index2 + 2])
        p3 = gemmi.Position(points[angle.index3], points[angle.index3 + 1],
                            points[angle.index3 + 2])
        vec1 = p1 - p2
        vec2 = p3 - p2
        len1 = p1.dist(p2)
        len2 = p3.dist(p2)
        if len1 < 0.01:
            len1 = 0.01
            vec1 = gemmi.Position(0.01, 0.01, 0.01)
        if len2 < 0.01:
            len2 = 0.01
            vec2 = gemmi.Position(0.01, 0.01, 0.01)

        cos_theta = vec1.dot(vec2) / (len1 * len2)
        cos_theta = -1 if cos_theta < -1 else cos_theta
        cos_theta = 1 if cos_theta > 1 else cos_theta
        theta = arccos(cos_theta)
        theta = 0.001 if theta < 0.001 else theta
        diff = rad2deg(theta) - angle.ideal
        diffsq = diff * diff
        weight = 1 / (angle.sigma * angle.sigma)
        inv_1sq = 1 / (len1 * len1)
        inv_2sq = 1 / (len2 * len2)
        inv_12 = 1 / (len1 * len2)
        x1_contrib = cos_theta*(p2.x-p1.x)*inv_1sq + (p3.x-p2.x)*inv_12
        y1_contrib = cos_theta*(p2.y-p1.y)*inv_1sq + (p3.y-p2.y)*inv_12
        z1_contrib = cos_theta*(p2.z-p1.z)*inv_1sq + (p3.z-p2.z)*inv_12
        x3_contrib = cos_theta*(p2.x-p3.x)*inv_2sq + (p1.x-p2.x)*inv_12
        y3_contrib = cos_theta*(p2.y-p3.y)*inv_2sq + (p1.y-p2.y)*inv_12
        z3_contrib = cos_theta*(p2.z-p3.z)*inv_2sq + (p1.z-p2.z)*inv_12
        x2term1 = -cos_theta*(p2.x-p1.x)*inv_1sq \
                  - cos_theta*(p2.x-p3.x)*inv_2sq
        y2term1 = -cos_theta*(p2.y-p1.y)*inv_1sq \
                  - cos_theta*(p2.y-p3.y)*inv_2sq
        z2term1 = -cos_theta*(p2.z-p1.z)*inv_1sq \
                  - cos_theta*(p2.z-p3.z)*inv_2sq
        x2term2 = (-(p1.x-p2.x)-(p3.x-p2.x))*inv_12
        y2term2 = (-(p1.y-p2.y)-(p3.y-p2.y))*inv_12
        z2term2 = (-(p1.z-p2.z)-(p3.z-p2.z))*inv_12
        x2_contrib = x2term1 + x2term2
        y2_contrib = y2term1 + y2term2
        z2_contrib = z2term1 + z2term2
        coeff = rad2deg(-2 * weight * diff / sin(theta))
        grad[angle.index1] += coeff * x1_contrib
        grad[angle.index1 + 1] += coeff * y1_contrib
        grad[angle.index1 + 2] += coeff * z1_contrib
        grad[angle.index3] += coeff * x3_contrib
        grad[angle.index3 + 1] += coeff * y3_contrib
        grad[angle.index3 + 2] += coeff * z3_contrib
        grad[angle.index2] += coeff * x2_contrib
        grad[angle.index2 + 1] += coeff * y2_contrib
        grad[angle.index2 + 2] += coeff * z2_contrib
        return weight*diffsq, grad

    def torsion_distortion(self, torsion, grad, points):
        '''
        Return the score for torsion distortion

        Arguments:
        *torsion*
            gemmi.torsion
        *grad*
            1D array of gradient of size n
        *points*
            1D flatten array of x,y,z coordinates of size n
        '''
        # check the numbers from this functionZ
        p1 = gemmi.Position(points[torsion.index1], points[torsion.index1 + 1],
                            points[torsion.index1 + 2])
        p2 = gemmi.Position(points[torsion.index2], points[torsion.index2 + 1],
                            points[torsion.index2 + 2])
        p3 = gemmi.Position(points[torsion.index3], points[torsion.index3 + 1],
                            points[torsion.index3 + 2])
        p4 = gemmi.Position(points[torsion.index4], points[torsion.index4 + 1],
                            points[torsion.index4 + 2])
        #print(f'p1, {p1}\np2, {p2}\np3, {p3}\np3, {p4}')
        vec_a = p2 - p1
        vec_b = p3 - p2
        vec_c = p4 - p3
        len_a = p2.dist(p1)
        len_b = p3.dist(p2)
        len_c = p4.dist(p3)
        adotb = vec_a.dot(vec_b)
        adotc = vec_a.dot(vec_c)
        bdotc = vec_b.dot(vec_c)
        #print(f'dots, {adotb:.4f}, {adotc:.4f}, {bdotc:.4f}')
        cos_a1 = adotb / (len_a * len_b)
        cos_a2 = bdotc / (len_b * len_c)
        #print(f'cos_a, {cos_a1:.4f}, {cos_a2:.4f}')
        if cos_a1 > 0.9 or cos_a2 > 0.9:
            return 0.0, grad
        E_bsq = vec_a.dot(vec_b.cross(vec_c))*len_b
        G_bsq = adotb * bdotc - adotc * (len_b * len_b)
        theta_rad = arctan2(E_bsq, G_bsq)
        theta_deg = rad2deg(theta_rad)
        theta_deg += 360 if theta_deg < 0 else 0
        diff = 99999.9
        for i in range(0, torsion.period):
            trial_target = torsion.ideal + i * 360.0 / torsion.period
            if trial_target >= 360:
                trial_target -= 360.00
            trial_diff = theta_deg - trial_target
            if trial_diff < -180:
                trial_diff += 360.0
            if trial_diff > 180:
                trial_diff -= 360.0
            if fabs(trial_diff) < fabs(diff):
                diff = trial_diff
        if diff < -180:
            diff += 360.0
        elif diff > 180:
            diff -= 360.0
        diffsq = diff * diff
        #print(f'diffsq, {diffsq:.4f}')
        weight = 1/(torsion.sigma*torsion.sigma)
        H = -adotc
        J = adotb
        K = bdotc
        L = 1/(len_b*len_b)
        invlen_b = 1/len_b
        E = invlen_b * vec_a.dot(vec_b.cross(vec_c))
        G = H + J * K * L
        F = 1/G
        dH_dxP1 = vec_c.x
        dH_dxP2 = -vec_c.x
        dH_dxP3 = vec_a.x
        dH_dxP4 = -vec_a.x
        dK_dxP1 = 0
        dK_dxP2 = -vec_c.x
        dK_dxP3 = vec_c.x - vec_b.x
        dK_dxP4 = vec_b.x
        dJ_dxP1 = -vec_b.x
        dJ_dxP2 = vec_b.x - vec_a.x
        dJ_dxP3 = vec_a.x
        dJ_dxP4 = 0
        dL_dxP1 = 0
        dL_dxP2 = 2*(p3.x-p2.x)*L*L  # check sign from Paul
        dL_dxP3 = -2*(p3.x-p2.x)*L*L
        dL_dxP4 = 0
        dH_dyP1 = vec_c.y
        dH_dyP2 = -vec_c.y
        dH_dyP3 = vec_a.y
        dH_dyP4 = -vec_a.y
        dK_dyP1 = 0
        dK_dyP2 = -vec_c.y
        dK_dyP3 = vec_c.y - vec_b.y
        dK_dyP4 = vec_b.y
        dJ_dyP1 = -vec_b.y
        dJ_dyP2 = vec_b.y - vec_a.y
        dJ_dyP3 = vec_a.y
        dJ_dyP4 = 0
        dL_dyP1 = 0
        dL_dyP2 = 2*(p3.y-p2.y)*L*L  # check sign from Paul
        dL_dyP3 = -2*(p3.y-p2.y)*L*L
        dL_dyP4 = 0
        dH_dzP1 = vec_c.z
        dH_dzP2 = -vec_c.z
        dH_dzP3 = vec_a.z
        dH_dzP4 = -vec_a.z
        dK_dzP1 = 0
        dK_dzP2 = -vec_c.z
        dK_dzP3 = vec_c.z - vec_b.z
        dK_dzP4 = vec_b.z
        dJ_dzP1 = -vec_b.z
        dJ_dzP2 = vec_b.z - vec_a.z
        dJ_dzP3 = vec_a.z
        dJ_dzP4 = 0
        dL_dzP1 = 0
        dL_dzP2 = 2*(p3.z-p2.z)*L*L  # check sign from Paul
        dL_dzP3 = -2*(p3.z-p2.z)*L*L
        dL_dzP4 = 0
        dM_dxP1 = -(vec_b.y*vec_c.z - vec_b.z*vec_c.y)
        dM_dxP2 = ((vec_b.y*vec_c.z - vec_b.z*vec_c.y)
                   + (vec_a.y*vec_c.z - vec_a.z*vec_c.y))
        dM_dxP3 = ((vec_b.y*vec_a.z - vec_b.z*vec_a.y)
                   - (vec_a.y*vec_c.z - vec_a.z*vec_c.y))
        dM_dxP4 = -(vec_b.y*vec_a.z - vec_b.z*vec_a.y)#

        dM_dyP1 = -(vec_b.z*vec_c.x - vec_b.x*vec_c.z)
        dM_dyP2 = ((vec_b.z*vec_c.x - vec_b.x*vec_c.z)
                   + (vec_a.z*vec_c.x - vec_a.x*vec_c.z))
        dM_dyP3 = ((vec_b.z*vec_a.x - vec_b.x*vec_a.z)
                   - (vec_a.z*vec_c.x - vec_a.x*vec_c.z))
        dM_dyP4 = -(vec_b.z*vec_a.x - vec_b.x*vec_a.z)

        dM_dzP1 = -(vec_b.x*vec_c.y - vec_b.y*vec_c.x)
        dM_dzP2 = ((vec_b.x*vec_c.y - vec_b.y*vec_c.x)
                   + (vec_a.x*vec_c.y - vec_a.y*vec_c.x))
        dM_dzP3 = ((vec_a.y*vec_b.x - vec_a.x*vec_b.y)
                   - (vec_a.x*vec_c.y - vec_a.y*vec_c.x))
        dM_dzP4 = -(vec_a.y*vec_b.x - vec_a.x*vec_b.y)
        dE_dxP1 = dM_dxP1*invlen_b
        dE_dyP1 = dM_dyP1*invlen_b
        dE_dzP1 = dM_dzP1*invlen_b
        dE_dxP2 = dM_dxP2*invlen_b + E*(p3.x - p2.x)*L
        dE_dyP2 = dM_dyP2*invlen_b + E*(p3.y - p2.y)*L
        dE_dzP2 = dM_dzP2*invlen_b + E*(p3.z - p2.z)*L
        dE_dxP3 = dM_dxP3*invlen_b - E*(p3.x - p2.x)*L
        dE_dyP3 = dM_dyP3*invlen_b - E*(p3.y - p2.y)*L
        dE_dzP3 = dM_dzP3*invlen_b - E*(p3.z - p2.z)*L
        dE_dxP4 = dM_dxP4*invlen_b
        dE_dyP4 = dM_dyP4*invlen_b
        dE_dzP4 = dM_dzP4*invlen_b
        EFF = E*F*F
        JL = J*L
        KL = K*L
        JK = J*K
        dD_dxP1 = F*dE_dxP1 - EFF*(dH_dxP1 + JL*dK_dxP1 + KL*dJ_dxP1 + JK*dL_dxP1)
        dD_dxP2 = F*dE_dxP2 - EFF*(dH_dxP2 + JL*dK_dxP2 + KL*dJ_dxP2 + JK*dL_dxP2)
        dD_dxP3 = F*dE_dxP3 - EFF*(dH_dxP3 + JL*dK_dxP3 + KL*dJ_dxP3 + JK*dL_dxP3)
        dD_dxP4 = F*dE_dxP4 - EFF*(dH_dxP4 + JL*dK_dxP4 + KL*dJ_dxP4 + JK*dL_dxP4)
        dD_dyP1 = F*dE_dyP1 - EFF*(dH_dyP1 + JL*dK_dyP1 + KL*dJ_dyP1 + JK*dL_dyP1)
        dD_dyP2 = F*dE_dyP2 - EFF*(dH_dyP2 + JL*dK_dyP2 + KL*dJ_dyP2 + JK*dL_dyP2)
        dD_dyP3 = F*dE_dyP3 - EFF*(dH_dyP3 + JL*dK_dyP3 + KL*dJ_dyP3 + JK*dL_dyP3)
        dD_dyP4 = F*dE_dyP4 - EFF*(dH_dyP4 + JL*dK_dyP4 + KL*dJ_dyP4 + JK*dL_dyP4)
        dD_dzP1 = F*dE_dzP1 - EFF*(dH_dzP1 + JL*dK_dzP1 + KL*dJ_dzP1 + JK*dL_dzP1)
        dD_dzP2 = F*dE_dzP2 - EFF*(dH_dzP2 + JL*dK_dzP2 + KL*dJ_dzP2 + JK*dL_dzP2)
        dD_dzP3 = F*dE_dzP3 - EFF*(dH_dzP3 + JL*dK_dzP3 + KL*dJ_dzP3 + JK*dL_dzP3)
        dD_dzP4 = F*dE_dzP4 - EFF*(dH_dzP4 + JL*dK_dzP4 + KL*dJ_dzP4 + JK*dL_dzP4)
        #print(f'dD_dx, {dD_dxP1:.4f}, {dD_dxP2:.4f}, {dD_dxP3:.4f}, {dD_dxP4:.4f}')
        #print(f'dD_dy, {dD_dyP1:.4f}, {dD_dyP2:.4f}, {dD_dyP3:.4f}, {dD_dyP4:.4f}')
        #print(f'dD_dz, {dD_dzP1:.4f}, {dD_dzP2:.4f}, {dD_dzP3:.4f}, {dD_dzP4:.4f}')
        tt = tan(theta_rad)
        torsion_scale = rad2deg(1.0/(1+tt*tt))
        xP1_contrib = 2.0 * weight * diff * dD_dxP1 * torsion_scale
        xP2_contrib = 2.0 * weight * diff * dD_dxP2 * torsion_scale
        xP3_contrib = 2.0 * weight * diff * dD_dxP3 * torsion_scale
        xP4_contrib = 2.0 * weight * diff * dD_dxP4 * torsion_scale
        yP1_contrib = 2.0 * weight * diff * dD_dyP1 * torsion_scale
        yP2_contrib = 2.0 * weight * diff * dD_dyP2 * torsion_scale
        yP3_contrib = 2.0 * weight * diff * dD_dyP3 * torsion_scale
        yP4_contrib = 2.0 * weight * diff * dD_dyP4 * torsion_scale
        zP1_contrib = 2.0 * weight * diff * dD_dzP1 * torsion_scale
        zP2_contrib = 2.0 * weight * diff * dD_dzP2 * torsion_scale
        zP3_contrib = 2.0 * weight * diff * dD_dzP3 * torsion_scale
        zP4_contrib = 2.0 * weight * diff * dD_dzP4 * torsion_scale
        grad[torsion.index1] += xP1_contrib
        grad[torsion.index1 + 1] += yP1_contrib

        grad[torsion.index1 + 2] += zP1_contrib
        grad[torsion.index2] += xP2_contrib
        grad[torsion.index2 + 1] += yP2_contrib
        grad[torsion.index2 + 2] += zP2_contrib
        grad[torsion.index3] += xP3_contrib
        grad[torsion.index3 + 1] += yP3_contrib
        grad[torsion.index3 + 2] += zP3_contrib
        grad[torsion.index4] += xP4_contrib
        grad[torsion.index4 + 1] += yP4_contrib
        grad[torsion.index4 + 2] += zP4_contrib
        return weight*diffsq, grad

    def plane_distortion(self, plane, grad, points):
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        n_atoms = len(plane.indices)
        #print(f'n_atoms, {n_atoms}')
        p_l = []
        for i in range(n_atoms):
            index = plane.indices[i]
            p_l.append([points[index], points[index+1], points[index+2]])
            #print(f'p{i}, {points[index]:.4f}, {points[index+1]:.4f}, {points[index+2]:.4f}')
            sum_x += points[index]
            sum_y += points[index+1]
            sum_z += points[index+2]
        
        p_array = array(p_l)
        x_cen = sum_x / float(n_atoms)
        y_cen = sum_y / float(n_atoms)
        z_cen = sum_z / float(n_atoms)
        #print(f'centres, {x_cen:.4f}, {y_cen:.4f}, {z_cen:.4f}')
        pcen_diff = p_array - array([x_cen, y_cen, z_cen])
        mat = zeros((3, 3))
        mat[0, 0], mat[1, 1], mat[2, 2] = npsum(npsq(pcen_diff), axis=0)
        mat[0, 1] = npsum(pcen_diff[:, 0] * pcen_diff[:, 1], axis=0)
        mat[0, 2] = npsum(pcen_diff[:, 0] * pcen_diff[:, 2], axis=0)
        mat[1, 2] = npsum(pcen_diff[:, 1] * pcen_diff[:, 2], axis=0)

        mat[1, 0] = mat[0, 1]
        mat[2, 0] = mat[0, 2]
        mat[2, 1] = mat[1, 2]
        #print('mat')
        #print(f'{mat[0,0]:.4f}, {mat[0,1]:.4f}, {mat[0,2]:.4f}')
        #print(f'{mat[1,0]:.4f}, {mat[1,1]:.4f}, {mat[1,2]:.4f}')
        #print(f'{mat[2,0]:.4f}, {mat[2,1]:.4f}, {mat[2,2]:.4f}')
        eval, evec = linalg.eig(mat)
        idx = eval.argsort()
        eval = eval[idx]
        #print('eigens')
        #for i in range(eval.size):
        #    print(f'{eval[i]:.4f}, ', end='')
        #print('\n')
        evec = evec[:, idx]
        #print('mat after eigen')
        #print(f'{evec[0,0]:.4f}, {evec[0,1]:.4f}, {evec[0,2]:.4f}')
        #print(f'{evec[1,0]:.4f}, {evec[1,1]:.4f}, {evec[1,2]:.4f}')
        #print(f'{evec[2,0]:.4f}, {evec[2,1]:.4f}, {evec[2,2]:.4f}')
        abcd = zeros(4, dtype='float64')
        abcd[0] = evec[0, 0]
        abcd[1] = evec[1, 0]
        abcd[2] = evec[2, 0]
        sqsum = npsum(npsq(abcd))
        abcd = abcd/sqsum
        abcd[3] = abcd[0]*x_cen + abcd[1]*y_cen + abcd[2]*z_cen
        #print(f'abcd, {abcd[0]:.4f}, {abcd[1]:.4f}, {abcd[2]:.4f}, {abcd[3]:.4f}')
        weight = 1 / (plane.sigma * plane.sigma)
        val = (npsum(multiply(abcd[:3], p_array), axis=1) - abcd[3])
        #print(f'valsize : {val.size}')
        sum_devi = npsum(weight * npsq(val))
        #for i in range(val.size):
        #    print(f'val: {val[i]:.4f}, ', end='')
        #print(f'\nsum_dev: {sum_devi:.4f}')
        devi_len = npsum(multiply(abcd[:3], p_array), axis=1) - abcd[3]
        devi_len = reshape(devi_len, (devi_len.size, 1))
        d = 2.0 * weight * devi_len * abcd[:3]
        for i in range(n_atoms):
            grad[plane.indices[i]] += d[i, 0]
            grad[plane.indices[i]+1] += d[i, 1]
            grad[plane.indices[i]+2] += d[i, 2]
        
        return sum_devi, grad

    def update_coords(self, structure, x_opt):
        model = structure[0]
        for cra in model.all():
            index = self.atom_indices.get(cra.atom)
            if index is not None:
                cra.atom.pos = gemmi.Position(x_opt[index], x_opt[index + 1],
                                          x_opt[index + 2])
        #structure[0] = model
        return structure

    def get_atom_indices_len(self):
        return len(self.atom_indices)

    def print_sorted_dict(self):
        #sort_dict = sorted(self.atom_indices)
        for i, j in self.atom_indices.items():
            print(i, j)

def print_all_atoms(st):
    for cra in st[0].all():
        print(cra)

if __name__ == '__main__':
    from scipy.optimize import minimize
    from timeit import default_timer as timer
    import sys

    pdbin = sys.argv[1]
    pdbin1 = sys.argv[2]
    remove_lig_water = False
    if len(sys.argv) > 3:
        remove_lig_water = sys.argv[3]
    #pdbin1 = '/home/swh514/Projects/testing_ground/shiftfield_python/testrun/translate_1_5ni1_a.pdb'
    #pdbin1 = '/home/swh514/Projects/sheetbend_python_git/test_check/out23/sheetbend_pdbout_result_final.pdb'
    #pdbin = '/home/swh514/Projects/data/EMD-3488/fittedModels/PDB/pdb5ni1.ent'
    st = gemmi.read_structure(pdbin)
    st2 = gemmi.read_structure(pdbin1)
    st.ensure_entities()
    st2.ensure_entities()
    if remove_lig_water:
        st.remove_ligands_and_waters()
        st2.remove_ligands_and_waters()

    #monlib_path = os.environ['CCP4'] + '/lib/data/monomers'
    monlib_path  = '/opt/xtal/ccp4-7.1/lib/data/monomers'

    resnames = st[0].get_all_residue_names()
    print('all res names len : {0}'.format(len(resnames)))
    monlib = gemmi.read_monomer_lib(monlib_path, resnames)
    topo = gemmi.prepare_topology(st2, monlib)
    print('1')
    print('2')

    r1,r2,r3,r4 = get_rmsz(topo)
    print(r1, r2, r3, r4)
    #gemmi.Topo.Force
    print('3')
    errors = {}
    model2 = st2[0]
    model1 = st[0]
    '''
    for cra in model2.all():
        atom_add = gemmi.make_address(cra.chain, cra.residue, cra.atom)
        cra2 = model1.find_cra(atom_add)
        if cra2.atom is not None:
            dist_moved = cra.atom.pos.dist(cra2.atom.pos)
            errors[cra.atom] = 0.5 * dist_moved
    '''
    for cra in model2.all():
        #errors[cra.atom.serial] = 0.3
        errors[cra.atom] = 0.3

    print(len(errors))
    
    targetfunction = TargetFunction(topo, errors)
    print('atom_indices_len : {0}'.format(targetfunction.get_atom_indices_len()))
    print('n : {0}, cs : {1}'.format(targetfunction.n, targetfunction.coords.size))
    #targetfunction.print_sorted_dict()
    #x0 = targetfunction.coords
    #print(targetfunction.get_atom_indices_len())
    
    start = timer()
    res = minimize(targetfunction, targetfunction.coords, method='L-BFGS-B',
                   jac=True, tol=1e-6, options={'disp': True})  #, 'maxiter': 1})
    end = timer()
    print('opt time : {0}s'.format(end-start))
    # targetfunction.coords = res.x.copy()
    #print(targetfunction.get_atom_indices_len())
    if res.success:
        print(f'Obj func val : {res.fun}')
        print(f'Grad val :\n{res.jac}')
        st_opt = targetfunction.update_coords(st2, res.x)
        st_opt.write_minimal_pdb('test_pdb_scipymin_lbfgs_check.pdb')
        topo2 = gemmi.prepare_topology(st_opt, monlib)
        r1,r2,r3,r4 = get_rmsz(topo2)
        with open('coord_rmsd_lbfgs_check.csv', 'w') as f:
            for i in range(targetfunction.n):
                f.write("{0}, {1}, {2}\n".format(targetfunction.coords[i], res.x[i], res.x[i]-targetfunction.coords[i]))
    
'''# %%
def bond_distortion():
    """
    Score_bond = sum_all_bonds((1/sigma^2)(b_i - b_0)^2 ; b_i = ith angle, b0 ideal angle
    angle contributed by atom k,l,m

    gradient = dAi/dXm = 2*(1/sigma^2)[b_i - b0](Xm - Xk)/b_i
    """
    p1 = atom1
    p2 = atom2
    length = p1 - p2
    if length < 0.1:
        length = 0.1
    diff = length - ideal_bond
    diffsq = diff * diff
    weight = 1.0 / (sigma_bond * sigma_bond)
    # derivation dsi/dxm
    coeff = 2 * weight * diff / length

    gradient[i1] = coeff * (x1 - x2)
    gradient[i1+1] = coeff * (y1 - y2)
    gradient[i1+2] = coeff * (z1 - z2)
    gradient[i2] = coeff * (x2 - x1)
    gradient[i2+1] = coeff * (y2 - y1)
    gradient[i2+2] = coeff * (z2 - z1)


def angle_distortion():
    """
    Score_angle = sum_all_bonds((1/sigma^2)(a_i - a_0)^2 ; a_i = ith bond length, a0 ideal bond length
    gradient = dthetai/dXk = dtheta/dP * dP/dXk
    P = (a dot b)/(a*b)
    dtheta/dP = 1/sin(theta)  

    2*(1/sigma^2)[b_i - b0](Xm - Xk)/b_i
    """
    
'''
# %%
