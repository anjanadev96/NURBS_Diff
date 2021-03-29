import numpy as np
from numpy.linalg import inv
from geomdl import NURBS
from geomdl import multi
from geomdl import construct
from geomdl import convert
from geomdl.visualization import VisVTK as vis
from geomdl.visualization import VisMpL
from geomdl import exchange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CTRL_knot_list = [[0.0, 0.0], [0.0, 1.0 / 3.0], [0.0, 2.0 / 3.0], [0.0, 1.0],
#                   [1.0 / 3.0, 0.0], [1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 1.0],
#                   [2.0 / 3.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0],
#                   [1.0, 0.0], [1.0, 1.0 / 3.0], [1.0, 2.0 / 3.0], [1.0, 1.0]]
# CNTRL_Knot_Side = [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0],
#                    [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0],
#                    [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0],
#                    [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

# tri_list = [[0, 1, 4], [5, 4, 1], [1, 2, 5], [6, 5, 2], [2, 3, 6], [7, 6, 3],
#             [4, 5, 8], [9, 8, 5], [5, 6, 9], [10, 9, 6], [6, 7, 10], [11, 10, 7],
#             [8, 9, 12], [13, 12, 9], [9, 10, 13], [14, 13, 10], [10, 11, 14], [15, 14, 11]]


tri_list_side = [[0, 2, 1], [0, 3, 1], [3, 2, 1], [3, 2, 0],
                 [3, 11, 7], [3, 15, 7], [15, 11, 7], [15, 11, 3],
                 [15, 13, 14], [15, 12, 14], [12, 13, 14], [12, 13, 15],
                 [12, 4, 8], [12, 0, 8], [0, 4, 8], [0, 4, 12]]


def bound_box(cntrl_pt):
    bmax = np.empty([3])
    bmin = np.empty([3])

    bmin = [min(cntrl_pt[:, 0]), min(cntrl_pt[:, 1]), min(cntrl_pt[:, 2])]
    bmax = [max(cntrl_pt[:, 0]), max(cntrl_pt[:, 1]), max(cntrl_pt[:, 2])]

    return bmin, bmax


def bound_box_simul(ctrl_pts):
    b_max = np.empty([ctrl_pts.shape[0], 3], dtype=np.float32)
    b_min = np.empty([ctrl_pts.shape[0], 3], dtype=np.float32)

    e = 4

    for i in range(0, b_max.shape[0]):
        b_min[i, :] = [min(ctrl_pts[i, :, 0]),
                       min(ctrl_pts[i, :, 1]),
                       min(ctrl_pts[i, :, 2])]

        b_max[i, :] = np.array([max(ctrl_pts[i, :, 0]),
                       max(ctrl_pts[i, :, 1]),
                       max(ctrl_pts[i, :, 2])])
        pass

    return b_min, b_max


def padding_simul(b_min, b_max, vox_count):
    origin = np.empty([b_max.shape[0], 3], dtype=np.float32)
    vox_size = np.empty([b_max.shape[0]], dtype=np.float32)

    for i in range(0, b_min.shape[0]):
        g_max = [max(b_max[i, :, 0]), max(b_max[i, :, 1]), max(b_max[i, :, 2])]
        g_min = [min(b_min[i, :, 0]), min(b_min[i, :, 1]), min(b_min[i, :, 2])]

        maj_x = g_max[0] - g_min[0]
        maj_y = g_max[1] - g_min[1]
        maj_z = g_max[2] - g_min[2]

        maj_axis = max(max(maj_x, maj_y), maj_z)

        vox_size[i] = maj_axis / vox_count

        pad_x = maj_axis - maj_x
        pad_y = maj_axis - maj_y
        pad_z = maj_axis - maj_z

        if pad_x != 0:
            g_max[0] += pad_x / 2
            g_min[0] -= pad_x / 2
        if pad_y != 0:
            g_max[1] += pad_y / 2
            g_min[1] -= pad_y / 2
        if pad_z != 0:
            g_max[2] += pad_z / 2
            g_min[2] -= pad_z / 2

        origin[i] = [g_min[0] + (vox_size[i] / 2), g_min[1] + (vox_size[i] / 2), g_min[2] + (vox_size[i] / 2)]

    return origin, vox_size
    pass


def voxel_assign_single(voxels_all, val, direc, i, t_count, vox_count):
    if direc == 0:
        for inst in range(0, voxels_all.shape[0]):
            if voxels_all[inst][i // vox_count][i % vox_count][t_count] == 0:
                voxels_all[inst][i // vox_count][i % vox_count][t_count] = val
                break
            elif voxels_all[inst][i // vox_count][i % vox_count][t_count] == val:
                break
    elif direc == 1:
        for inst in range(0, voxels_all.shape[0]):
            if voxels_all[inst][i % vox_count][t_count][i // vox_count] == 0:
                voxels_all[inst][i % vox_count][t_count][i // vox_count] = val
                break
            elif voxels_all[inst][i % vox_count][t_count][i // vox_count] == val:
                break
    elif direc == 2:
        for inst in range(0, voxels_all.shape[0]):
            if voxels_all[inst][t_count][i // vox_count][i % vox_count] == 0:
                voxels_all[inst][t_count][i // vox_count][i % vox_count] = val
                break
            elif voxels_all[inst][t_count][i // vox_count][i % vox_count] == val:
                break
            pass
    pass


def nr_inter_single(b_max, b_min, vox_count, vox_size, origin, dir_1, dir_2, ray_d,
                    tri_list_3, ctrl_pts, knot_list_3, vox_all, direc, arr_idx):
    tri = np.empty([3, 3], dtype=np.float32)

    for j in range(0, vox_count * vox_count):
        ray = [origin[0] + ((j // vox_count) * vox_size * dir_1[0]) + ((j % vox_count) * vox_size * dir_2[0]),
               origin[1] + ((j // vox_count) * vox_size * dir_1[1]) + ((j % vox_count) * vox_size * dir_2[1]),
               origin[2] + ((j // vox_count) * vox_size * dir_1[2]) + ((j % vox_count) * vox_size * dir_2[2])]

        for k in range(0, b_max.shape[0]):

            if ray_box_inter(b_min[k], b_max[k], ray, ray_d):

                for t in range(0, tri_list_3.shape[0]):
                    TRI_ptS = ctrl_pts[k // 6][k % 6]

                    tri[0] = [TRI_ptS[tri_list_3[t][0]][0], tri_pts[tri_list_3[t][0]][1], tri_pts[tri_list_3[t][0]][2]]
                    tri[1] = [tri_pts[tri_list_3[t][1]][0], tri_pts[tri_list_3[t][1]][1], tri_pts[tri_list_3[t][1]][2]]
                    tri[2] = [tri_pts[tri_list_3[t][2]][0], tri_pts[tri_list_3[t][2]][1], tri_pts[tri_list_3[t][2]][2]]

                    A = np.array([[-ray_d[0], tri[2][0] - tri[0][0], tri[1][0] - tri[0][0]],
                                  [-ray_d[1], tri[2][1] - tri[0][1], tri[1][1] - tri[0][1]],
                                  [-ray_d[2], tri[2][2] - tri[0][2], tri[1][2] - tri[0][2]]])

                    B = np.array([[ray[0] - tri[0][0]], [ray[1] - tri[0][1]], [ray[2] - tri[0][2]]])

                    param = np.matmul(inv(A), B)
                    if param[1] >= 0.0 and param[2] >= 0.0:
                        if param[1] + param[2] <= 1.0:
                            # print('intersection')

                            knot_inter = [knot_list_3[tri_list_3[t][0]][0], knot_list_3[tri_list_3[t][0]][1]]
                            t_inter = param[0]

                            if t % 2 == 0:
                                u_inter = knot_inter[0] + (param[1] * 0.33)
                                v_inter = knot_inter[1] + (param[2] * 0.33)
                            else:
                                u_inter = knot_inter[0] - (param[1] * 0.33)
                                v_inter = knot_inter[1] - (param[2] * 0.33)

                            [bol, t_count] = newton_method(t_inter, u_inter, v_inter, ray, ray_d, vox_size, tri_pts, 3)

                            if bol:
                                # val = (int(k // 6) + 1)
                                val = int(arr_idx[j])
                                voxel_assign_single(vox_all, val, direc, j, t_count, vox_count)

    return vox_all


def post_process(voxels_all, voxel_master, vox_x, vox_y, vox_z, direc, vox_1, vox_2):
    for i in range(0, vox_1 * vox_2):
        inout_vox = np.empty(2, dtype=np.uint8)

        vox_list_1 = np.zeros(5, dtype=np.float32)
        vox_list_2 = np.zeros(5, dtype=np.float32)

        if direc == 0:

            for j in range(0, vox_z):
                if voxels_all[0][i // vox_y][i % vox_y][j] != 0:

                    for inst in range(0, voxels_all.shape[0]):
                        vox_list_2[inst] = voxels_all[inst][i // vox_y][i % vox_y][j]

                    elem = list_compare(voxels_all.shape[0], vox_list_1, vox_list_2)
                    if elem != -1:
                        inout_vox[1] = j
                        for vox in range(inout_vox[0], inout_vox[1] + 1):
                            voxel_master[i // vox_y][i % vox_y][vox] = elem

                    for inst in range(0, voxels_all.shape[0]):
                        vox_list_1[inst] = voxels_all[inst][i // vox_y][i % vox_y][j]
                    inout_vox[0] = j

        elif direc == 1:

            for j in range(0, vox_y):
                if voxels_all[0][i % vox_x][j][i // vox_x] != 0:

                    for inst in range(0, voxels_all.shape[0]):
                        vox_list_2[inst] = voxels_all[inst][i % vox_x][j][i // vox_x]

                    elem = list_compare(voxels_all.shape[0], vox_list_1, vox_list_2)
                    if elem != -1:
                        inout_vox[1] = j
                        for vox in range(inout_vox[0], inout_vox[1] + 1):
                            voxel_master[i % vox_x][vox][i // vox_x] = elem

                    for inst in range(0, voxels_all.shape[0]):
                        vox_list_1[inst] = voxels_all[inst][i % vox_x][j][i // vox_x]
                    inout_vox[0] = j

        elif direc == 2:

            for j in range(0, vox_x):
                if voxels_all[0][j][i // vox_z][i % vox_z] != 0:

                    for inst in range(0, voxels_all.shape[0]):
                        vox_list_2[inst] = voxels_all[inst][j][i // vox_z][i % vox_z]

                    elem = list_compare(voxels_all.shape[0], vox_list_1, vox_list_2)
                    if elem != -1:
                        inout_vox[1] = j
                        for vox in range(inout_vox[0], inout_vox[1] + 1):
                            voxel_master[vox][i // vox_z][i % vox_z] = elem

                    for inst in range(0, voxels_all.shape[0]):
                        vox_list_1[inst] = voxels_all[inst][j][i // vox_z][i % vox_z]
                    inout_vox[0] = j

    return voxel_master

    pass


def list_compare(depth, vox_list_1, vox_list_2):
    elem = -1

    for idx_1 in range(0, depth):
        if vox_list_1[idx_1] != 0:
            for idx_2 in range(0, depth):
                if vox_list_2[idx_2] != 0:
                    if vox_list_1[idx_1] == vox_list_2[idx_2]:
                        elem = vox_list_1[idx_1]

    return elem
    pass


# def gauss_val(vox_master, vox_count, origin, vox_size, stress): # for two chamber
def gauss_val(vox_master, vox_count, origin, vox_size, ctrl_pts, stress):  # for Aorta
    for i in range(0, vox_count * vox_count):
        for j in range(0, vox_count):

            if vox_master[i // vox_count][i % vox_count][j] != 0:
                elem = int(vox_master[i // vox_count][i % vox_count][j]) - 1
                gauss_vals = min_dist_vox(j, origin, vox_count, vox_size, i, ctrl_pts[elem])

                vox_master[i // vox_count][i % vox_count][j] = gauss_vals

    return vox_master
    pass


def newton_single(t, u, v, ray, direc, cps, degree):
    non_conv = 0
    iter_count = 0
    t_count = 0

    knot_u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    knot_v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    s_1 = surf_pt(u, v, cps, knot_u, knot_v, degree, degree)
    p_1 = ray + (t * direc)

    obj_func = np.abs(np.array([[s_1[0] - p_1[0]], [s_1[1] - p_1[1]], [s_1[2] - p_1[2]]]))
    # obj_func = np.array([[s_1[0] - p_1[0]], [s_1[1] - p_1[1]], [s_1[2] - p_1[2]]])
    dist = np.linalg.norm(s_1 - p_1)

    print(dist)

    while dist > 0.001:
        deri = deri_surf(u, v, 1, cps, knot_u, knot_v, degree, degree)

        jacob = np.array([[-direc[0], deri[1][0][0], deri[0][1][0]],
                          [-direc[1], deri[1][0][1], deri[0][1][1]],
                          [-direc[2], deri[1][0][2], deri[0][1][2]]])

        opti_sub = np.matmul(inv(jacob), obj_func)

        t -= opti_sub[0]
        u -= opti_sub[1]
        v -= opti_sub[2]

        print(t, u, v)

        if u < 0.0:  u = np.random.random()
        if u > 1.0:  u = np.random.random()
        if v < 0.0:  v = np.random.random()
        if v > 1.0:  v = np.random.random()

        if t < 0.0:
            print(t, u, v, '  ====  its negative')
            t = 0.0

        if (u == 0.0 and v == 0.0) or (u == 1.0 and v == 1.0):
            non_conv += 1
            print('Non Conver')
        if non_conv == 50 or iter_count == 50:
            print(non_conv, iter_count)
            return False, t_count

        s_1 = surf_pt(u, v, cps, knot_u, knot_v, degree, degree)
        p_1 = ray + (t * direc)

        obj_func = np.array([[s_1[0] - p_1[0]], [s_1[1] - p_1[1]], [s_1[2] - p_1[2]]])
        # obj_func = np.abs(np.array([[s_1[0] - p_1[0]], [s_1[1] - p_1[1]], [s_1[2] - p_1[2]]]))
        dist = np.linalg.norm(s_1 - p_1)

        print(dist)

        iter_count += 1

    pts = surf_pt(u, v, cps, knot_u, knot_v, degree, degree)

    return True, pts
    pass


def newton_method(t, u, v, ray, ray_d, vox_size, ctrl_pts, degree):
    non_conv = 0
    iter_count = 0
    t_count = 0

    s_1 = surf_pt(u, v, ctrl_pts, degree)
    p_1 = ray + (t * ray_d)

    obj_func = np.array([[s_1[0] - p_1[0]], [s_1[1] - p_1[1]], [s_1[2] - p_1[2]]])
    dist = np.linalg.norm(s_1 - p_1)

    while dist > 0.001:
        deri = deri_surf(u, v, 1, ctrl_pts, 3)

        jacob = np.array([[-ray_d[0], deri[1][0][0], deri[0][1][0]],
                          [-ray_d[1], deri[1][0][1], deri[0][1][1]],
                          [-ray_d[2], deri[1][0][2], deri[0][1][2]]])

        opti_sub = np.matmul(inv(jacob), obj_func)

        t -= opti_sub[0]
        u -= opti_sub[1]
        v -= opti_sub[2]

        if u < 0.0:  u = 0.0
        if u > 1.0:  u = 1.0
        if v < 0.0:  v = 0.0
        if v > 1.0:  v = 1.0

        if (u == 0.0 and v == 0.0) or (u == 1.0 and v == 1.0):
            non_conv += 1
        if non_conv == 50 or iter_count == 50:
            return False, t_count

        s_1 = surf_pt(u, v, ctrl_pts, degree)
        p_1 = ray + (t * ray_d)

        obj_func = np.array([[s_1[0] - p_1[0]], [s_1[1] - p_1[1]], [s_1[2] - p_1[2]]])
        dist = np.linalg.norm(s_1 - p_1)

        iter_count += 1

    if dist < 0.001:
        t_count = int(t // vox_size)
        if t % vox_size >= vox_size / 2:
            t_count += 1
    pass
    return True, t_count


def ray_box_inter(b_min, b_max, ray_o, ray_d):
    invR = np.empty([3], dtype=np.float32)

    if ray_d[0] == 0.0:
        invR[0] = np.inf
    else:
        invR[0] = 1.0 / ray_d[0]

    if ray_d[1] == 0.0:
        invR[1] = np.inf
    else:
        invR[1] = 1.0 / ray_d[1]

    if ray_d[2] == 0.0:
        invR[2] = np.inf
    else:
        invR[2] = 1.0 / ray_d[2]

    tbot = [invR[0] * (b_min[0] - ray_o[0]), invR[1] * (b_min[1] - ray_o[1]), invR[2] * (b_min[2] - ray_o[2])]
    ttop = [invR[0] * (b_max[0] - ray_o[0]), invR[1] * (b_max[1] - ray_o[1]), invR[2] * (b_max[2] - ray_o[2])]

    tmin = [min(ttop[0], tbot[0]), min(ttop[1], tbot[1]), min(ttop[2], tbot[2])]
    tmax = [max(ttop[0], tbot[0]), max(ttop[1], tbot[1]), max(ttop[2], tbot[2])]

    t_near = max(max(tmin[0], tmin[1]), max(tmin[0], tmin[2]))
    t_far = min(min(tmax[0], tmax[1]), min(tmax[0], tmax[2]))

    return t_far >= t_near


def tri_intersection_single():
    pass


def tri_intersection(ray_o, ray_d, tri_c, tri_s):
    intersection = False
    initial_uvt = []

    for tri in range(0, tri_c.shape[0]):
        A = np.array([[-ray_d[0], tri_c[tri][2][0] - tri_c[tri][0][0], tri_c[tri][1][0] - tri_c[tri][0][0]],
                      [-ray_d[1], tri_c[tri][2][1] - tri_c[tri][0][1], tri_c[tri][1][1] - tri_c[tri][0][1]],
                      [-ray_d[2], tri_c[tri][2][2] - tri_c[tri][0][2], tri_c[tri][1][2] - tri_c[tri][0][2]]])

        B = np.array([[ray_o[0] - tri_c[tri][0][0]], [ray_o[1] - tri_c[tri][0][1]], [ray_o[2] - tri_c[tri][0][2]]])

        param = np.matmul(inv(A), B)

        if param[1] >= 0.0 and param[2] >= 0.0:
            if param[1] + param[2] <= 1.0:
                intersection = True
                # triangle_Number.append(i)
                knot_inter = CTRL_knot_list[tri_list[tri][0]]
                if tri % 2 == 0:
                    initial_uvt.append([param[0][0],
                                        knot_inter[0] + (param[1][0] * 1.0 / 3.0),
                                        knot_inter[1] + (param[2][0] * 1.0 / 3.0)])
                else:
                    initial_uvt.append([param[0][0],
                                        knot_inter[0] - (param[1][0] * 1.0 / 3.0),
                                        knot_inter[1] - (param[2][0] * 1.0 / 3.0)])
                pass
            pass

    for tri in range(0, tri_s.shape[0]):
        A = np.array([[-ray_d[0], tri_s[tri][2][0] - tri_s[tri][0][0], tri_s[tri][1][0] - tri_s[tri][0][0]],
                      [-ray_d[1], tri_s[tri][2][1] - tri_s[tri][0][1], tri_s[tri][1][1] - tri_s[tri][0][1]],
                      [-ray_d[2], tri_s[tri][2][2] - tri_s[tri][0][2], tri_s[tri][1][2] - tri_s[tri][0][2]]])

        B = np.array([[ray_o[0] - tri_s[tri][0][0]], [ray_o[1] - tri_s[tri][0][1]], [ray_o[2] - tri_s[tri][0][2]]])

        param = np.matmul(inv(A), B)

        if param[1] >= 0.0 and param[2] >= 0.0:
            if param[1] + param[2] <= 1.0:
                intersection = True
                # triangle_Number.append(i)
                # knot_inter = CTRL_knot_list[tri_list_side[tri][0]]
                if tri // 4 == 0:
                    if tri_list_side[tri][0] == 0:
                        initial_uvt.append([param[0][0], 0.0, param[2][0]])
                    else:
                        initial_uvt.append([param[0][0], 0.0, 1 - param[2][0]])
                elif tri // 4 == 1:
                    if tri_list_side[tri][0] == 3:
                        initial_uvt.append([param[0][0], param[1][0], 1.0])
                    else:
                        initial_uvt.append([param[0][0], 1 - param[1][0], 1.0])
                elif tri // 4 == 2:
                    if tri_list_side[tri][0] == 15:
                        initial_uvt.append([param[0][0], 1.0, 1 - param[2][0]])
                    else:
                        initial_uvt.append([param[0][0], 1.0, param[2][0]])
                elif tri // 4 == 3:
                    if tri_list_side[tri][0] == 12:
                        initial_uvt.append([param[0][0], 1 - param[1][0], 0.0])
                    else:
                        initial_uvt.append([param[0][0], param[1][0], 0.0])

    tuv = np.array(initial_uvt)
    return intersection, tuv


def ray_grid(vox_res, origin, size):
    dir_x = [1.0, 0.0, 0.0]
    dir_y = [0.0, 1.0, 0.0]
    ray_origin = np.empty([vox_res[0] * vox_res[1], 3])

    for i in range(0, vox_res[0] * vox_res[1]):
        ray_origin[i] = [origin[0] + ((i // vox_res[1]) * size * dir_x[0]) + ((i % vox_res[1]) * size * dir_y[0]),
                         origin[1] + ((i // vox_res[1]) * size * dir_x[1]) + ((i % vox_res[1]) * size * dir_y[1]),
                         origin[2] + ((i // vox_res[1]) * size * dir_x[2]) + ((i % vox_res[1]) * size * dir_y[2])]

    return ray_origin


# def tri_list(control_points_All_Face):
#     # Computing Triangle list
#     tri_list_C = np.empty([control_points_All_Face.shape[0], 18, 3, 3])
#
#     for i in range(0, control_points_All_Face.shape[0]):
#         control_points = control_points_All_Face[i]
#         for j in range(0, 18):
#             tri_list_C[i][j] = [[control_points[tri_list[j][0]][0], control_points[tri_list[j][0]][1],
#                                 control_points[tri_list[j][0]][2]],
#                                 [control_points[tri_list[j][1]][0], control_points[tri_list[j][1]][1],
#                                 control_points[tri_list[j][1]][2]],
#                                 [control_points[tri_list[j][2]][0], control_points[tri_list[j][2]][1],
#                                 control_points[tri_list[j][2]][2]]]
#     return tri_list_C


def nr_iter(control_points, ctrl_pt_u, ctrl_pt_v, ini_uvt, ray_start_p, ray_direc):
    # surf = NURBS.Surface()
    # surf.delta = 0.1
    #
    # surf.degree_u = 3
    # surf.degree_v = 3
    #
    bol = False
    inter_point = np.empty(3)
    #
    # surf.set_ctrlpts(control_points, ctrl_pt_u, ctrl_pt_v)
    #
    # surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    non_conv = 0;
    iter_count = 0

    s_1 = surf_pt(ini_uvt[1], ini_uvt[2], control_points)
    p_1 = ray_start_p + (ini_uvt[0] * ray_direc)

    f_x = s_1[0] - p_1[0]
    g_y = s_1[1] - p_1[1]
    h_z = s_1[2] - p_1[2]
    obj_func = [[f_x], [g_y], [h_z]]
    dist = np.linalg.norm(s_1 - p_1)

    temp = ini_uvt[0]

    while dist > 0.0001:
        deri = deri_surf(ini_uvt[1], ini_uvt[2], 1, control_points)

        f_x_u = deri[1][0][0]
        f_x_v = deri[0][1][0]
        f_x_t = -ray_direc[0]
        g_y_u = deri[1][0][1]
        g_y_v = deri[0][1][1]
        g_y_t = -ray_direc[1]
        h_z_u = deri[1][0][2]
        h_z_v = deri[0][1][2]
        h_z_t = -ray_direc[2]

        jacob = np.array([[f_x_t, f_x_u, f_x_v],
                          [g_y_t, g_y_u, g_y_v],
                          [h_z_t, h_z_u, h_z_v]])

        opti_sub = np.matmul(inv(jacob), obj_func)

        temp = ini_uvt[0]

        ini_uvt = ini_uvt - opti_sub  # Next iteration

        for j in range(1, 3):
            if ini_uvt[j] < 0.0:
                ini_uvt[j] = 0.0
            if ini_uvt[j] > 1.0:
                ini_uvt[j] = 1.0

        if (ini_uvt[1] == 1.0 and ini_uvt[2] == 1.0) or (ini_uvt[1] == 0.0 and ini_uvt[2] == 0.0):
            non_conv = non_conv + 1  # Check for non convergence.

        if non_conv == 15 or iter_count == 100:
            break

        s_1 = surf_pt(ini_uvt[1], ini_uvt[2], control_points)
        p_1 = ray_start_p + (ini_uvt[0] * ray_direc)

        f_x = s_1[0] - p_1[0];
        g_y = s_1[1] - p_1[1];
        h_z = s_1[2] - p_1[2]
        obj_func = [[f_x], [g_y], [h_z]]
        dist = np.linalg.norm(s_1 - p_1)
        iter_count = iter_count + 1

    if dist < 0.0001:
        # Final_uvt.append([ini_uvt[0], ini_uvt[1], ini_uvt[2]])
        # norm_uv = np.array(surf.normal([initial_uvt[1][i], initial_uvt[2][i]], normalize=True))
        bol = True
        inter_point = s_1
        # print('iteration : ', iter_count, ' t = ', initial_uvt[0][i], ' u = ', initial_uvt[1][i], ' v = ',
    #       initial_uvt[2][i], ' Dist = ', dist, '\n Inter_point S = ', s_1, '\n Inter point T = ', p_1)

    # print(iter_count)

    return bol, inter_point, ini_uvt.transpose(), iter_count


def compute_normal_surface(ctrl_pts, knot_u, knot_v, grid_1, grid_2):
    count = grid_1.shape[1]
    pt = np.empty([grid_1.size, 3])
    # OFF_pt = np.empty([grid_1.size, 3])
    # edge_bool = np.zeros([grid_1.size])
    # OFF_pt_2 = np.empty([grid_1.size, 3])
    deri = np.empty([grid_1.size, 2, 2, 3])
    normals = np.empty([grid_1.size, 3])

    for i in range(0, grid_1.shape[0]):
        for j in range(0, grid_1.shape[1]):
            # if grid_1[i][j] == 0.0 or grid_2[i][j] == 0.0 or \
            #         grid_1[i][j] == 1.0 or grid_2[i][j] == 1.0:
            #     edge_bool[i * count + j] = 1

            # pt[i * delta + j] = surf_pt(grid_1[i][j], grid_2[i][j], ctrl_pts[times], 3)
            pt[i * count + j] = surf_pt(grid_1[i][j], grid_2[i][j], ctrl_pts, 3, knot_u, knot_v)

            deri[i * count + j] = deri_surf(grid_1[i][j], grid_2[i][j], 1, ctrl_pts, 3, knot_u, knot_v)

            temp = np.cross(deri[i * count + j][0][1], deri[i * count + j][1][0])

            normals[i * count + j] = temp / np.linalg.norm(temp)

            # OFF_pt[i * delta_u + j] = pt[i * delta_u + j] + (0.1 * normals[i * delta_u + j])
            # #
            # OFF_pt_2[i * delta_u + j] = OFF_pt[i * delta_u + j] + (0.1 * normals[i * delta_u + j])

            pass
    # count = np.sum(edge_bool)
    return pt, normals
    pass


def compute_surf_offset(off_layers, ctrl_pts, knot_u, knot_v, degree_u, degree_v):
    delta_u = 64
    delta_v = 64
    grid_1, grid_2 = np.meshgrid(np.linspace(0.0, 1.0, delta_u), np.linspace(0.0, 1.0, delta_v))
    off_pts = np.empty([off_layers, grid_1.shape[0], grid_1.shape[1], 3], dtype=np.float32)
    thickness = 5.0
    for i in range(0, grid_1.shape[0]):
        for j in range(0, grid_1.shape[1]):

            # ptS, normals = compute_normal_surface(ctrl_pts, knot_u, knot_v, grid_1, grid_2)
            pt = surf_pt(grid_1[i][j], grid_2[i][j], ctrl_pts, knot_u, knot_v, degree_u, degree_v)
            deri = deri_surf(grid_1[i][j], grid_2[i][j], 1, ctrl_pts, knot_u, knot_v, degree_u, degree_v)
            temp = np.cross(deri[0][1], deri[1][0])
            normals = temp / np.linalg.norm(temp)

            # if pt[1] < 12:
            #     thickness = 0.5
            # elif 40 > pt[1] > 12:
            #     thickness = 0.25
            # else:
            #     thickness = 0.1

            for layer in range(0, off_layers):
                off_pts[layer][i][j] = pt + (thickness * layer * normals)

    return off_pts


def compute_volumes(ctrl_pts, Count_u, Count_v, Edge_point_count):
    map_size = 4
    normals = np.empty([ctrl_pts.shape[0], ctrl_pts.shape[1], 3])
    edge_pts_idx = np.empty([ctrl_pts.shape[0], Edge_point_count, map_size], dtype=np.int)

    knot_u = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    knot_v = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    for i in range(0, ctrl_pts.shape[0]): # ctrl_pts.shape[0]):
        normals[i, :], edge_pts_idx[i, :] = compute_cntrlpts_normals(ctrl_pts[i], Count_u, Count_v,
                                                                     Edge_point_count, map_size)

    edge_pts_idx = map_edge_points(ctrl_pts, edge_pts_idx)

    normals = normals_reassign(edge_pts_idx, normals)

    compute_model_offset(ctrl_pts, normals, Count_u, Count_v, 3, knot_u, knot_v, off_layer=2, thickness=10)
    pass


def compute_model_offset(cntrlpts, cntrlpts_normals, Count_u, Count_v, Degree, knot_u, knot_v, off_layer, thickness, ):
    ctrlpts_offset = np.empty([cntrlpts.shape[0], cntrlpts.shape[1], 3], dtype=np.float32)

    msurf = multi.SurfaceContainer()
    msurf.delta = 0.05

    Multi = multi.SurfaceContainer()
    Multi.delta = 0.1
    for i in range(0, cntrlpts_normals.shape[0]):

        # Multi.delta_u = 0.1
        # Multi.delta_v = 0.1
        for off in range(1, off_layer):
            ctrlpts_offset[i] = compute_cntrlpt_offset(cntrlpts[i], cntrlpts_normals[i], off, thickness)

            # surf = NURBS.Surface()
            # surf.delta = 0.1
            # # surf.delta_u = 0.1
            # # surf.delta_v = 0.1
            #
            # surf.degree_u = 3
            # surf.degree_v = 3
            #
            # surf.set_ctrlpts(ctrlpts_offset[i].tolist(), Count_u, Count_v)
            #
            # surf.knotvector_u = knot_u
            # surf.knotvector_v = knot_v
            #
            # Multi.add(surf)

        # for i in range(1, off_layer):
        #     volume = construct.construct_volume('w', Multi[i - 1], Multi[i], degree=1)
        #     nvolume = convert.bspline_to_nurbs(volume)
        #
        #     surfvol = construct.extract_isosurface(nvolume)
        #     msurf.add(surfvol)
    # print(len(msurf))
    # # vis_config = VisMpL.VisConfig(legend=False, axes=False, ctrlpts=False)
    # msurf.vis = vis.VisSurface(vis.VisConfig(ctrlpts=False, figure_size=[1024, 768]))
    # msurf.render()  # , plot=False, filename="VLM_3DBeam_Before.pdf")

    # exchange.export_obj(Multi, 'Wind_blade_%d_layers.obj' % off_layer)
    # exchange.export_obj(Multi, 'HeartModel_layers_%d.obj' % off_layer)
    # exchange.export_obj(msurf, 'Blade_combined_layer.obj')
    # exchange.export_csv(Multi, 'WindBlade_surfpoint.csv', point_type='evalpts')
    # vis_config = VisMpL.VisConfig(legend=False, axes=False, ctrlpts=False)
    # vis_comp = VisMpL.VisSurface(vis_config)
    # Multi.vis = vis.VisSurface(vis.VisConfig(ctrlpts=False, figure_size=[1024, 768]))
    # Multi.render()

    return ctrlpts_offset
    pass


def compute_cntrlpt_offset(cntrlpts, cntrlpts_normals, off_layer, thickness):
    off_cntrlpts = np.empty(cntrlpts.shape)
    for i in range(0, cntrlpts.shape[0]):
        off_cntrlpts[i][0:3] = (thickness * off_layer * cntrlpts_normals[i]) + cntrlpts[i][0:3]
        # off_cntrlpts[i][3] = 1.0

    return off_cntrlpts


def map_edge_points(cntrlpts, cntrlpts_edge_map):
    for i in range(0, cntrlpts.shape[0]):
        for j in range(0, cntrlpts.shape[0]):

            if i != j:
                for k in range(0, cntrlpts_edge_map[i].shape[0]):
                    pt1 = cntrlpts[i][cntrlpts_edge_map[i][k][0]]
                    for l in range(0, cntrlpts_edge_map[j].shape[0]):
                        pt2 = cntrlpts[j][cntrlpts_edge_map[j][l][0]]
                        if np.linalg.norm(pt1 - pt2) == 0:
                            cntrlpts_edge_map[i][k][1] += 1
                            if cntrlpts_edge_map[i][k][1] == 6:
                                print("5 Common")
                            cntrlpts_edge_map[i][k][(2 * cntrlpts_edge_map[i][k][1])] = j
                            cntrlpts_edge_map[i][k][(2 * cntrlpts_edge_map[i][k][1]) + 1] = cntrlpts_edge_map[j][l][0]

                            # if cntrlpts_edge_idk[i][k][1] == 0 and cntrlpts_edge_idk[j][l][1] == 0:
                            #     # cntrlpts_edge_idk[i][k][1] = 1
                            #     # cntrlpts_edge_idk[j][l][1] = 1
                            #
                            #     temp_norm = (cntrlpts_normals[i][cntrlpts_edge_idk[i][k][0]] +
                            #                  cntrlpts_normals[j][cntrlpts_edge_idk[j][l][0]])
                            #     temp_norm_mag = np.linalg.norm(temp_norm)
                            #     if temp_norm_mag != 0.0:
                            #         cntrlpts_normals[i][cntrlpts_edge_idk[i][k][0]] = temp_norm / temp_norm_mag
                            #         cntrlpts_normals[j][cntrlpts_edge_idk[j][l][0]] = temp_norm / temp_norm_mag
                            #     else:
                            #         cntrlpts_normals[i][cntrlpts_edge_idk[i][k][0]] = temp_norm
                            #         cntrlpts_normals[j][cntrlpts_edge_idk[j][l][0]] = temp_norm
    return cntrlpts_edge_map
    pass


def normals_reassign(cntrlpts_edge_map, cntrlpts_normals):
    for i in range(0, len(cntrlpts_edge_map)):
        for j in range(0, cntrlpts_edge_map[i].shape[0]):
            temp = cntrlpts_normals[i][cntrlpts_edge_map[i][j][0]]
            for k in range(0, cntrlpts_edge_map[i][j][1]):
                idx_patch = cntrlpts_edge_map[i][j][(k + 1) * 2]
                idx_cntrl_pt = cntrlpts_edge_map[i][j][((k + 1) * 2) + 1]
                temp += cntrlpts_normals[idx_patch][idx_cntrl_pt]

            temp_norm = np.linalg.norm(temp)
            if temp_norm != 0.0:
                temp = temp / temp_norm
                # cntrlpts_normals[i][cntrlpts_edge_map[i][j][0]] = temp / temp_norm
            cntrlpts_normals[i][cntrlpts_edge_map[i][j][0]] = temp
            for k in range(0, cntrlpts_edge_map[i][j][1]):
                idx_patch = cntrlpts_edge_map[i][j][(k + 1) * 2]
                idx_cntrl_pt = cntrlpts_edge_map[i][j][((k + 1) * 2) + 1]
                cntrlpts_normals[idx_patch][idx_cntrl_pt] = temp
                pass

    return cntrlpts_normals
    pass


def compute_cntrlpts_normals(ctrl_pts, count_u, count_v, edge_pts_count, map_size):

    vec_combo = [[0, 1], [1, 2], [2, 3], [3, 0]]
    normals_res = np.empty([ctrl_pts.shape[0], 3])
    edge_pts_idx = np.empty([edge_pts_count, map_size], dtype=np.int32)
    count = 0
    for i in range(0, ctrl_pts.shape[0]):
        adj_pts = np.empty(4, dtype=np.int16)
        normals = np.zeros([4, 3])

        adj_pts[0] = i + count_v
        adj_pts[1] = i + 1
        adj_pts[2] = i - count_v
        adj_pts[3] = i - 1

        if adj_pts[0] >= count_u * count_v:
            adj_pts[0] = -1
        if adj_pts[1] == count_v * ((i // count_v) + 1):
            adj_pts[1] = -1
        if adj_pts[2] < 0:
            adj_pts[2] = -1
        if adj_pts[3] == (count_v * (i // count_v)) - 1:
            adj_pts[3] = -1

        for vec in range(0, 4):
            if adj_pts[vec_combo[vec][0]] != -1 and adj_pts[vec_combo[vec][1]] != -1:
                normals[vec] = unit_normal(np.array(ctrl_pts[i, 0:3]),
                                           np.array(ctrl_pts[adj_pts[vec_combo[vec][0]], 0:3]),
                                           np.array(ctrl_pts[adj_pts[vec_combo[vec][1]], 0:3]))

        res_vec = [np.sum(normals[:, 0]), np.sum(normals[:, 1]), np.sum(normals[:, 2])]
        if np.linalg.norm(res_vec) != 0.0:
            normals_res[i] = res_vec / np.linalg.norm(res_vec)
        else:
            normals_res[i] = np.array(res_vec)

        if np.any(adj_pts == -1):
            edge_pts_idx[count, 0] = i
            edge_pts_idx[count, 1] = 0
            edge_pts_idx[count, 2:map_size] = -1
            count += 1
        pass

    return normals_res, edge_pts_idx


def unit_normal(pt_0, pt_1, pt_2):
    a = pt_1[0:3] - pt_0[0:3]
    b = pt_2[0:3] - pt_0[0:3]
    normal = np.zeros(3)

    if np.all(a == 0.0) or np.all(b == 0.0):
        return normal
    else:
        normal = np.cross(a, b)

    return normal


# A3.6
def deri_surf(u, v, order, p, knot_u, knot_v, degree_u, degree_v):
    d = np.array([min(3, order), min(3, order)])

    count = knot_v.shape[0] - degree_v - 1
    span_u = span_linear(knot_u.shape[0] - degree_u - 1, knot_u, u)
    span_v = span_linear(knot_v.shape[0] - degree_v - 1, knot_v, v)

    skl = np.zeros([2, 2, 3])

    ders_u = basis_deri(u, order, span_u, degree_u, knot_u)
    ders_v = basis_deri(v, order, span_v, degree_v, knot_v)

    for k in range(0, d[0] + 1):
        temp = np.zeros([4, 3])
        for s in range(0, degree_v + 1):
            for r in range(0, degree_u + 1):
                temp[s][0] = temp[s][0] + ders_u[k][r] * p[((span_u - degree_u + r) * count) + (span_v - degree_v + s)][
                    0]
                temp[s][1] = temp[s][1] + ders_u[k][r] * p[((span_u - degree_u + r) * count) + (span_v - degree_v + s)][
                    1]
                temp[s][2] = temp[s][2] + ders_u[k][r] * p[((span_u - degree_u + r) * count) + (span_v - degree_v + s)][
                    2]

        dd = min(order - k, d[1])
        for l in range(0, dd + 1):
            # skl[(k * 3) + l][0] = 0.0 ; skl[(k * 3) + l][1] = 0.0 ; skl[(k * 3) + l][2] = 0.0
            for s in range(0, degree_v + 1):
                skl[k][l][0] = skl[k][l][0] + (ders_v[l][s] * temp[s][0])
                skl[k][l][1] = skl[k][l][1] + (ders_v[l][s] * temp[s][1])
                skl[k][l][2] = skl[k][l][2] + (ders_v[l][s] * temp[s][2])

    return skl


def basis_deri(u, order, span, degree, knot_v):
    left = np.empty([4], dtype=np.float32)
    right = np.empty([4], dtype=np.float32)
    ndu = np.full([4, 4], 1.0)  # ndu[0][0] = 1.0
    ders = np.zeros([2, 4])

    for j in range(1, degree + 1):
        left[j] = u - knot_v[span + 1 - j]
        right[j] = knot_v[span + j] - u
        saved = 0.0

        for r in range(0, j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]

            ndu[r][j] = saved + (right[r + 1] * temp)
            saved = left[j - r] * temp

        ndu[j][j] = saved

    for j in range(0, degree + 1):
        ders[0][j] = ndu[j][degree]

    a = np.full([4, 2], 1.0)

    for r in range(0, degree + 1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0

        for k in range(1, order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]

            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if (r - 1) <= pk:
                j2 = k - 1
            else:
                j2 = degree - r

            for j in range(j1, j2 + 1):
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += (a[s2][j] * ndu[rk + j][pk])

            if r <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                d += (a[s2][k] * ndu[r][pk])
            ders[k][r] = d

            # Switch rows
            j = s1
            s1 = s2
            s2 = j

    r = degree
    for k in range(1, order + 1):
        for j in range(0, degree + 1):
            ders[k][j] *= r
        r *= (degree - k)

    return ders


def basis_surf(u, degree, span_i, knot_v):
    left = np.empty([4])
    right = np.empty([4])

    N = np.empty([4])

    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knot_v[span_i + 1 - j]
        right[j] = knot_v[span_i + j] - u
        saved = 0.0

        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N


def surf_pt(u, v, p, knot_u, knot_v, degree_u, degree_v):
    count = knot_v.shape[0] - degree_v - 1
    span_u = span_linear(knot_u.shape[0] - degree_u - 1, knot_u, u)
    span_v = span_linear(knot_v.shape[0] - degree_v - 1, knot_v, v)

    N_u = basis_surf(u, degree_u, span_u, knot_u)
    N_v = basis_surf(v, degree_v, span_v, knot_v)

    S = np.zeros(3)

    uind = span_u - degree_u
    for i in range(0, degree_v + 1):
        temp = np.zeros(3)
        vind = span_v - degree_v + i

        for j in range(0, degree_u + 1):
            temp[0] = temp[0] + N_u[j] * p[((uind + j) * count) + vind][0]
            temp[1] = temp[1] + N_u[j] * p[((uind + j) * count) + vind][1]
            temp[2] = temp[2] + N_u[j] * p[((uind + j) * count) + vind][2]

        S[0] = S[0] + N_v[i] * temp[0]
        S[1] = S[1] + N_v[i] * temp[1]
        S[2] = S[2] + N_v[i] * temp[2]

    return S


# Finding span of
def span(cntrl_pt_count, degree, u, knot_v):
    if u == knot_v[-1]:
        return cntrl_pt_count

    low = degree
    high = cntrl_pt_count + 1
    mid = int((low + high) / 2)
    while u < knot_v[mid] or u >= knot_v[mid + 1]:
        if u < knot_v[mid]:
            high = mid
        else:
            low = mid
        mid = int((low + high) / 2)
    return mid
    pass


def span_linear(cntrl_pts_count, knot_vec, knot):
    span_lin = 0

    while span_lin < cntrl_pts_count and knot_vec[span_lin] <= knot:
        span_lin += 1

    return span_lin - 1


def multi_list(degree, knot_v):
    mul_list = []
    m = degree + 1
    while m < knot_v.shape[0] - degree - 1:
        n = knot_v[m]
        mul = find_multiplicity(n, knot_v)

        mul_list.append(mul)
        m += mul

    return mul_list
    pass


def find_multiplicity(u, knot_v):
    multi = 0

    for val in knot_v:
        if u == val:
            multi += 1
    return multi
    pass


def new_knot_v(degree, knot_v, multi, size):
    modi_knot_v = []
    modi_knot_array = np.empty([size])

    for i in range(0, degree + 1):  modi_knot_v.append(knot_v[i])

    ind = degree + 1

    for val in multi:

        for i in range(0, val):
            modi_knot_v.append(knot_v[ind])

        if val < degree:
            for i in range(0, degree - val):
                modi_knot_v.append(knot_v[ind])

        ind += val

    for i in range(0, degree + 1): modi_knot_v.append(knot_v[ind])

    return modi_knot_v


def alpha_knot_ins(multi, degree, span_i, uv, knot_v):
    alpha = np.empty([degree - multi, degree - multi])
    for j in range(1, degree - multi):
        l = span_i - degree + j

        for i in range(0, degree - j - multi):
            alpha[i][j] = (uv - knot_v[l + i]) / (knot_v[i + span_i + 1] - knot_v[l + i])
        pass
    pass


def new_control_points(cntrl_pt_count_u, cntrl_pt_count_v, modi_ctrl_pts, ctrl_pts, degree, span_i, multi, ins_count,
                       alpha):
    for row in range(0, cntrl_pt_count_v):

        # Save unaltered control points
        for i in range(0, span_i - degree):
            modi_ctrl_pts[i, row] = ctrl_pts[i, row]

        for i in range(span_i - multi, cntrl_pt_count_u):
            modi_ctrl_pts[i + ins_count, row] = ctrl_pts[i, row]

        # Save Auxiliary points
        auxi_pts = []
        for i in range(0, degree - multi):
            auxi_pts[i] = ctrl_pts[span_i - degree + i, row]

        for j in range(1, ins_count):
            l = span_i - degree + j

            for i in range(0, degree - j - multi):
                auxi_pts[i] = alpha[i, j] * auxi_pts[i + 1] + (1.0 - alpha[i, j] * auxi_pts[i])

            modi_ctrl_pts[l, row] = auxi_pts[0]
            modi_ctrl_pts[span_i + ins_count - j - multi, row] = auxi_pts[degree - j - multi]

        for i in range(l + 1, span_i - degree - 1):
            modi_ctrl_pts[i, row] = auxi_pts[i - l]

    pass


def knot_fill_vec(degree, knot_v, multi):
    size = (len(multi) * degree) + ((degree + 1) * 2)

    fill_vec = np.empty([size - knot_v.shape[0]])
    idx = 0

    ind = degree + 1

    for val in multi:

        if val < degree:
            for i in range(0, degree - val):
                fill_vec[idx] = knot_v[ind]
                idx += 1

        ind += val

    return fill_vec
    pass


def curve_knot_refine(cntrl_pt_count, degree, fill_vec, knot_v, ctrl_pts):
    m = cntrl_pt_count + degree + 1
    r = len(fill_vec) - 1

    modi_knot_v = np.empty([len(knot_v) + len(fill_vec)])
    modi_ctrl_pts = np.empty([len(modi_knot_v) - degree - 1, 4])

    a = span(cntrl_pt_count, degree, fill_vec[0], knot_v)
    b = span(cntrl_pt_count, degree, fill_vec[r], knot_v) + 1

    for j in range(0, a + 1):
        modi_knot_v[j] = knot_v[j]
    for j in range(b + degree, m + 1):
        modi_knot_v[j + r + 1] = knot_v[j]

    for j in range(0, a - degree + 1):
        modi_ctrl_pts[j] = ctrl_pts[j]
    for j in range(b - 1, cntrl_pt_count + 1):
        modi_ctrl_pts[j + r + 1] = ctrl_pts[j]

    i = b + degree - 1
    k = b + degree + r
    j = r

    while j >= 0:

        while fill_vec[j] <= knot_v[i] and i > a:
            modi_ctrl_pts[k - degree - 1] = ctrl_pts[i - degree - 1]
            modi_knot_v[k] = knot_v[i]
            k -= 1
            i -= 1

        modi_ctrl_pts[k - degree - 1] = modi_ctrl_pts[k - degree]

        for l in range(1, degree + 1):
            idx = k - degree + l
            alpha = modi_knot_v[k + l] - fill_vec[j]

            if abs(alpha) == 0.0:
                modi_ctrl_pts[idx - 1] = modi_ctrl_pts[idx]
            else:
                alpha = alpha / (modi_knot_v[k + l] - knot_v[i - degree + l])
                modi_ctrl_pts[idx - 1] = (alpha * modi_ctrl_pts[idx - 1]) + ((1.0 - alpha) * modi_ctrl_pts[idx])
        modi_knot_v[k] = fill_vec[j]
        k -= 1
        j -= 1

    return modi_knot_v, modi_ctrl_pts
    pass


def surf_knot_refine(cntrl_pt_count_u, cntrl_pt_count_v, degree_u, degree_v,
                     fill_vec_u, knot_vec_u, knot_vec_v, ctrl_pts, uv):
    if uv == 'u':
        r = len(fill_vec_u) - 1

        a = span(cntrl_pt_count_u, degree_u, fill_vec_u[0], knot_vec_u)
        b = span(cntrl_pt_count_u, degree_u, fill_vec_u[r], knot_vec_u) + 1

        modi_knot_vec_u = np.empty([len(knot_vec_u) + len(fill_vec_u)])
        modi_knot_vec_v = knot_vec_v

        modi_ctrl_pts = np.empty([(len(modi_knot_vec_u) - degree_u - 1) * (len(modi_knot_vec_v) - degree_v - 1), 4])

        # Unaltered knot vector values
        for j in range(0, a + 1):
            modi_knot_vec_u[j] = knot_vec_u[j]
        for j in range(b + degree_u, (cntrl_pt_count_u + degree_u + 1) + 1):
            modi_knot_vec_u[j + r + 1] = knot_vec_u[j]

        # Unaltered Control points
        for row in range(0, cntrl_pt_count_v + 1):
            for k in range(0, a - degree_u + 1):
                modi_ctrl_pts[(k * (cntrl_pt_count_v + 1)) + row] = ctrl_pts[(k * (cntrl_pt_count_v + 1)) + row]
            for k in range(b - 1, cntrl_pt_count_u + 1):
                modi_ctrl_pts[((k + r + 1) * (cntrl_pt_count_v + 1)) + row] = \
                    ctrl_pts[(k * (cntrl_pt_count_v + 1)) + row]

        i = b + degree_u - 1
        k = b + degree_u + r
        j = r
        while j >= 0:

            while fill_vec_u[j] <= knot_vec_u[i] and i > a:
                modi_knot_vec_u[k] = knot_vec_u[i]

                for row in range(0, cntrl_pt_count_v + 1):
                    modi_ctrl_pts[((k - degree_u - 1) * (cntrl_pt_count_v + 1)) + row] = \
                        ctrl_pts[((i - degree_u - 1) * (cntrl_pt_count_v + 1)) + row]
                k -= 1
                i -= 1

            for row in range(0, cntrl_pt_count_v + 1):
                modi_ctrl_pts[((k - degree_u - 1) * (cntrl_pt_count_v + 1)) + row] = \
                    modi_ctrl_pts[((k - degree_u) * (cntrl_pt_count_v + 1)) + row]

            for l in range(1, degree_u + 1):
                idx = k - degree_u + l
                alpha = modi_knot_vec_u[k + l] - fill_vec_u[j]

                if abs(alpha) == 0.0:
                    for row in range(0, cntrl_pt_count_v + 1):
                        modi_ctrl_pts[((idx - 1) * (cntrl_pt_count_v + 1)) + row] = \
                            modi_ctrl_pts[(idx * (cntrl_pt_count_v + 1)) + row]
                else:
                    alpha = alpha / (modi_knot_vec_u[k + l] - knot_vec_u[i - degree_u + l])
                    for row in range(0, cntrl_pt_count_v + 1):
                        modi_ctrl_pts[((idx - 1) * (cntrl_pt_count_v + 1)) + row] = \
                            (alpha * modi_ctrl_pts[((idx - 1) * (cntrl_pt_count_v + 1)) + row]) + \
                            ((1.0 - alpha) * modi_ctrl_pts[(idx * (cntrl_pt_count_v + 1)) + row])

            modi_knot_vec_u[k] = fill_vec_u[j]
            k -= 1
            j -= 1

        return modi_knot_vec_u, modi_ctrl_pts

    pass


def surf_knot_refine_2(cntrl_pt_count_u, cntrl_pt_count_v, degree_u, degree_v,
                       fill_vec_u, fill_vec_v, knot_vec_u, knot_vec_v, ctrl_pts, uv):
    '''
    :param cntrl_pt_count_u: Number of control points in u dir starting at 0
    :param cntrl_pt_count_v: Number of control points in v dir starting at 0
    :param degree_u: Degree in u dir
    :param degree_v: Degree in v dir
    :param fill_vec_u: Knot insertion vector in u dir in the form [X_0, X_1, ..... , X_r]
    :param fill_vec_v: Knot insertion vector in v dir in the form [X_0, X_1, ..... , X_r]
    :param knot_vec_u: Knot vector in u dir
    :param knot_vec_v: Knot vector in v dir
    :param ctrl_pts: CONTROL pOINTS with v changing first for every u
    :param uv: parameter that determines refinement direction in form of [u, v] = [
    :return:
    '''

    if uv[0] == 1:
        r = len(fill_vec_u) - 1

        a = span(cntrl_pt_count_u, degree_u, fill_vec_u[0], knot_vec_u)
        b = span(cntrl_pt_count_u, degree_u, fill_vec_u[r], knot_vec_u) + 1

        a_lin = span_linear(cntrl_pt_count_u, knot_vec_u, fill_vec_u[0])
        b_lin = span_linear(cntrl_pt_count_u, knot_vec_u, fill_vec_u[r]) + 1

        modi_knot_vec_u = np.empty([len(knot_vec_u) + len(fill_vec_u)])
        modi_knot_vec_v = knot_vec_v

        modi_ctrl_pts = np.empty([(len(modi_knot_vec_u) - degree_u - 1) * (len(modi_knot_vec_v) - degree_v - 1), 4])

        # Unaltered knot vector values
        for j in range(0, a + 1):
            modi_knot_vec_u[j] = knot_vec_u[j]
        for j in range(b + degree_u, (cntrl_pt_count_u + degree_u + 1) + 1):
            modi_knot_vec_u[j + r + 1] = knot_vec_u[j]

        # Unaltered Control points
        for row in range(0, cntrl_pt_count_v + 1):
            for k in range(0, a - degree_u + 1):
                modi_ctrl_pts[(k * (cntrl_pt_count_v + 1)) + row] = ctrl_pts[(k * (cntrl_pt_count_v + 1)) + row]
            for k in range(b - 1, cntrl_pt_count_u + 1):
                modi_ctrl_pts[((k + r + 1) * (cntrl_pt_count_v + 1)) + row] = \
                    ctrl_pts[(k * (cntrl_pt_count_v + 1)) + row]

        i = b + degree_u - 1
        k = b + degree_u + r
        j = r
        while j >= 0:

            while fill_vec_u[j] <= knot_vec_u[i] and i > a:
                modi_knot_vec_u[k] = knot_vec_u[i]

                for row in range(0, cntrl_pt_count_v + 1):
                    modi_ctrl_pts[((k - degree_u - 1) * (cntrl_pt_count_v + 1)) + row] = \
                        ctrl_pts[((i - degree_u - 1) * (cntrl_pt_count_v + 1)) + row]
                k -= 1
                i -= 1

            for row in range(0, cntrl_pt_count_v + 1):
                modi_ctrl_pts[((k - degree_u - 1) * (cntrl_pt_count_v + 1)) + row] = \
                    modi_ctrl_pts[((k - degree_u) * (cntrl_pt_count_v + 1)) + row]

            for l in range(1, degree_u + 1):
                idx = k - degree_u + l
                alpha = modi_knot_vec_u[k + l] - fill_vec_u[j]

                if abs(alpha) == 0.0:
                    for row in range(0, cntrl_pt_count_v + 1):
                        modi_ctrl_pts[((idx - 1) * (cntrl_pt_count_v + 1)) + row] = \
                            modi_ctrl_pts[(idx * (cntrl_pt_count_v + 1)) + row]
                else:
                    alpha = alpha / (modi_knot_vec_u[k + l] - knot_vec_u[i - degree_u + l])
                    for row in range(0, cntrl_pt_count_v + 1):
                        modi_ctrl_pts[((idx - 1) * (cntrl_pt_count_v + 1)) + row] = \
                            (alpha * modi_ctrl_pts[((idx - 1) * (cntrl_pt_count_v + 1)) + row]) + \
                            ((1.0 - alpha) * modi_ctrl_pts[(idx * (cntrl_pt_count_v + 1)) + row])

            modi_knot_vec_u[k] = fill_vec_u[j]
            k -= 1
            j -= 1

        cntrl_pt_count_u = len(modi_knot_vec_u) - degree_u - 2

    else:
        modi_knot_vec_u = knot_vec_u
        cntrl_pt_count_u = len(modi_knot_vec_u) - degree_u - 2
        modi_ctrl_pts = ctrl_pts

    if uv[1] == 1:

        flip_ctrl_pts = flip_orientation(modi_ctrl_pts, cntrl_pt_count_u + 1, cntrl_pt_count_v + 1, flip_dir='v')

        r = len(fill_vec_v) - 1

        a = span(cntrl_pt_count_v, degree_v, fill_vec_v[0], knot_vec_v)
        b = span(cntrl_pt_count_v, degree_v, fill_vec_v[r], knot_vec_v) + 1

        modi_knot_vec_v = np.empty([len(knot_vec_v) + len(fill_vec_v)])
        # modi_knot_vec_u = knot_vec_u

        temp_ctrl_pts = np.empty([(len(modi_knot_vec_u) - degree_u - 1) * (len(modi_knot_vec_v) - degree_v - 1), 4])

        # Unaltered knot vector values
        for j in range(0, a + 1):
            modi_knot_vec_v[j] = knot_vec_v[j]
        for j in range(b + degree_v, (cntrl_pt_count_v + degree_v + 1) + 1):
            modi_knot_vec_v[j + r + 1] = knot_vec_v[j]

        # Unaltered Control points
        for row in range(0, cntrl_pt_count_u + 1):
            for k in range(0, a - degree_v + 1):
                temp_ctrl_pts[(k * (cntrl_pt_count_u + 1)) + row] = flip_ctrl_pts[(k * (cntrl_pt_count_u + 1)) + row]
            for k in range(b - 1, cntrl_pt_count_v + 1):
                temp_ctrl_pts[((k + r + 1) * (cntrl_pt_count_u + 1)) + row] = \
                    flip_ctrl_pts[(k * (cntrl_pt_count_u + 1)) + row]

        i = b + degree_v - 1
        k = b + degree_v + r
        j = r
        while j >= 0:

            while fill_vec_v[j] <= knot_vec_v[i] and i > a:
                modi_knot_vec_v[k] = knot_vec_v[i]

                for row in range(0, cntrl_pt_count_u + 1):
                    temp_ctrl_pts[((k - degree_v - 1) * (cntrl_pt_count_u + 1)) + row] = \
                        flip_ctrl_pts[((i - degree_v - 1) * (cntrl_pt_count_u + 1)) + row]
                k -= 1
                i -= 1

            for row in range(0, cntrl_pt_count_u + 1):
                temp_ctrl_pts[((k - degree_v - 1) * (cntrl_pt_count_u + 1)) + row] = \
                    temp_ctrl_pts[((k - degree_v) * (cntrl_pt_count_u + 1)) + row]

            for l in range(1, degree_v + 1):
                idx = k - degree_v + l
                alpha = modi_knot_vec_v[k + l] - fill_vec_v[j]

                if abs(alpha) == 0.0:
                    for row in range(0, cntrl_pt_count_u + 1):
                        temp_ctrl_pts[((idx - 1) * (cntrl_pt_count_u + 1)) + row] = \
                            temp_ctrl_pts[(idx * (cntrl_pt_count_u + 1)) + row]
                else:
                    alpha = alpha / (modi_knot_vec_v[k + l] - knot_vec_v[i - degree_v + l])
                    for row in range(0, cntrl_pt_count_u + 1):
                        temp_ctrl_pts[((idx - 1) * (cntrl_pt_count_u + 1)) + row] = \
                            (alpha * temp_ctrl_pts[((idx - 1) * (cntrl_pt_count_u + 1)) + row]) + \
                            ((1.0 - alpha) * temp_ctrl_pts[(idx * (cntrl_pt_count_u + 1)) + row])

            modi_knot_vec_v[k] = fill_vec_v[j]
            k -= 1
            j -= 1

        cntrl_pt_count_v = len(modi_knot_vec_v) - degree_v - 2

        decompo_ctrl_pts = flip_orientation(temp_ctrl_pts, cntrl_pt_count_u + 1, cntrl_pt_count_v + 1, flip_dir='u')

    else:
        modi_knot_vec_v = knot_vec_v
        cntrl_pt_count_v = len(modi_knot_vec_v) - degree_u - 2
        decompo_ctrl_pts = modi_ctrl_pts

    return modi_knot_vec_u, modi_knot_vec_v, cntrl_pt_count_u, cntrl_pt_count_v, decompo_ctrl_pts


def decompose_curve(cntrl_pt_c, degree, knot_v, ctrl_pts):
    m = cntrl_pt_c + degree + 1
    a = degree
    b = degree + 1
    nb = 0

    modi_ctrl_pts = np.zeros([len(knot_v) - (2 * (degree + 1)) + 1, degree + 1, 4])
    alphas = np.empty([25])

    for i in range(0, degree + 1):
        modi_ctrl_pts[nb][i] = ctrl_pts[i]

    while b < m:
        i = b
        while b < m and knot_v[b + 1] == knot_v[b]:
            b += 1
        mult = b - i + 1

        if mult < degree:
            numer = knot_v[b] - knot_v[a]

            for j in range(degree, mult, -1):
                alphas[j - mult - 1] = numer / (knot_v[a + j] - knot_v[a])
            r = degree - mult

            for j in range(1, r + 1):
                save = r - j
                s = mult + j

                for k in range(degree, s + 1, -1):
                    alpha = alphas[k - s]
                    modi_ctrl_pts[nb][k] = (alpha * modi_ctrl_pts[nb][k]) + (
                            (1.0 - alpha) * modi_ctrl_pts[nb][k - 1])
                if b < m:
                    modi_ctrl_pts[nb + 1][save] = modi_ctrl_pts[nb][degree]

        nb += 1
        if b < m:
            for i in range(degree - mult, degree + 1):
                modi_ctrl_pts[nb][i] = ctrl_pts[b - degree + i]
            a = b
            b += 1

    pass


def decompose_surface(degree_u, degree_v, cntrl_pt_c_u, knot_vec_u, knot_vec_v, cntrl_pt_c_v, ctrl_pts, uv):
    if uv == 'u':
        a = degree_u
        b = degree_u + 1
        nb = 0

        modi_ctrl_pts = np.empty([20, (degree_u + 1) * (cntrl_pt_c_v + 1), 4])

        for i in range(0, degree_u + 1):
            for row in range(0, cntrl_pt_c_v + 1):
                modi_ctrl_pts[nb][(i * (degree_u + 1)) + row] = ctrl_pts[(i * (degree_u + 1)) + row]

        while b < cntrl_pt_c_v:

            i = b
            while b < cntrl_pt_c_v and knot_vec_u[b + 1] == knot_vec_v[b]:
                b += 1
            mult = b - i + 1

            if mult < degree_u:
                pass

def flip_orientation(ctrl_pts, count_u, count_v, flip_dir):
    if flip_dir == 'u':

        modi_ctrl_pts = np.empty(shape=ctrl_pts.shape)

        for i in range(0, count_u):
            for j in range(0, count_v):
                modi_ctrl_pts[i * count_v + j] = ctrl_pts[j * count_u + i]

        return modi_ctrl_pts

    if flip_dir == 'v':

        modi_ctrl_pts = np.empty(shape=ctrl_pts.shape)

        for i in range(0, count_v):
            for j in range(0, count_u):
                modi_ctrl_pts[i * count_u + j] = ctrl_pts[j * count_v + i]

        return modi_ctrl_pts


def split_surf(knot_vec_u, knot_vec_v, degree_u, degree_v, ctrl_pts, count_v):
    split_c_u = int(((len(knot_vec_u) - (2 * (degree_u + 1))) / degree_u) + 1)
    split_c_v = int(((len(knot_vec_v) - (2 * (degree_v + 1))) / degree_v) + 1)

    decompose_surface = np.empty([split_c_u * split_c_v, (degree_u + 1) * (degree_v + 1), 4])

    for i in range(0, split_c_u):
        for j in range(0, split_c_v):
            for k in range(0, degree_u + 1):
                for l in range(0, degree_v + 1):
                    decompose_surface[i * split_c_v + j][k * (degree_v + 1) + l] = \
                        ctrl_pts[(i * (count_v * 2)) + (j * degree_v) + (k * count_v) + l]

    return decompose_surface

    pass


def post_pross(param, rays, idx, count, strain, vox_count, vox_size, vox_model, origin):
    for i in range(0, rays.shape[0]):
        temp_elem = param[idx[i]][2]
        temp_vox = param[idx[i]][6]

        for j in range(1, count[i]):
            if param[idx[i] + j][2] == temp_elem:
                if param[idx[i] + j][6] == temp_vox:
                    pass
                else:
                    gap = param[idx[i] + j][6] - temp_vox + 1
                    for k in range(0, gap):
                        vox_gauss_val = min_dist_vox(temp_vox + k, origin, vox_count, vox_size,
                                                     rays[i], strain[int(temp_elem)])
                        pass
        pass
    pass


def min_dist_vox(vox, origin, vox_count, vox_size, ray, strain):
    vox_pts = [origin[0] + ((ray // vox_count) * vox_size),
               origin[1] + ((ray % vox_count) * vox_size),
               origin[2] + (vox * vox_size)]
    min_dist = np.inf
    idx = 0

    for i in range(0, 27):
        dist = np.sqrt(pow(vox_pts[0] - strain[i][0], 2) +
                       pow(vox_pts[1] - strain[i][1], 2) +
                       pow(vox_pts[2] - strain[i][2], 2))
        if dist < min_dist:
            min_dist = dist
            idx = i

    return strain[idx][3]
