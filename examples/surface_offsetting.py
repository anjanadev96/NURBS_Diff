import torch
import numpy as np
torch.manual_seed(120)
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from torch_nurbs_eval.surf_eval import SurfEval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import offset_eval as off
import sys
sys.path.insert(0, 'D:/Reasearch Data and Codes/ray_inter_nurbs')
import cpu_eval as cpu




def main():
    timing = []
    eval_pts_size = 50
    eval_pts_size_HD = 100

    off_dist = 1.5
    cuda = False

    # Turbine Blade Surfaces
    # num_ctrl_pts1 = 50
    # num_ctrl_pts2 = 24
    # ctrl_pts = np.load('TurbineBladectrlpts.npy').astype('float32')
    # element_array = np.array([0, 1])
    # knot_u = np.load('TurbineKnotU.npy')
    # knot_v = np.load('TurbineKnotV.npy')

    # Cardiac Model Surfaces
    # num_ctrl_pts1 = 4
    # num_ctrl_pts2 = 4
    # ctrl_pts = np.load('cntrl_pts_2_Chamber.npy').astype('float32')
    # # ctrl_pts[:, :, :, -1] = 1.0
    # element_array = np.array([24, 30])
    # ctrl_pts = ctrl_pts[0, :, :, :3]
    # knot_u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    # knot_v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    # Roof Files
    # num_ctrl_pts1 = 6
    # num_ctrl_pts2 = 6
    # ctrl_pts = np.load('ctrlptsRoof.npy').astype('float32')
    # element_array = np.array([0])
    # ctrl_pts = ctrl_pts[:, :, :3]
    # knot_u = np.array([0.0, 0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0])
    # knot_v = np.array([0.0, 0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0])

    # Double Curve
    num_ctrl_pts1 = 6
    num_ctrl_pts2 = 6
    ctrl_pts = np.load('DoubleCurve.npy').astype('float32')
    element_array = np.array([0])
    ctrl_pts = np.reshape(ctrl_pts, [1, ctrl_pts.shape[0], 3])
    knot_u = np.array([0.0, 0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0])
    knot_v = np.array([0.0, 0.0, 0.0, 0.0, 0.33, 0.67, 1.0, 1.0, 1.0, 1.0])

    edge_pts_count = 2 * (num_ctrl_pts1 + num_ctrl_pts2 - 2)
    ctrl_pts_normal = np.empty([element_array.size, num_ctrl_pts1 * num_ctrl_pts2, 3])
    edge_pts_idx = np.empty([element_array.size, edge_pts_count, 4], dtype=np.int)
    edge_ctrl_pts_map = np.empty([element_array.size, ctrl_pts.shape[1], 3], dtype=np.uint)
    if element_array.size > 1:
        edge_ctrl_pts_map = off.map_ctrl_point(ctrl_pts[element_array, :, :3])

    for i in range(element_array.size):
        ctrl_pts_normal[i], edge_pts_idx[i] = cpu.compute_cntrl_pts_normals(ctrl_pts[element_array[i]], num_ctrl_pts1, num_ctrl_pts2, edge_pts_count, 4)

    edge_pts_idx = cpu.map_edge_points(ctrl_pts[element_array], edge_pts_idx)

    ctrl_pts_normal = cpu.Normals_reassign(edge_pts_idx, ctrl_pts_normal)

    temp_2 = cpu.compute_model_offset(ctrl_pts[element_array], ctrl_pts_normal, num_ctrl_pts1, num_ctrl_pts2, 3, knot_u, knot_v, off_layer=2, thickness=off_dist)
    weights = torch.ones(element_array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    ctrlptsoffset = torch.cat((torch.from_numpy(np.reshape(temp_2,[element_array.size, num_ctrl_pts1, num_ctrl_pts2,3])), weights), axis=-1)

    off_pts = off.compute_surf_offset(ctrl_pts[element_array], knot_u, knot_v, 3, 3, eval_pts_size, -off_dist)
    max_size = off.max_size(off_pts)
    target = torch.from_numpy(np.reshape(off_pts, [element_array.size, eval_pts_size, eval_pts_size, 3]))

    temp = np.reshape(ctrl_pts[element_array], [ctrl_pts[element_array].shape[0], num_ctrl_pts1, num_ctrl_pts2, 3])
    isolate_pts = torch.from_numpy(temp)
    weights = torch.ones(element_array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    base_inp_ctrl_pts = torch.cat((isolate_pts, weights), axis=-1)
    # inp_ctrl_pts = torch.nn.Parameter(isolate_pts)
    inp_ctrl_pts = torch.nn.Parameter(base_inp_ctrl_pts)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=eval_pts_size, out_dim_v=eval_pts_size)
    layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=eval_pts_size_HD, out_dim_v=eval_pts_size_HD)
    basesurf = layer_2(base_inp_ctrl_pts)

    ctrlptsoffsetsurf = layer_2(ctrlptsoffset)
    basesurf_pts = basesurf.detach().cpu().numpy().squeeze()
    ctrlptsoffsetsurf_pts = ctrlptsoffsetsurf.detach().cpu().numpy().squeeze()
    # ctrl_pts_2 = np.reshape(ctrl_pts, [num_ctrl_pts1, num_ctrl_pts2, 3])

    opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.01)
    pbar = tqdm(range(20000))
    for i in pbar:
        opt.zero_grad()
        # weights = torch.ones(element_array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
        # out = layer(torch.cat((inp_ctrl_pts, weights), axis=-1))
        out = layer(inp_ctrl_pts)
        loss = ((target-out)**2).mean()
        loss.backward()
        opt.step()

        if (i+1) % 100000 == 0:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

            # target_mpl = np.reshape(target.cpu().numpy().squeeze(), [eval_pts_size * eval_pts_size, 3])
            predicted = out.detach().cpu().numpy().squeeze()
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            predctrlpts = predctrlpts[:, :, :3] / predctrlpts[:, :, 3:]
            # surf1 = ax.scatter(target_mpl[:, 0], target_mpl[:, 1], target_mpl[:, 2], s=1.0, color='red', label='Target offset surface')
            # surf1 = ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed',
            #                           color='orange', label='Target CP')

            surf2 = ax.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green', label='Predicted Surface')
            surf2 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
                                      linestyle='dashed', color='orange', label='Predicted CP')

            surf3 = ax.plot_surface(basesurf_pts[:, :, 0], basesurf_pts[:, :, 1], basesurf_pts[:, :, 2], color='blue', alpha=0.5)
            surf3 = ax.plot_wireframe(temp[0, :, :, 0], temp[0, :, :, 1], temp[0, :, :, 2], linestyle='dashed', color='pink', label='Predicted CP')

            # surf4 = ax.plot_surface(ctrlptsoffsetsurf_pts[:, :, 0], ctrlptsoffsetsurf_pts[:, :, 1], ctrlptsoffsetsurf_pts[:, :, 2], color='yellow')

            # ax.set_zlim(-1,3)
            # ax.set_xlim(-1,4)
            # ax.set_ylim(-2,2)

            ax.set_box_aspect([1, 1, 0.5])
            ax.azim = 46
            ax.dist = 10
            ax.elev = 30
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax._axis3don = False
            # ax.legend()

            # ax.set_aspect(1)
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.tight_layout()
            plt.show()

        if loss.item() < 5e-4:
            break
        pbar.set_description("Mse Loss is %s: %s" % (i+1, loss.item()))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d', adjustable='box')
    out = layer(inp_ctrl_pts)
    predicted = out.detach().cpu().numpy().squeeze()
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    predctrlpts = predctrlpts[:, :, :3] / predctrlpts[:, :, 3:]
    surf2 = ax.plot_surface(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green',
                            label='Predicted Surface', alpha=0.6)
    surf2 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
                              linestyle='dashed', color='orange', label='Predicted CP')
    surf3 = ax.plot_surface(basesurf_pts[:, :, 0], basesurf_pts[:, :, 1], basesurf_pts[:, :, 2], color='blue')

    # surf4 = ax.plot_surface(ctrlptsoffsetsurf_pts[:, :, 0], ctrlptsoffsetsurf_pts[:, :, 1],
    #                         ctrlptsoffsetsurf_pts[:, :, 2], color='yellow')

    ax.azim = 45
    ax.dist = 10
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_box_aspect([1, 1, 0.5])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    # ax.legend(loc='upper left')

    # ax.set_aspect(1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plt.show()

    # layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3,
    #                    out_dim_u=100, out_dim_v=100)
    weights = torch.ones(element_array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    out = layer_2(torch.cat((inp_ctrl_pts, weights), axis=-1))
    out_2 = out.view(1, eval_pts_size_HD * eval_pts_size_HD, 3)
    target_2 = target.view(1, eval_pts_size * eval_pts_size, 3)

    print('offset Distance is  ==  ', off_dist)

    print('max Size is  ==  ', max_size)

    loss_cp_pc, _ = chamfer_distance(ctrlptsoffsetsurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), target_2)
    print('Chamfer loss  --  Control Point , Point Cloud   ==  ', loss_cp_pc * 10000)

    loss_pred_pc, _ = chamfer_distance(target_2, out_2)
    print('Chamfer loss  --  Predicted, Point Cloud   ==  ', loss_pred_pc * 10000)

    loss_cp_base, _ = chamfer_distance(ctrlptsoffsetsurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), basesurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3))
    print('Chamfer loss  --  Control Point, Base   ==  ', loss_cp_base * 10000)

    loss_pred_base, _ = chamfer_distance(basesurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), out_2)
    print('Chamfer loss  --  Predicted , Base   ==  ', loss_pred_base * 10000)

    loss_cp_pred, _ = chamfer_distance(ctrlptsoffsetsurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), out_2)
    print('Chamfer loss  --  Control Point , Predicted   ==  ', loss_cp_pred * 10000)


if __name__ == '__main__':
    main()
