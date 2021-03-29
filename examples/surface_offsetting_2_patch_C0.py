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

import cpu_eval as cpu


def main():
    timing = []
    eval_pts_size = 20
    eval_pts_size_HD = 100

    off_dist = 0.1
    cuda = False

    # Cardiac Model Surfaces
    num_ctrl_pts1 = 4
    num_ctrl_pts2 = 4
    ctrl_pts = np.load('CNTRL_PTS_2_Chamber.npy').astype('float32')
    # ctrl_pts[:, :, :, -1] = 1.0
    element_array = np.array([24, 30])
    ctrl_pts = ctrl_pts[0, :, :, :3]
    # ctrl_pts = ctrl_pts[0, :, :, :]
    knot_u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    knot_v = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    edge_pts_count = 2 * (num_ctrl_pts1 + num_ctrl_pts2 - 2)

    ctrl_pts_normal = np.empty([element_array.size, num_ctrl_pts1 * num_ctrl_pts2, 3])
    edge_pts_idx = np.empty([element_array.size, edge_pts_count, 4], dtype=np.int)

    edge_ctrlpts_map = np.empty([element_array.size, ctrl_pts.shape[1], 3], dtype=np.uint)
    if element_array.size > 1:
        edge_ctrlpts_map = off.map_ctrl_point(ctrl_pts[element_array, :, :3])

    for i in range(element_array.size):
        ctrl_pts_normal[i], edge_pts_idx[i] = cpu.compute_cntrlpts_normals(ctrl_pts[element_array[i]], num_ctrl_pts1, num_ctrl_pts2, edge_pts_count, 4)

    edge_pts_idx = cpu.map_edge_points(ctrl_pts[element_array], edge_pts_idx)

    ctrl_pts_normal = cpu.normals_reassign(edge_pts_idx, ctrl_pts_normal)

    temp_2 = cpu.compute_model_offset(ctrl_pts[element_array], ctrl_pts_normal, num_ctrl_pts1, num_ctrl_pts2, 3, knot_u, knot_v, off_layer=2, thickness=-off_dist)
    weights = torch.ones(element_array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    ctrl_pts_offset = torch.cat((torch.from_numpy(np.reshape(temp_2,[element_array.size, num_ctrl_pts1, num_ctrl_pts2,3])), weights), axis=-1)

    off_pts = off.compute_surf_offset(ctrl_pts[element_array], knot_u, knot_v, 3, 3, eval_pts_size, off_dist)
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
    ctrl_pts_offsetSurf = layer_2(ctrl_pts_offset)
    basesurf_pts = basesurf.detach().cpu().numpy().squeeze()
    ctrl_pts_offsetSurf_Pts = ctrl_pts_offsetSurf.detach().cpu().numpy().squeeze()
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

        if element_array.size > 1:
            pass
            with torch.no_grad():
                # test = inp_ctrl_pts.numpy()
                for k in range(edge_ctrlpts_map.shape[0]):
                    for h in range(edge_ctrlpts_map.shape[1]):
                        if edge_ctrlpts_map[k][h][0] != 0:
                            temp = torch.zeros(4)

                            temp += inp_ctrl_pts.data[k][h % num_ctrl_pts2][h // num_ctrl_pts2]
                            for j in range(edge_ctrlpts_map[k][h][0]):
                                elem = edge_ctrlpts_map[k][h][2 * j + 1]
                                pts = edge_ctrlpts_map[k][h][2 * j + 2]
                                inp_ctrl_pts.data[elem][pts % num_ctrl_pts2][pts // num_ctrl_pts2] = temp

        if i % 10000 == 0:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')

            target_mpl = np.reshape(target.cpu().numpy().squeeze(), [2, eval_pts_size * eval_pts_size, 3])
            predicted = out.detach().cpu().numpy().squeeze()
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            predctrlpts = predctrlpts[:, :, :, :3] / predctrlpts[:, :, :, 3:]
            surf1 = ax.scatter(target_mpl[0, :, 0], target_mpl[0, :, 1], target_mpl[0, :, 2], s=1.0, color='red', label='Target Offset surface')
            surf1 = ax.scatter(target_mpl[1, :, 0], target_mpl[1, :, 1], target_mpl[1, :, 2], s=1.0, color='red', label='Target Offset surface')
            # surf1 = ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed',
            #                           color='orange', label='Target CP')

            surf2 = ax.plot_surface(predicted[0, :, :, 0], predicted[0, :, :, 1], predicted[0, :, :, 2], color='green', label='Predicted Surface')
            # surf2 = ax.plot_wireframe(predctrlpts[0, :, :, 0], predctrlpts[0, :, :, 1], predctrlpts[0, :, :, 2],
            #                           linestyle='dashed', color='orange', label='Predicted CP')

            surf3 = ax.plot_surface(basesurf_pts[0, :, :, 0], basesurf_pts[0, :, :, 1], basesurf_pts[0, :, :, 2], color='blue')

            surf2 = ax.plot_surface(predicted[1, :, :, 0], predicted[1, :, :, 1], predicted[1, :, :, 2], color='red', label='Predicted Surface')
            # surf2 = ax.plot_wireframe(predctrlpts[1, :, :, 0], predctrlpts[1, :, :, 1], predctrlpts[1, :, :, 2],
            #                           linestyle='dashed', color='purple', label='Predicted CP')

            surf3 = ax.plot_surface(basesurf_pts[1, :, :, 0], basesurf_pts[1, :, :, 1], basesurf_pts[1, :, :, 2], color='pink')
            # surf3 = ax.plot_wireframe(ctrl_pts_2[:, :, 0], ctrl_pts_2[:, :, 1], ctrl_pts_2[:, :, 2],
            #                           linestyle='dashed', color='pink', label='Predicted CP')

            surf4 = ax.plot_surface(ctrl_pts_offsetSurf_Pts[0, :, :, 0], ctrl_pts_offsetSurf_Pts[0, :, :, 1], ctrl_pts_offsetSurf_Pts[0, :, :, 2], color='yellow')
            surf4 = ax.plot_surface(ctrl_pts_offsetSurf_Pts[1, :, :, 0], ctrl_pts_offsetSurf_Pts[1, :, :, 1], ctrl_pts_offsetSurf_Pts[1, :, :, 2], color='yellow')
            # ax.set_zlim(-1,3)
            # ax.set_xlim(-1,4)
            # ax.set_ylim(-2,2)
            ax.azim = 44
            ax.dist = 6.5
            ax.elev = 58

            ax.set_box_aspect([1, 2, 1])
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

        if loss.item() < 5e-6:
             break
        pbar.set_description("Mse Loss is %s: %s" % (i+1, loss.item()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    target_mpl = np.reshape(target.cpu().numpy().squeeze(), [2, eval_pts_size * eval_pts_size, 3])
    predicted = out.detach().cpu().numpy().squeeze()
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    predctrlpts = predctrlpts[:, :, :, :3] / predctrlpts[:, :, :, 3:]
    # surf1 = ax.scatter(target_mpl[0, :, 0], target_mpl[0, :, 1], target_mpl[0, :, 2], s=1.0, color='black',
    #                    label='Target Offset surface')
    # surf1 = ax.scatter(target_mpl[1, :, 0], target_mpl[1, :, 1], target_mpl[1, :, 2], s=1.0, color='black',
    #                    label='Target Offset surface')

    surf2 = ax.plot_surface(predicted[0, :, :, 0], predicted[0, :, :, 1], predicted[0, :, :, 2], color='green',
                            label='Predicted Surface', alpha=0.7)
    surf2 = ax.plot_wireframe(predctrlpts[0, :, :, 0], predctrlpts[0, :, :, 1], predctrlpts[0, :, :, 2],
                              linestyle='dashed', color='orange', label='Predicted CP')

    surf3 = ax.plot_surface(basesurf_pts[0, :, :, 0], basesurf_pts[0, :, :, 1], basesurf_pts[0, :, :, 2], color='blue')

    surf2 = ax.plot_surface(predicted[1, :, :, 0], predicted[1, :, :, 1], predicted[1, :, :, 2], color='red',
                            label='Predicted Surface', alpha=0.7)
    surf2 = ax.plot_wireframe(predctrlpts[1, :, :, 0], predctrlpts[1, :, :, 1], predctrlpts[1, :, :, 2],
                              linestyle='dashed', color='orange', label='Predicted CP')

    surf3 = ax.plot_surface(basesurf_pts[1, :, :, 0], basesurf_pts[1, :, :, 1], basesurf_pts[1, :, :, 2], color='pink')

    # surf4 = ax.plot_surface(ctrl_pts_offsetSurf_Pts[0, :, :, 0], ctrl_pts_offsetSurf_Pts[0, :, :, 1],
    #                         ctrl_pts_offsetSurf_Pts[0, :, :, 2], color='yellow')
    # surf4 = ax.plot_surface(ctrl_pts_offsetSurf_Pts[1, :, :, 0], ctrl_pts_offsetSurf_Pts[1, :, :, 1],
    #                         ctrl_pts_offsetSurf_Pts[1, :, :, 2], color='yellow')

    # ax.set_zlim(-1,3)
    # ax.set_xlim(-1,4)
    # ax.set_ylim(-2,2)
    ax.azim = 14
    ax.dist = 6.5
    ax.elev = 64

    # ax.set_box_aspect([1, 2, 1])
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

    # layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3,
    #                    out_dim_u=eval_pts_size_HD, out_dim_v=eval_pts_size_HD)
    weights = torch.ones(element_array.size, num_ctrl_pts1, num_ctrl_pts2, 1)
    out = layer_2(torch.cat((inp_ctrl_pts, weights), axis=-1))
    out_2 = out.view(2, eval_pts_size_HD * eval_pts_size_HD, 3)
    target_2 = target.view(2, eval_pts_size * eval_pts_size, 3)

    print('Offset Distance is  ==  ', off_dist)

    print('max Size is  ==  ', max_size)

    loss_CP_PC, _ = chamfer_distance(ctrl_pts_offsetSurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), target_2)
    print('Chamfer loss  --  Control Point , Point Cloud   ==  ', loss_CP_PC * 10000)

    loss_Pred_PC, _ = chamfer_distance(target_2, out_2)
    print('Chamfer loss  --  Predicted, Point Cloud   ==  ', loss_Pred_PC * 10000)

    loss_CP_Base, _ = chamfer_distance(ctrl_pts_offsetSurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), basesurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3))
    print('Chamfer loss  --  Control Point, Base   ==  ', loss_CP_Base * 10000)

    loss_Pred_Base, _ = chamfer_distance(basesurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), out_2)
    print('Chamfer loss  --  Predicted , Base   ==  ', loss_Pred_Base * 10000)

    loss_CP_Pred, _ = chamfer_distance(ctrl_pts_offsetSurf.view(element_array.size, eval_pts_size_HD * eval_pts_size_HD, 3), out_2)
    print('Chamfer loss  --  Control Point , Predicted   ==  ', loss_CP_Pred * 10000)

if __name__ == '__main__':
    main()
