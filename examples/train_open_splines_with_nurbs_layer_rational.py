import json
import logging
import os
import sys
from shutil import copyfile

import numpy as np
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from train_open_spline_utils.read_config import Config
from train_open_spline_utils.src.dataset import DataSetControlPointsPoisson
from train_open_spline_utils.src.dataset import generator_iter
from train_open_spline_utils.src.loss import (
    control_points_permute_reg_loss,
)
from train_open_spline_utils.src.loss import laplacian_loss
from train_open_spline_utils.src.loss import (
    uniform_knot_bspline,
    # spline_reconstruction_loss_one_sided,
    chamfer_distance_one_side
)
from train_open_spline_utils.src.model import DGCNNControlPoints2
from train_open_spline_utils.src.utils import rescale_input_outputs
from torch_nurbs_eval.surf_eval import SurfEval
from pytorch3d.loss import chamfer_distance

def nurbs_reconstruction_loss_one_sided(nu, nv, output, points, config, side=1):
    """
    Spline reconsutruction loss defined using chamfer distance, but one
    sided either gt surface can cover the prediction or otherwise, which
    is defined by the network. side=1 means prediction can cover gt.
    :param nu: spline basis function in u direction.
    :param nv: spline basis function in v direction.
    :param points: points sampled over the spline.
    :param config: object of configuration class for extra parameters. 
    """
    reconst_points = []
    batch_size = output.shape[0]
    c_size_u = output.shape[1]
    c_size_v = output.shape[2]
    grid_size_u = nu.shape[0]
    grid_size_v = nv.shape[0]

    output = output.view(config.batch_size, config.grid_size, config.grid_size, 4)
    points = points.permute(0, 2, 1)

    # ones = torch.ones((output.size(0),
    #         output.size(1),
    #         output.size(2),1), 
    #         requires_grad=True).cuda()

    # output = torch.cat((output,ones),-1)
    reconst_points = nurbs_layer(output)
    reconst_points = reconst_points.view(config.batch_size, grid_size_u * grid_size_v, 3)
    dist = chamfer_distance_one_side(reconst_points, points, side)
    return dist, reconst_points


config = Config('./train_open_spline_utils/configs/config_open_splines.yml')

model_name = config.model_path.format(
    config.mode,
    config.num_points,
    config.loss_weight,
    config.batch_size,
    config.lr,
    config.num_train,
    config.num_test,
    config.loss_weight,
)

print("Model name: ", model_name)
print(config.config)

userspace = os.path.dirname(os.path.abspath(__file__))
configure("logs/tensorboard/{}".format(model_name), flush_secs=5)


control_decoder = DGCNNControlPoints2(20, num_points=10, mode=config.mode)

if torch.cuda.device_count() > 1:
    control_decoder = torch.nn.DataParallel(control_decoder)
control_decoder.cuda()


split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

align_canonical = True
anisotropic = True
if_augment = True

dataset = DataSetControlPointsPoisson(
    config.dataset_path,
    config.batch_size,
    splits=split_dict,
    size_v=config.grid_size,
    size_u=config.grid_size)

get_train_data = dataset.load_train_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic, if_augment=if_augment
)

get_val_data = dataset.load_val_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic
)

loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

nurbs_layer = SurfEval(20,20, dimension=3, p=3, q=3, out_dim_u=40, out_dim_v=40, method='tc', dvc='cuda')
nurbs_layer.cuda()

optimizer = torch.optim.Adam(control_decoder.parameters(),lr=config.lr)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=3e-5
)



nu, nv = uniform_knot_bspline(20, 20, 3, 3, 40)
nu = torch.from_numpy(nu.astype(np.float32)).cuda()
nv = torch.from_numpy(nv.astype(np.float32)).cuda()

mse_loss = torch.nn.MSELoss()

prev_test_cd = 1e8
for e in range(config.epochs):
    train_reg = []
    train_str = []
    train_cd = []
    train_lap = []

    train_cd_pointclouds = []
    train_mse_pointclouds = []
    train_mse_ctrlpts = []

    train_control_decoder = []
    train_nurbs_layer = []
    train_net_loss = []

    control_decoder.train()
    for train_b_id in range(config.num_train // config.batch_size):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        points_, parameters, control_points, scales, _ = next(get_train_data)[0]
        control_points = Variable(
            torch.from_numpy(control_points.astype(np.float32))
        ).cuda()

        points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
        points = points.permute(0, 2, 1)

        # Sample random number of points to make network robust to density.
        rand_num_points = config.num_points + np.random.choice(np.arange(-300, 1300), 1)[0]

        output = control_decoder(points[:, :, 0:rand_num_points])
        
        weights = output[:,:,3]
        weights = weights.view(weights.size(0),weights.size(1),1)
        

        if anisotropic:
            # rescale all tensors to original dimensions for evaluation
            scales, output, points, control_points = rescale_input_outputs(scales, output[:,:,:-1], points, control_points,
                                                                           config.batch_size)

        # # Chamfer Distance loss, between predicted and GT surfaces
        # cd, reconstructed_points = spline_reconstruction_loss_one_sided(
        #     nu, nv, output, points, config
        # )

        output = torch.cat((output,weights),-1)
        # Chamfer Distance loss, between predicted and GT surfaces
        cd, reconstructed_points = nurbs_reconstruction_loss_one_sided(
            nu, nv, output, points, config
        )

        # Permutation Regression Loss
        # permute_cp has the best permutation of gt control points grid
        l_reg, permute_cp = control_points_permute_reg_loss(
            output[:,:,:-1], control_points, config.grid_size
        )

        laplac_loss = laplacian_loss(
            output[:,:,:-1].reshape((config.batch_size, config.grid_size, config.grid_size, 3)),
            permute_cp,
            dist_type="l2",
        )
                    
        control_decoder_loss = l_reg * config.loss_weight + (cd + laplac_loss) * (
                1 - config.loss_weight
        )


        # #NURBS layer
        # control_decoder_output = output.view(-1,20,20,3).cuda()
        # # rand = torch.rand((control_decoder_output.size(0),
        # #                     control_decoder_output.size(1),
        # #                     control_decoder_output.size(2),1), 
        # #                     requires_grad=True).cuda()
        # # control_decoder_output = torch.cat((control_decoder_output,rand),-1)
        # ones = torch.ones((control_decoder_output.size(0),
        #             control_decoder_output.size(1),
        #             control_decoder_output.size(2),1), 
        #             requires_grad=True).cuda()

        # control_decoder_output = torch.cat((control_decoder_output,ones),-1)
        

        # ground_truth_control_points = control_points.clone().detach() 

        # ground_truth_control_points = ground_truth_control_points.view(-1,20,20,3)

        # ground_truth_control_points = torch.cat((ground_truth_control_points,ones),-1)               
      
        # ground_truth_nurbs_layer = nurbs_layer(ground_truth_control_points)
        # ground_truth_nurbs_layer = ground_truth_nurbs_layer.view(-1,1600,3)

        # nurbs_layer_output = nurbs_layer(control_decoder_output)
        # nurbs_layer_output = nurbs_layer_output.view(-1,1600,3)

        # print(ground_truth_nurbs_layer)
        # print("NURBS on their ctrl_pts")
        # print(nurbs_layer_output)


        # #Compute NURBS layer losses

        # raw_input_points = Variable(points.permute(0,2,1).clone().detach()).cuda()
        # cd_pointclouds,_ = chamfer_distance(nurbs_layer_output, raw_input_points)
        # mse_pointclouds = mse_loss(nurbs_layer_output, ground_truth_nurbs_layer)
        # mse_ctrlpts = mse_loss(ground_truth_control_points[:,:,:,:-1],control_decoder_output[:,:,:,:-1])

        # nurbs_layer_loss = 0.1*cd_pointclouds + mse_pointclouds + mse_ctrlpts

        #Net loss
        # loss = control_decoder_loss + nurbs_layer_loss 
        loss = control_decoder_loss

        loss.backward()

        train_cd.append(cd.data.cpu().numpy())
        train_reg.append(l_reg.data.cpu().numpy())
        train_lap.append(laplac_loss.data.cpu().numpy())

        # train_cd_pointclouds.append(cd_pointclouds.data.cpu().numpy())
        # train_mse_pointclouds.append(mse_pointclouds.data.cpu().numpy())
        # train_mse_ctrlpts.append(mse_ctrlpts.data.cpu().numpy())

        train_control_decoder.append(control_decoder_loss.data.cpu().numpy())
        # train_nurbs_layer.append(nurbs_layer_loss.data.cpu().numpy())
        train_net_loss.append(loss.data.cpu().numpy())

        optimizer.step()

        log_value(
            "cd",
            cd.data.cpu().numpy(),
            train_b_id + e * (config.num_train // config.batch_size),
        )
        log_value(
            "l_reg",
            l_reg.data.cpu().numpy(),
            train_b_id + e * (config.num_train // config.batch_size),
        )
        log_value(
            "l_lap",
            laplac_loss.data.cpu().numpy(),
            train_b_id + e * (config.num_train // config.batch_size),
        )
        log_value(
            "control_decoder_loss",
            control_decoder_loss.data.cpu().numpy(),
            train_b_id + e * (config.num_train // config.batch_size),
        )

        # log_value(
        #     "cd_pointclouds",
        #     cd_pointclouds.data.cpu().numpy(),
        #     train_b_id + e * (config.num_train // config.batch_size),
        # )
        # log_value(
        #     "mse_pointclouds",
        #     mse_pointclouds.data.cpu().numpy(),
        #     train_b_id + e * (config.num_train // config.batch_size),
        # )
        # log_value(
        #     "mse_ctrlpts",
        #     mse_ctrlpts.data.cpu().numpy(),
        #     train_b_id + e * (config.num_train // config.batch_size),
        # )
        # log_value(
        #     "nurbs_layer_loss",
        #     nurbs_layer_loss.data.cpu().numpy(),
        #     train_b_id + e * (config.num_train // config.batch_size),
        # )
        
        log_value(
            "loss",
            loss.data.cpu().numpy(),
            train_b_id + e * (config.num_train // config.batch_size),
        )
        
        print(
            "\rEpoch: {} iter: {}, loss: {}".format(
                e, train_b_id, loss.item()
            ),
            end="",
        )


    distances = []
    test_reg = []
    test_cd = []
    test_str = []
    test_lap = []
    control_decoder.eval()

    for val_b_id in range(config.num_test // config.batch_size - 1):
        torch.cuda.empty_cache()
        points_, parameters, control_points, scales, _ = next(get_val_data)[0]

        control_points = Variable(
            torch.from_numpy(control_points.astype(np.float32))
        ).cuda()
        points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
        points = points.permute(0, 2, 1)
        with torch.no_grad():
            output = control_decoder(points[:, :, 0:config.num_points])
            weights = output[:,:,3]
            weights = weights.view(weights.size(0),weights.size(1),1)
        
            if anisotropic:
                scales, output, points, control_points = rescale_input_outputs(scales, output[:,:,:-1], points, control_points,
                                                                               config.batch_size)

                # Chamfer Distance loss, between predicted and GT surfaces
        
        
        output = torch.cat((output,weights),-1)
        
        cd, reconstructed_points = nurbs_reconstruction_loss_one_sided(
            nu, nv, output, points, config
        )

        l_reg, permute_cp = control_points_permute_reg_loss(
            output[:,:,:-1], control_points, config.grid_size
        )
        laplac_loss = laplacian_loss(
            output[:,:,:-1].reshape((config.batch_size, config.grid_size, config.grid_size, 3)),
            permute_cp,
            dist_type="l2",
        )

        loss = l_reg * config.loss_weight + (cd + laplac_loss) * (
                1 - config.loss_weight
        )
        test_reg.append(l_reg.data.cpu().numpy())
        test_cd.append(cd.data.cpu().numpy())
        test_lap.append(laplac_loss.data.cpu().numpy())

    print("\n")

    log_value("train_cd", np.mean(train_cd), e)
    log_value("test_cd", np.mean(test_cd), e)
    log_value("train_reg", np.mean(train_reg), e)
    log_value("test_reg", np.mean(test_reg), e)

    log_value("train_control_decoder", np.mean(train_control_decoder), e)
    # log_value("train_nurbs_layer", np.mean(train_nurbs_layer), e)
    log_value("train_net_loss", np.mean(train_net_loss), e)

    scheduler.step(np.mean(test_cd))
    if prev_test_cd > np.mean(test_cd):
        # logger.info("CD improvement, saving model at epoch: {}".format(e))
        prev_test_cd = np.mean(test_cd)
        torch.save(
            control_decoder.state_dict(),
            "logs/trained_models/{}.pth".format(model_name),
        )