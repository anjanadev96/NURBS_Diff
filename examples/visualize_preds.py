import numpy as np
from open3d import *    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_all():
    for i in range(100):
            print(i)
            index = i
            ctrlpts = read_point_cloud('./logs/results/open_spline_rational_cd_10/ctrlpts_'+str(index)+'.ply') # Read the point cloud
            ctrlpts = np.asarray(ctrlpts.points)

            weights = np.load(('./logs/results/open_spline_rational_cd_10/weights_'+str(index)+'.npy')) # Read the point cloud
            weights = np.asarray(weights)

            cloud = read_point_cloud('./logs/results/open_spline_rational_cd_10/gt_'+str(index)+'.ply') # Read the point cloud
            cloud_arr = np.asarray(cloud.points)

            pred = read_point_cloud('./logs/results/open_spline_rational_cd_10/pred_'+str(index)+'.ply') # Read the point cloud
            pred = np.asarray(pred.points)

            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
            target_mpl = cloud_arr.reshape(40,40,3)
            predicted = pred.reshape(40,40,3)
            weights = weights.reshape(20,20,1)


            predctrlpts = ctrlpts.reshape(20,20,3)
            # predctrlpts = ctrlpts

            predctrlpts = predctrlpts[:, :, :3] / weights
            # surf1 = ax.scatter(target_mpl[0, :, 0], target_mpl[0, :, 1], target_mpl[0, :, 2], s=1.0, color='black',
            #                    label='Target Offset surface')
            # surf1 = ax.scatter(target_mpl[:, 0], target_mpl[:, 1], target_mpl[:, 2], s=1.0, color='black',
            #                    label='Target Offset surface')
            
            surf2 = ax.plot_surface(predicted[:,:, 0], predicted[:,:, 1], predicted[:,:, 2], color='blue',
                                    label='Predicted Surface', alpha=0.4)
            surf1 = ax.plot_surface(target_mpl[:,:, 0], target_mpl[:,:, 1], target_mpl[:,:, 2], color='green',
                                    label='Target Surface', alpha=0.5)


            # surf3 = ax.scatter(predctrlpts[:, 0], predctrlpts[:, 1], predctrlpts[:, 2], s=1.0, color='black',
            #                 label='Target Offset surface')            
            surf3 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
                                    linestyle='dashed', color='orange', label='Predicted CP')
            # surf3 = ax.plot_surface(BaseSurf_pts[0, :, :, 0], BaseSurf_pts[0, :, :, 1], BaseSurf_pts[0, :, :, 2], color='blue')
            # surf2 = ax.plot_surface(predicted[1, :, :, 0], predicted[1, :, :, 1], predicted[1, :, :, 2], color='red',
            #                         label='Predicted Surface', alpha=0.7)
            # surf2 = ax.plot_wireframe(predctrlpts[1, :, :, 0], predctrlpts[1, :, :, 1], predctrlpts[1, :, :, 2],
            #                         linestyle='dashed', color='purple', label='Predicted CP')
            # surf3 = ax.plot_surface(BaseSurf_pts[1, :, :, 0], BaseSurf_pts[1, :, :, 1], BaseSurf_pts[1, :, :, 2], color='pink')
            # surf4 = ax.plot_surface(CtrlPtsOffsetSurf_Pts[0, :, :, 0], CtrlPtsOffsetSurf_Pts[0, :, :, 1],
            #                         CtrlPtsOffsetSurf_Pts[0, :, :, 2], color='yellow')
            # surf4 = ax.plot_surface(CtrlPtsOffsetSurf_Pts[1, :, :, 0], CtrlPtsOffsetSurf_Pts[1, :, :, 1],
            #                         CtrlPtsOffsetSurf_Pts[1, :, :, 2], color='yellow')
            ax.set_zlim(-0.5,0.5)
            ax.set_xlim(-0.5,0.5)
            ax.set_ylim(-0.5,0.5)
            ax.azim = 44
            ax.dist = 6.5
            ax.elev = 64
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax._axis3don = False
            # ax.legend()

            fig.subplots_adjust(hspace=0, wspace=0)
            fig.tight_layout()
            plt.show()

if __name__ == "__main__":
    # visualize_all()


    fig = plt.figure(figsize=(25,8))
    ax = fig.add_subplot(151, projection='3d', adjustable='box', proj_type='ortho')
    index = 35
    ctrlpts = read_point_cloud('./logs/results/open_spline_rational_cd_10/ctrlpts_'+str(index)+'.ply') # Read the point cloud
    ctrlpts = np.asarray(ctrlpts.points)

    cloud = read_point_cloud('./logs/results/open_spline_rational_cd_10/gt_'+str(index)+'.ply') # Read the point cloud
    cloud_arr = np.asarray(cloud.points)

    pred = read_point_cloud('./logs/results/open_spline_rational_cd_10/pred_'+str(index)+'.ply') # Read the point cloud
    pred = np.asarray(pred.points)

    target_mpl = cloud_arr.reshape(40,40,3)
    predicted = pred.reshape(40,40,3)
    predctrlpts = ctrlpts.reshape(20,20,3)
    surf2 = ax.plot_surface(predicted[:,:, 0], predicted[:,:, 1], predicted[:,:, 2], color='blue',
                            label='Predicted Surface', alpha=0.9)
    surf1 = ax.scatter(target_mpl[:,:, 0], target_mpl[:,:, 1], target_mpl[:,:, 2], color='green',
                            label='Target Surface', alpha=0.9, s=0.7)        
    # surf3 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
    #                         linestyle='dashed', color='orange', label='Predicted CP')
    ax.set_zlim(-0.4,0.4)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.azim = 44
    ax.dist = 6.5
    ax.elev = 64
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    ax = fig.add_subplot(152, projection='3d', adjustable='box', proj_type='ortho')
    index = 20
    ctrlpts = read_point_cloud('./logs/results/open_spline_rational_cd_10/ctrlpts_'+str(index)+'.ply') # Read the point cloud
    ctrlpts = np.asarray(ctrlpts.points)

    cloud = read_point_cloud('./logs/results/open_spline_rational_cd_10/gt_'+str(index)+'.ply') # Read the point cloud
    cloud_arr = np.asarray(cloud.points)

    pred = read_point_cloud('./logs/results/open_spline_rational_cd_10/pred_'+str(index)+'.ply') # Read the point cloud
    pred = np.asarray(pred.points)
    target_mpl = cloud_arr.reshape(40,40,3)
    predicted = pred.reshape(40,40,3)
    predctrlpts = ctrlpts.reshape(20,20,3)
    surf1 = ax.scatter(target_mpl[:,:, 0], target_mpl[:,:, 1], target_mpl[:,:, 2], color='green',
                            label='Input Point cloud', alpha=0.9, s=0.7) 
    surf2 = ax.plot_surface(predicted[:,:, 0], predicted[:,:, 1], predicted[:,:, 2], color='blue',
                            label='Predicted Surface', alpha=0.9)
       
    # surf3 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
    #                         linestyle='dashed', color='orange', label='Predicted Control Points')
    ax.set_zlim(-0.4,0.4)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.azim = 44
    ax.dist = 6.5
    ax.elev = 64
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
  

    ax = fig.add_subplot(153, projection='3d', adjustable='box', proj_type='ortho')
    index = 5
    ctrlpts = read_point_cloud('./logs/results/open_spline_rational_cd_10/ctrlpts_'+str(index)+'.ply') # Read the point cloud
    ctrlpts = np.asarray(ctrlpts.points)

    cloud = read_point_cloud('./logs/results/open_spline_rational_cd_10/gt_'+str(index)+'.ply') # Read the point cloud
    cloud_arr = np.asarray(cloud.points)

    pred = read_point_cloud('./logs/results/open_spline_rational_cd_10/pred_'+str(index)+'.ply') # Read the point cloud
    pred = np.asarray(pred.points)
    target_mpl = cloud_arr.reshape(40,40,3)
    predicted = pred.reshape(40,40,3)
    predctrlpts = ctrlpts.reshape(20,20,3)
    
    surf1 = ax.scatter(target_mpl[:,:, 0], target_mpl[:,:, 1], target_mpl[:,:, 2], color='green',
                            label='Input Point Cloud', alpha=0.9, s=0.7)      
    surf2 = ax.plot_surface(predicted[:,:, 0], predicted[:,:, 1], predicted[:,:, 2], color='blue',
                            label='Predicted Surface', alpha=0.9)  
    # surf3 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
    #                         linestyle='dashed', color='orange', label='Predicted CP')
    ax.set_zlim(-0.4,0.4)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.azim = 44
    ax.dist = 6.5
    ax.elev = 64
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    surf1._facecolors2d=surf2._facecolor3d
    surf1._edgecolors2d=surf2._edgecolor3d
    surf2._facecolors2d=surf2._facecolor3d
    surf2._edgecolors2d=surf2._edgecolor3d
    # surf3._facecolors2d=surf2._facecolor3d
    # surf3._edgecolors2d=surf2._edgecolor3d
    plt.rc('legend', fontsize=12)  
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=5)



    ax = fig.add_subplot(154, projection='3d', adjustable='box', proj_type='ortho')
    index = 6
    ctrlpts = read_point_cloud('./logs/results/open_spline_rational_cd_10/ctrlpts_'+str(index)+'.ply') # Read the point cloud
    ctrlpts = np.asarray(ctrlpts.points)

    cloud = read_point_cloud('./logs/results/open_spline_rational_cd_10/gt_'+str(index)+'.ply') # Read the point cloud
    cloud_arr = np.asarray(cloud.points)

    pred = read_point_cloud('./logs/results/open_spline_rational_cd_10/pred_'+str(index)+'.ply') # Read the point cloud
    pred = np.asarray(pred.points)
    target_mpl = cloud_arr.reshape(40,40,3)
    predicted = pred.reshape(40,40,3)
    predctrlpts = ctrlpts.reshape(20,20,3)
    surf2 = ax.plot_surface(predicted[:,:, 0], predicted[:,:, 1], predicted[:,:, 2], color='blue',
                            label='Predicted Surface', alpha=0.9)
    surf1 = ax.scatter(target_mpl[:,:, 0], target_mpl[:,:, 1], target_mpl[:,:, 2], color='green',
                            label='Target Surface', alpha=0.9, s=0.7)        
    # surf3 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
    #                         linestyle='dashed', color='orange', label='Predicted CP')
    ax.set_zlim(-0.4,0.4)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.azim = 44
    ax.dist = 6.5
    ax.elev = 64
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False


    ax = fig.add_subplot(155, projection='3d', adjustable='box', proj_type='ortho')
    index = 34
    ctrlpts = read_point_cloud('./logs/results/open_spline_rational_cd_10/ctrlpts_'+str(index)+'.ply') # Read the point cloud
    ctrlpts = np.asarray(ctrlpts.points)

    cloud = read_point_cloud('./logs/results/open_spline_rational_cd_10/gt_'+str(index)+'.ply') # Read the point cloud
    cloud_arr = np.asarray(cloud.points)

    pred = read_point_cloud('./logs/results/open_spline_rational_cd_10/pred_'+str(index)+'.ply') # Read the point cloud
    pred = np.asarray(pred.points)
    target_mpl = cloud_arr.reshape(40,40,3)
    predicted = pred.reshape(40,40,3)
    predctrlpts = ctrlpts.reshape(20,20,3)
    surf2 = ax.plot_surface(predicted[:,:, 0], predicted[:,:, 1], predicted[:,:, 2], color='blue',
                            label='Predicted Surface', alpha=0.9)
    surf1 = ax.scatter(target_mpl[:,:, 0], target_mpl[:,:, 1], target_mpl[:,:, 2], color='green',
                            label='Target Surface', alpha=0.9, s=0.7)        
    # surf3 = ax.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
    #                         linestyle='dashed', color='orange', label='Predicted CP')
    ax.set_zlim(-0.4,0.4)
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.4,0.4)
    ax.azim = 44
    ax.dist = 6.5
    ax.elev = 64
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    


    fig.subplots_adjust(hspace=-0.01, wspace=0)
    # fig.tight_layout()
    plt.show()