import numpy as np
import gtsam
import matplotlib.pyplot as plt
from visualization import visualization
from utils_gtsam_v5_viz import gtsam_graph
from gtsam.symbol_shorthand import T, U
import imageio

########################################################

#data_dir = '/home/devicereal/projects/tactile_FG/data/210708_case_1/rectangle/0/'
#data_dir = '/home/devicereal/projects/tactile_FG/data/devel_210801/rectangle/7/2/'
#data_dir = '/home/devicereal/projects/tactile_FG/data/devel_wall_210806_no_tactile/rectangle/29/'
data_dir = '/home/devicereal/projects/tactile_FG/data/demo_210811/rectangle/3/0/'

cart_init = np.load(data_dir+'cart_init.npy')
cart_seq = np.load(data_dir+'cart_g1_rock.npy')[1:].squeeze()
tact_seq = np.load(data_dir+'tact_g1_rock.npy')[1:].squeeze()
time_seq = np.load(data_dir+'time_g1_rock.npy')
pose_error = np.load(data_dir+'misalign.npy')
rand_pose = np.load(data_dir+'rand_pose.npy')
timeflag = np.load(data_dir+'timeflag.npy', allow_pickle=True)

height = rand_pose[2] + 51.5 - 10
reach = 35

t_stick = np.where(timeflag[1]!=None, timeflag[1], np.where(timeflag[5]!=None, timeflag[5], np.inf))
stick_idx = np.argmin(abs(time_seq - t_stick))

########################################################

viz = visualization(reach, pose_error, height, view_elev=45, view_azim=215, view_center=(0,0,-52), view_radius=40, env_type='hole')
viz.set_show(env=True, grp=True, obj_gt=True, obj_est=False, cline=True)

gtsam_graph = gtsam_graph(gt_height=height, reach=reach)
gtsam_graph.restart(cart_init)

gtsam_on = False

err_log_contact = []
err_log_fixed = []
err_total = []
D_opt = []

for i in range(len(cart_seq)):
    
    if gtsam_on == False and tact_seq[i,2] < -0.03:
        gtsam_on = True
        viz.stage = 'touch!'
    if timeflag[2] != None:
        if viz.stage=='touch!' and i > np.argmin(abs(time_seq - timeflag[2])):
            viz.stage = 'tilting for rocking'
        elif viz.stage=='tilting for rocking' and i > np.argmin(abs(time_seq - timeflag[3])):
            viz.stage = 'rocking!'
        elif viz.stage=='rocking!' and i > np.argmin(abs(time_seq - timeflag[4])):
            viz.stage = 'finishing rocking'
    if timeflag[5] != None:
        if i >  np.argmin(abs(time_seq - timeflag[5])):
            viz.stage = 'wiggle wiggle'

    gtsam_graph.add_new_gt(cart_seq[i], tact_seq[i])
    if gtsam_on:
        gtsam_graph.add_new(cart_seq[i], tact_seq[i])##np.zeros(6))#
    if i == stick_idx:
        #gtsam_graph.discard_n_reset()
        gtsam_graph.stick_on = True

    viz.plot_clear()
    
    #err_log.append(gtsam_graph.isam.getFactorsUnsafe().error(gtsam_graph.isam.calculateEstimate()))
    err_total.append(gtsam_graph.isam.getFactorsUnsafe().error(gtsam_graph.isam.calculateEstimate())/(gtsam_graph.i-gtsam_graph.idx_window_begin+1))
    contact_total_err = 0
    fixed_total_err = 0
    N_contact = 1e-5
    N_fixed = 1e-5
    for gtsam_i in range(gtsam_graph.i):
        if T(gtsam_i) in list(gtsam_graph.factor_dict.keys()):
            factor = gtsam_graph.isam.getFactorsUnsafe().at(gtsam_graph.factor_dict[T(gtsam_i)])
            if factor != None:
                contact_total_err += factor.error(gtsam_graph.isam.calculateEstimate())
                N_contact += 1
        if U(gtsam_i) in list(gtsam_graph.factor_dict.keys()):
            factor = gtsam_graph.isam.getFactorsUnsafe().at(gtsam_graph.factor_dict[U(gtsam_i)])
            if factor != None:
                fixed_total_err += factor.error(gtsam_graph.isam.calculateEstimate())
                N_fixed += 1
    err_log_contact.append(contact_total_err/N_contact)
    err_log_fixed.append(fixed_total_err/N_fixed)
    D_opt.append(gtsam_graph.D_opt)
    """
    if T(gtsam_graph.i) in list(gtsam_graph.factor_dict.keys()):
        err_log.append(gtsam_graph.isam.getFactorsUnsafe().at(gtsam_graph.factor_dict[T(gtsam_graph.i)]).error(gtsam_graph.isam.calculateEstimate()))
    else:
        err_log.append(0)
    """
    
    if i % 4 == 0:
        
        for _ in range(0):
            dd = 0.4 * np.random.multivariate_normal(np.zeros(6), gtsam_graph.ctl_cov)
            ctl_ = gtsam_graph.ctl.compose(gtsam.Pose3(gtsam.Rot3.RzRyRx(dd[:3]), dd[3:]))
            viz.plot_cline(ctl_.translation(), ctl_.rotation().matrix(), lw=4, c='k', alpha=.1)
            
        viz.plot_update(gtsam_graph.grp_gt.translation(), gtsam_graph.grp_gt.rotation().matrix(),
                   gtsam_graph.obj.translation(), gtsam_graph.obj.rotation().matrix(),
                   gtsam_graph.ctl.translation(), gtsam_graph.ctl.rotation().matrix(),
                   gtsam_graph.obj_gt.translation(), gtsam_graph.obj_gt.rotation().matrix(), alpha=.1)
    
imageio.mimsave('./plots/video.gif',viz.images[1:],fps=10)