import numpy as np
import math
import gtsam
from gtsam_packing import TactileTransformFactor_3D, StickingContactFactor_3D, ClineCenterFactor
import gtsam.utils.plot as gtsam_plot
#from plot_tools import error_ellipse
#import test_example
import matplotlib.pyplot as plt
from gtsam.utils import plot
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Rectangle
from gtsam.symbol_shorthand import G, O, C  # G: Gripper, O: Object, C: Contact
from gtsam.symbol_shorthand import P, Q, X, S, T, U, V, W
from collections import deque
# P: Gripper Prior
# Q: Object Prior
# X: Contact Prior
# S: Gelslim Deform
# T: Contact on Plane
# U: Fixed Contact
# V: Sticking Contact
# W: Center
########################################################


def kl_mvn(m0, S0, m1, S1):
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff
    return .5 * (tr_term + det_term + quad_term - N)


class gtsam_graph:
    def __init__(self, reach=54.5, window=200, buffer_len=10, env_type='hole'):

        self.reach = reach
        self.window = window
        self.buffer_len = buffer_len
        self.env_type = env_type
        self.window_decay = float(window) / (window + 1)
        self.r_convert = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.cart_init = None
        self.cart = None
        self.stick_on = False

        self.GRIPPER_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1]))
        self.OBJECT_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, 1e-3, 1e-1, 2e1, 1e-1]))
        self.CLINE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e2, 1e2, 1e0, 5e1, 2e1, 5e1]))
        self.CLINE_PLANE_PARL = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-3, np.inf, 1e-3, np.inf, 1e-1, np.inf])))
        self.CLINE_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.inf, 1e-3, 1e-3, np.inf, 1e-1, 1e-1]))
        self.TACTILE_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1])))
        self.STICK_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 5e-1])))
        self.CLINE_CENTER = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]))

        self.reset_graph()

    def set_current_pose_as_origin(self):
        self.O_0 = self.obj
        self.G_0 = self.grp

    def reset_graph(self,
                    height_est=None,
                    height_cov=None,
                    O_0=None,
                    C_0=None,
                    nominal_thres=[1.1, 1.2],
                    terminal_thres=[1e-2, 50, 40],
                    max_deform_limit=[np.inf, 0.4, 0.4, np.inf, 0.35, 1.2],
                    ctl_rollback_length=5):

        self.nominal_thres = nominal_thres # if the tilting angle exceeds this value, set the estimation as the nominal estimation
        self.terminal_thres = terminal_thres # if the estimation differs too much from the nominal estimate, raise the mode detection and terminate
        self.max_deform_limit = max_deform_limit # if the gripper-object relative pose is too big, terminate
        self.ctl_rollback_length = ctl_rollback_length
        self.got_nominal = False
        self.terminate = False
        self.mode_detect_on = False
        self.mode_detected = False
        self.error_raise = False

        self.gt_buffer = deque(maxlen=self.buffer_len)
        self.tact_buffer = deque(maxlen=self.buffer_len)
        self.ctl_buffer = deque(maxlen=self.ctl_rollback_length)
        self.ctl_cov_buffer = deque(maxlen=self.ctl_rollback_length)
        self.obj_tilt_buffer = deque(maxlen=1000)
        self.grp_tilt_buffer = deque(maxlen=1000)
        self.gt_buffer.append(np.zeros(6))
        self.tact_buffer.append(np.zeros(6))
        self.ctl_nominal = None  #np.zeros(6)
        self.ctl_cov_nominal = np.zeros((6, 6))

        self.tilt_max_vec = np.zeros(3)

        # coordinate system is different from the paper (x, y, z order is different)
        if O_0 is None and height_est is None:
            O_0 = gtsam.Pose3(gtsam.Rot3.Roll(-0.5 * np.pi),
                              np.array([0, 0, -self.reach]))
        elif height_est:
            O_0 = gtsam.Pose3(gtsam.Rot3.Roll(-0.5 * np.pi),
                              np.array([0, 0, -height_est]))
        if C_0 is None and height_est is None:
            C_0 = gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.5 * np.pi, 0, 0.5 * np.pi),
                              np.array([0, 0, -self.reach]))
        elif height_est:
            C_0 = gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.5 * np.pi, 0, 0.5 * np.pi),
                              np.array([0, 0, -height_est]))

        self.O_0 = O_0

        if self.cart is None:
            G_0 = gtsam.Pose3()
        else:
            xyz_world = self.cart[:3] - self.cart_init[:3]
            r_g = R.from_quat(self.cart[3:]) * self.r_convert
            xyz = self.r_g_init.inv().as_matrix().dot(xyz_world)
            ypr = (self.r_g_init.inv() * r_g).as_euler('zyx')
            gt = np.hstack((xyz, ypr))
            g_rot = R.from_euler('zyx', gt[3:]).as_matrix()
            g_trn = gt[:3]
            G_0 = gtsam.Pose3(gtsam.Rot3(g_rot), g_trn)

        self.G_0 = G_0

        self.factor_num = 0
        self.factor_dict = {}
        self.i = 0
        self.i_ema = 0
        self.idx_window_begin = 0

        self.i_nominal = 0

        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)

        self.isam = gtsam.ISAM2(parameters)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        self.graph.push_back(
            gtsam.PriorFactorPose3(G(0), G_0, self.GRIPPER_PRIOR_NOISE))
        self.factor_dict[P(0)] = self.factor_num
        self.factor_num += 1

        if height_est:
            OBJECT_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-3, 1e-3, 1e-3, 1e-1, height_cov**0.5, 1e-1]))
            self.graph.push_back(
                gtsam.PriorFactorPose3(O(0), O_0, OBJECT_PRIOR_NOISE))
        else:
            self.graph.push_back(
                gtsam.PriorFactorPose3(O(0), O_0, self.OBJECT_PRIOR_NOISE))
        self.factor_dict[Q(0)] = self.factor_num
        self.factor_num += 1

        if height_est:
            CLINE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e2, 1e2, 1e0, 5e1, height_cov**0.5, 5e1]))
            self.graph.push_back(
                gtsam.PriorFactorPose3(C(0), C_0, CLINE_PRIOR_NOISE))
        else:
            self.graph.push_back(
                gtsam.PriorFactorPose3(C(0), C_0, self.CLINE_PRIOR_NOISE))
        self.factor_dict[X(0)] = self.factor_num
        self.factor_num += 1

        self.graph.push_back(ClineCenterFactor(C(0), self.CLINE_CENTER))
        self.factor_dict[W(0)] = self.factor_num
        self.factor_num += 1

        self.initial_estimate.insert(G(0), G_0)
        self.initial_estimate.insert(O(0), O_0)
        self.initial_estimate.insert(C(0), C_0)
        self.isam.update(self.graph, self.initial_estimate)
        for _ in range(2):
            self.isam.update()
        self.graph.resize(0)
        self.initial_estimate.clear()

        self.current_estimate = self.isam.calculateEstimate()

        self.grp = self.current_estimate.atPose3(G(self.i))
        self.ctl = self.current_estimate.atPose3(C(self.i))
        self.obj = self.current_estimate.atPose3(O(self.i))
        self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(),
                                         self.isam.calculateEstimate())
        self.ctl_cov = self.marginals.marginalCovariance(C(self.i))
        self.obj_cov = self.marginals.marginalCovariance(O(self.i))

        self.ctl_buffer.append(self.ctl)
        self.ctl_cov_buffer.append(self.ctl_cov)

        self.D_opt = np.linalg.det(self.ctl_cov)**(1 / 6)

    def restart(self, cart_init, height_est=None, height_cov=None):

        self.cart = None
        self.cart_init = cart_init
        self.r_g_init = R.from_quat(self.cart_init[3:]) * self.r_convert
        self.tactile = np.zeros(6)
        self.gt = np.zeros(6)

        self.reset_graph(height_est=height_est, height_cov=height_cov)

    def check_nominal_after_rock(self):
        if np.max(self.obj_tilt_buffer) < 0.8 or self.got_nominal:
            return
        else:
            max_val = np.max(self.obj_tilt_buffer)
            obj_tilt_history = np.array(self.obj_tilt_buffer)
            i_min_1 = np.min(np.where(obj_tilt_history > 0.8 * max_val)[0])
            i_max_1 = np.max(np.where(obj_tilt_history > 0.8 * max_val)[0])
            i_min_2 = np.min(
                (np.where(obj_tilt_history > 0.8 * max_val)[0] +
                 len(obj_tilt_history) / 2) % len(obj_tilt_history))
            i_max_2 = np.max(
                (np.where(obj_tilt_history > 0.8 * max_val)[0] +
                 len(obj_tilt_history) / 2) % len(obj_tilt_history))
            l = min(i_max_1 - i_min_1, i_max_2 - i_min_2)
            if l < 40:
                self.i_nominal_ = self.i
                self.ctl_nominal_ = self.ctl
                self.ctl_cov_nominal_ = self.ctl_cov
                self.i_nominal = self.i
                self.ctl_nominal = self.ctl
                self.ctl_cov_nomninal = self.ctl_cov
                self.got_nominal = True
                if self.env_type == 'hole':
                    self.mode_detect_on = True

    def add_new(self, cart_new, tactile_new):

        if not self.terminate:

            self.i += 1

            self.cart = np.array(cart_new)

            xyz_world = self.cart[:3] - self.cart_init[:3]
            self.r_g = R.from_quat(self.cart[3:]) * self.r_convert

            xyz = self.r_g_init.inv().as_matrix().dot(xyz_world)
            ypr = (self.r_g_init.inv() * self.r_g).as_euler('zyx')

            self.gt = np.hstack((xyz, ypr))
            g_rot = R.from_euler('zyx', self.gt[3:]).as_matrix()
            g_trn = self.gt[:3]

            self.graph.push_back(
                gtsam.PriorFactorPose3(G(self.i),
                                       gtsam.Pose3(gtsam.Rot3(g_rot), g_trn),
                                       self.GRIPPER_PRIOR_NOISE))
            self.factor_dict[P(self.i)] = self.factor_num
            self.factor_num += 1

            self.graph.push_back(
                ClineCenterFactor(C(self.i), self.CLINE_CENTER))
            self.factor_dict[W(self.i)] = self.factor_num
            self.factor_num += 1

            self.graph.push_back(
                gtsam.BetweenFactorPose3(C(self.i - 1), C(self.i),
                                         gtsam.Pose3(),
                                         self.CLINE_ODOMETRY_NOISE))
            self.factor_dict[U(self.i)] = self.factor_num
            self.factor_num += 1

            CLINE_PLANE_PARL = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.clip(10 * 0.995**self.i, 1, 10) *
                    np.array([1e-3, np.inf, 1e-3, np.inf, 1e-1, np.inf])))
            self.graph.push_back(
                gtsam.BetweenFactorPose3(O(self.i), C(self.i), gtsam.Pose3(),
                                         CLINE_PLANE_PARL))
            self.factor_dict[T(self.i)] = self.factor_num
            self.factor_num += 1

            if self.stick_on:
                self.graph.push_back(
                    StickingContactFactor_3D(O(self.i - 1), O(self.i),
                                             C(self.i - 1), C(self.i),
                                             self.STICK_NOISE))
                self.factor_dict[V(self.i)] = self.factor_num
                self.factor_num += 1

            TM_ = self.tactile.copy()
            TM_[3:] *= np.pi / 180
            TM_rot_ = R.from_euler('zyx', TM_[3:]).as_matrix()
            TM_trn_ = TM_[:3].copy()
            TM = tactile_new.copy()
            TM[3:] *= np.pi / 180
            TM_rot = R.from_euler('zyx', TM[3:]).as_matrix()
            TM_trn = TM[:3].copy()
            self.tactile = tactile_new.copy()

            self.gt_buffer.append(self.gt)
            self.tact_buffer.append(self.tactile)
            d_gt = self.gt - self.gt_buffer[0]
            d_gt[3:] *= 180 / np.pi
            d_tact = self.tactile - self.tact_buffer[0]

            self.graph.push_back(
                TactileTransformFactor_3D(
                    O(self.i - 1), O(self.i), G(self.i - 1), G(self.i),
                    gtsam.Pose3.between(
                        gtsam.Pose3(gtsam.Rot3(TM_rot_), TM_trn_),
                        gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn)),
                    self.TACTILE_NOISE))
            self.factor_dict[S(self.i)] = self.factor_num
            self.factor_num += 1

            self.i_ema *= self.window_decay
            self.i_ema += (1 - self.window_decay) * self.i

            remove_idx = []

            if self.i_ema - 1 > self.idx_window_begin:
                remove_idx.append(self.factor_dict[P(self.idx_window_begin)])
                remove_idx.append(self.factor_dict[Q(self.idx_window_begin)])
                remove_idx.append(self.factor_dict[X(self.idx_window_begin)])
                if T(self.idx_window_begin) in self.factor_dict.keys():
                    remove_idx.append(self.factor_dict[T(
                        self.idx_window_begin)])
                remove_idx.append(self.factor_dict[W(self.idx_window_begin)])
                remove_idx.append(self.factor_dict[S(self.idx_window_begin +
                                                     1)])
                remove_idx.append(self.factor_dict[U(self.idx_window_begin +
                                                     1)])
                if V(self.idx_window_begin + 1) in self.factor_dict.keys():
                    remove_idx.append(
                        self.factor_dict[V(self.idx_window_begin + 1)])

                obj_cov_ = 2 * self.marginals.marginalCovariance(
                    O(self.idx_window_begin + 1))
                obj_std_ = np.diag(obj_cov_)**0.5
                obj_std_min = np.array([1e-3, 1e-3, 1e-3, 1e-1, 2e1, 1e-1])
                obj_std_max = np.array([1e-3, 1e-3, 1e-3, 1e-1, 2e1, 1e-1])
                OBJECT_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
                    np.clip(obj_std_, obj_std_min, obj_std_max))
                
                ctl_cov_ = 2 * self.marginals.marginalCovariance(
                    C(self.idx_window_begin + 1))
                ctl_std_ = np.diag(ctl_cov_)**0.5
                ctl_std_min = np.array([5e-1, 5e-1, 2.5e-1, 2e1, 2e1, 2e1])
                ctl_std_max = np.array([1e2, 1e2, 1e0, 5e1, 2e1, 5e1])
                CLINE_PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
                    np.clip(ctl_std_, ctl_std_min, ctl_std_max))
                
                self.graph.push_back(
                    gtsam.PriorFactorPose3(
                        O(self.idx_window_begin + 1),
                        self.current_estimate.atPose3(
                            O(self.idx_window_begin + 1)), OBJECT_PRIOR_NOISE))

                self.factor_dict[Q(self.idx_window_begin +
                                   1)] = self.factor_num
                self.factor_num += 1

                self.graph.push_back(
                    gtsam.PriorFactorPose3(
                        C(self.idx_window_begin + 1),
                        self.current_estimate.atPose3(
                            C(self.idx_window_begin + 1)), CLINE_PRIOR_NOISE))

                self.factor_dict[X(self.idx_window_begin +
                                   1)] = self.factor_num
                self.factor_num += 1

                self.idx_window_begin += 1

            self.initial_estimate.insert(G(self.i),
                                         gtsam.Pose3(gtsam.Rot3(g_rot), g_trn))
            self.initial_estimate.insert(O(self.i), self.obj)
            self.initial_estimate.insert(C(self.i), self.ctl)
            try:
                self.isam.update(self.graph, self.initial_estimate,
                                 gtsam.KeyVector(remove_idx))
                for _ in range(2):
                    self.isam.update()
            except RuntimeError:
                print("Runtime Error Occurred, Restarting ISAM Graph")
                self.error_raise = True
                self.terminate = True

            self.graph.resize(0)
            self.initial_estimate.clear()

            self.current_estimate = self.isam.calculateEstimate()

            self.grp = self.current_estimate.atPose3(G(self.i))
            self.ctl = self.current_estimate.atPose3(C(self.i))
            self.obj = self.current_estimate.atPose3(O(self.i))
            self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(),
                                             self.isam.calculateEstimate())
            self.ctl_cov = self.marginals.marginalCovariance(C(self.i))
            self.obj_cov = self.marginals.marginalCovariance(O(self.i))

            self.ctl_buffer.append(self.ctl)
            self.ctl_cov_buffer.append(self.ctl_cov)

            self.D_opt = np.linalg.det(self.ctl_cov)**(1 / 6)

            self.obj_tilt = np.linalg.norm(
                R.from_matrix((self.O_0.rotation().between(
                    self.obj.rotation())).matrix()).as_rotvec()) / np.pi * 180
            self.grp_tilt = np.linalg.norm(
                R.from_matrix((self.G_0.rotation().between(
                    self.grp.rotation())).matrix()).as_rotvec()) / np.pi * 180

            self.obj_tilt_buffer.append(self.obj_tilt)
            self.grp_tilt_buffer.append(self.grp_tilt)

            tilt_vec = R.from_matrix((self.O_0.rotation().between(
                self.obj.rotation())).matrix()).as_rotvec()
            tilt_vec = self.O_0.rotation().matrix().dot(tilt_vec)
            if np.linalg.norm(self.tilt_max_vec) < np.linalg.norm(tilt_vec):
                self.tilt_max_vec = tilt_vec

            if self.obj_tilt > self.nominal_thres[
                    0] and self.grp_tilt > self.nominal_thres[
                        1] and not self.got_nominal:

                self.i_nominal_ = self.i
                self.ctl_nominal_ = self.ctl
                self.ctl_cov_nominal_ = self.ctl_cov
                self.i_nominal = self.i
                self.ctl_nominal = self.ctl
                self.ctl_cov_nomninal = self.ctl_cov
                self.got_nominal = True
                if self.env_type == 'hole':
                    self.mode_detect_on = True

            if self.got_nominal:
                ctl_between = self.ctl_nominal_.between(self.ctl)
                if np.abs(ctl_between.rotation().ypr()[1] / np.pi * 180) < 5.:
                    self.i_nominal = self.i
                    self.ctl_nominal = self.ctl
                    self.ctl_cov_nominal = self.ctl_cov
            
            if np.mean([
                    ctl_cov[1, 1] for ctl_cov in list(self.ctl_cov_buffer)[-5:]
            ]) < self.terminal_thres[0] and np.mean([
                    ctl_cov[4, 4] for ctl_cov in list(self.ctl_cov_buffer)[-5:]
            ]) < self.terminal_thres[1] and np.mean([
                    ctl_cov[5, 5] for ctl_cov in list(self.ctl_cov_buffer)[-5:]
            ]) < self.terminal_thres[2]:
                self.terminate = True

            if self.stick_on and np.any(
                    np.abs(self.tactile) > self.max_deform_limit):
                print('too much deformation! stopping wiggling')
                print(self.tactile)
                self.terminate = True
                if self.env_type == 'hole':
                    self.mode_detected = True

            if self.mode_detect_on and self.got_nominal:

                if np.all(
                        np.abs(
                            np.array([
                                self.ctl_nominal_.between(
                                    ctl).rotation().ypr()[1] / np.pi * 180
                                for ctl in self.ctl_buffer
                            ])) > 30.):
                    self.terminate = True
                    self.mode_detected = True
                    
                    self.i = self.i_nominal
                    self.ctl = self.ctl_nominal
                    self.ctl_cov = self.ctl_cov_nominal
