import numpy as np
import gtsam
from gtsam_packing import TactileTransformFactor_3D, StickingContactFactor_3D, ClineCenterFactor
from scipy.spatial.transform import Rotation as R
from collections import deque
from gtsam.symbol_shorthand import G, O, C  # G: Gripper, O: Object, C: Contact
from gtsam.symbol_shorthand import P, Q, X, S, T, U, V, W
# P: Gripper Prior
# Q: Object Prior
# X: Contact Prior
# S: Gelslim Deform
# T: Contact on Plane
# U: Fixed Contact
# V: Sticking Contact
# W: Center
########################################################


class gtsam_graph:
    def __init__(self, gt_height, reach=54.5, window=200, buffer_len=10, env_type='hole'):

        self.height = gt_height
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
                np.array([1e-2, np.inf, 1e-2, np.inf, 4e-1, np.inf])))
        self.CLINE_ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.inf, 1e-3, 1e-3, np.inf, 1e-1, 1e-1]))
        self.TACTILE_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1])))
        self.STICK_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=1.345),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1e-1])))
        self.CLINE_CENTER = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]))

        self.OBJECT_PRIOR_NOISE_GT = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1]))

        self.reset_graph()

    def reset_graph(self,
                    height_est=None,
                    height_cov=None,
                    O_0=None,
                    C_0=None,
                    nominal_thres=[1.1, 1.2],
                    terminal_thres=[1e-2, 50, 40],
                    dotprod_thres=[-0.3, -0.6],
                    dotprod_count=8,
                    ctl_rollback_length=5):

        self.nominal_thres = nominal_thres
        self.terminal_thres = terminal_thres
        self.dotprod_thres = dotprod_thres
        self.dotprod_count = dotprod_count
        self.ctl_rollback_length = ctl_rollback_length
        self.got_nominal = False
        self.terminate = False
        self.mode_detect_on = False
        self.mode_detected = False
        self.error_raise = False

        self.dotprod_buffer = deque(maxlen=self.dotprod_count)
        self.gt_buffer = deque(maxlen=self.buffer_len)
        self.tact_buffer = deque(maxlen=self.buffer_len)
        self.ctl_buffer = deque(maxlen=self.ctl_rollback_length)
        self.ctl_cov_buffer = deque(maxlen=self.ctl_rollback_length)
        self.obj_tilt_buffer = deque(maxlen=1000)
        self.grp_tilt_buffer = deque(maxlen=1000)
        self.gt_buffer.append(np.zeros(6))
        self.tact_buffer.append(np.zeros(6))
        self.ctl_nominal = None
        self.ctl_cov_nominal = np.zeros((6, 6))

        self.tilt_max_vec = np.zeros(3)

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

        self.tactile_gt = np.zeros(6)
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)
        self.i_gt = 0
        self.isam_gt = gtsam.ISAM2(parameters)
        self.graph_gt = gtsam.NonlinearFactorGraph()
        self.initial_estimate_gt = gtsam.Values()
        self.graph_gt.push_back(gtsam.PriorFactorPose3(G(0), gtsam.Pose3(), self.GRIPPER_PRIOR_NOISE))
        self.graph_gt.push_back(gtsam.PriorFactorPose3(O(0), gtsam.Pose3(gtsam.Rot3.Roll(-0.5*np.pi), np.array([0, 0, -self.height])), self.OBJECT_PRIOR_NOISE_GT))
        self.initial_estimate_gt.insert(G(0), gtsam.Pose3(gtsam.Rot3(), np.zeros(3)))
        self.initial_estimate_gt.insert(O(0), gtsam.Pose3(gtsam.Rot3.Roll(-0.5*np.pi), np.array([0, 0, -self.height])))
        self.isam_gt.update(self.graph_gt, self.initial_estimate_gt)
        for _ in range(2): self.isam_gt.update()
        self.graph_gt.resize(0)
        self.initial_estimate_gt.clear()
        self.current_estimate_gt = self.isam_gt.calculateEstimate()
        self.grp_gt = self.current_estimate_gt.atPose3(G(self.i_gt))
        self.obj_gt = self.current_estimate_gt.atPose3(O(self.i_gt))

    def discard_n_reset(self, scale_init_cov=2):

        self.ctl_cov *= scale_init_cov
        self.obj_cov *= scale_init_cov

        if self.ctl_cov[4, 4] < 400: self.ctl_cov[4, 4] = 400
        if self.obj_cov[4, 4] < 400: self.obj_cov[4, 4] = 400

        self.OBJECT_PRIOR_NOISE = gtsam.noiseModel.Gaussian.Covariance(
            self.obj_cov)
        self.CLINE_PRIOR_NOISE = gtsam.noiseModel.Gaussian.Covariance(
            self.ctl_cov)

        self.reset_graph(self.obj, self.ctl)

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
                    np.clip(10 * 0.95**self.i, 1, 10) *
                    np.array([4e-2, np.inf, 4e-2, np.inf, 2e0, np.inf])))
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
            """
            if np.mean([
                    ctl_cov[1, 1] for ctl_cov in list(self.ctl_cov_buffer)[-5:]
            ]) < self.terminal_thres[0] and np.mean([
                    ctl_cov[4, 4] for ctl_cov in list(self.ctl_cov_buffer)[-5:]
            ]) < self.terminal_thres[1] and np.mean([
                    ctl_cov[5, 5] for ctl_cov in list(self.ctl_cov_buffer)[-5:]
            ]) < self.terminal_thres[2]:
                print('threshold satisfied')
                self.terminate = True
            """
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

    def add_new_gt(self, cart_new, tactile_new):

        self.cart = np.array(cart_new)

        xyz_world = self.cart[:3] - self.cart_init[:3]
        self.r_g = R.from_quat(self.cart[3:]) * self.r_convert

        xyz = self.r_g_init.inv().as_matrix().dot(xyz_world)
        ypr = (self.r_g_init.inv() * self.r_g).as_euler('zyx')

        self.gt = np.hstack((xyz, ypr))
        g_rot = R.from_euler('zyx', self.gt[3:]).as_matrix()
        g_trn = self.gt[:3]   
        
        TM_ = self.tactile_gt.copy()
        TM_[3:] *= np.pi / 180
        TM_rot_ = R.from_euler('zyx', TM_[3:]).as_matrix()
        TM_trn_ = TM_[:3]
        TM = tactile_new.copy()
        TM[3:] *= np.pi / 180
        TM_rot = R.from_euler('zyx', TM[3:]).as_matrix()
        TM_trn = TM[:3]
        self.tactile_gt = tactile_new.copy()

        self.i_gt += 1
        self.graph_gt.push_back(gtsam.PriorFactorPose3(G(self.i_gt), gtsam.Pose3(gtsam.Rot3(g_rot), g_trn), self.GRIPPER_PRIOR_NOISE))
        self.graph_gt.push_back(TactileTransformFactor_3D(O(self.i_gt-1), O(self.i_gt), G(self.i_gt-1), G(self.i_gt),
                gtsam.Pose3.between(gtsam.Pose3(gtsam.Rot3(TM_rot_), TM_trn_),
                                    gtsam.Pose3(gtsam.Rot3(TM_rot), TM_trn)), self.TACTILE_NOISE))
            
        self.initial_estimate_gt.insert(G(self.i_gt), gtsam.Pose3(gtsam.Rot3(), np.zeros(3)))
        self.initial_estimate_gt.insert(O(self.i_gt), gtsam.Pose3(gtsam.Rot3.Roll(-0.5*np.pi), np.array([0, 0, -self.height])))
        self.isam_gt.update(self.graph_gt, self.initial_estimate_gt)
        for _ in range(2): self.isam_gt.update()
        self.graph_gt.resize(0)
        self.initial_estimate_gt.clear()
        self.current_estimate_gt = self.isam_gt.calculateEstimate()
        self.grp_gt = self.current_estimate_gt.atPose3(G(self.i_gt))
        self.obj_gt = self.current_estimate_gt.atPose3(O(self.i_gt))