#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage, JointState, ChannelFloat32
from std_msgs.msg import Bool
import numpy as np
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from robot_comm.msg import *
from robot_comm.srv import *
from std_srvs.srv import *
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import *
from collections import deque
import sys
sys.path = sys.path[::-1]
import rospy, math, cv2, os, pickle
import std_srvs.srv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from networkmodels import DecoderFC, EncoderCNN, DecoderRNN
import gc
gc.collect()

from utils_gtsam_devel import gtsam_graph
from scipy.spatial.transform import Rotation as R


class tactile_module:
    def __init__(self, TCP_offset, env_type='hole', verbose=False, isRNN=False):
        self.verbose = verbose
        self.TCP_offset = TCP_offset
        self.env_type = env_type
        self.cha_length = 50.
        self.new_thres = 0.1
        self.gtsam_graph = gtsam_graph(env_type=self.env_type)
        self.gtsam_on = False
        self.new_added1 = False
        self.new_added2 = False

        self.restart_gtsam = False
        self.restart1 = True
        self.restart2 = True
        self.isfresh1 = False
        self.isfresh2 = False
        self.isRNN = isRNN
        self.h_nc = None
        if not isRNN:
            self.load_nn_model()
        else:
            self.load_rnn_model()

        self.image_sub1 = rospy.Subscriber("/raspicam_node1/image/compressed",
                                           CompressedImage,
                                           self.call_back1,
                                           queue_size=1,
                                           buff_size=2**24)
        self.image_sub2 = rospy.Subscriber("/raspicam_node2/image/compressed",
                                           CompressedImage,
                                           self.call_back2,
                                           queue_size=1,
                                           buff_size=2**24)
        self.EGM_cart_sub = rospy.Subscriber("robot2_EGM/GetCartesian",
                                             PoseStamped, self.callback_cart)

        self.data1 = deque(maxlen=1000)
        self.data2 = deque(maxlen=1000)

        self.cart_EGM_ = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float)
        self.cart_EGM = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float)

        self.m, self.n = 320, 427
        self.pad_m, self.pad_n = 160, 213  #145, 200

        self.g1_mean = 82.97
        self.g1_std = 47.74
        self.g2_mean = 76.44
        self.g2_std = 48.14

        self.count_1 = 0
        self.count_2 = 0
        self.sampling_interval = 1

        self.nn_output = np.zeros(6)
        self.nn_output_ema = np.zeros(6)
        self.ema_decay_ = 0.9  #0.8
        self.ema_decay = 0.99

        self.height_est = None
        self.height_cov = None

    def load_nn_model(self):
        """
        model_dataset_name = 'FG_000_210717_large'

        self.xyzypr_limit = np.array(
            [1., 2., 2., 2. / 180 * np.pi, 2. / 180 * np.pi, 4. / 180 * np.pi])

        model_dataset_name = 'FG_000_210726_small'
        self.xyzypr_limit = np.array([
            .5, .8, .8, .8 / 180 * np.pi, .8 / 180 * np.pi, 1.5 / 180 * np.pi
        ])
        """
        model_dataset_name = 'FG_000_210731_very_small'
        self.xyzypr_limit = np.array([
            .25, .5, .5, .2 / 180 * np.pi, .6 / 180 * np.pi, 1.2 / 180 * np.pi
        ])

        save_name = "tactile_model"
        obj = 'all'
        obj = model_dataset_name + '_' + obj
        model_name = 'devel_cnn_fc_no_crop'
        save_model_path = "./weights/" + model_name + "/" + obj + '/'

        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        in_channels = 3
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        img_x, img_y = 300, 218  #
        dropout_cnn = 0.  # dropout probability

        # DecoderFC architecture
        FC_layer_nodes = [512, 512, 256]
        dropout_fc = 0.15
        k = 6  # output_dims (XYZYPR)

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device(
            "cuda" if use_cuda else "cpu")  # use CPU or GPU

        # Create model
        self.cnn_encoder = EncoderCNN(img_x=img_x,
                                      img_y=img_y,
                                      input_channels=in_channels,
                                      fc_hidden1=CNN_fc_hidden1,
                                      fc_hidden2=CNN_fc_hidden2,
                                      drop_p=dropout_cnn,
                                      CNN_embed_dim=CNN_embed_dim).to(
                                          self.device)

        self.fc_decoder = DecoderFC(CNN_embed_dim=CNN_embed_dim,
                                    FC_layer_nodes=FC_layer_nodes,
                                    drop_p=dropout_fc,
                                    output_dim=k).to(self.device)

        self.cnn_encoder.load_state_dict(
            torch.load(save_model_path + save_name + '_cnn_encoder_best.pth'))
        self.fc_decoder.load_state_dict(
            torch.load(save_model_path + save_name + '_decoder_best.pth'))

        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 0:
            print("Using", torch.cuda.device_count(), "GPUs!")
            if torch.cuda.device_count() > 1:
                self.cnn_encoder = nn.DataParallel(self.cnn_encoder)
                self.fc_decoder = nn.DataParallel(self.fc_decoder)

    def load_rnn_model(self):

        model_dataset_name = 'FG_000_210717_large'

        self.xyzypr_limit = np.array(
            [1., 2., 2., 2. / 180 * np.pi, 2. / 180 * np.pi, 4. / 180 * np.pi])

        save_name = "tactile_model"
        obj = 'all'
        obj = model_dataset_name + '_' + obj
        model_name = 'devel_cnn_lstm'
        save_model_path = "./weights/" + model_name + "/" + obj + '/'

        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        in_channels = 3
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        img_x, img_y = 300, 218  #
        dropout_cnn = 0.5  # dropout probability

        # DecoderRNN architecture
        RNN_hidden_layers = 2
        RNN_hidden_nodes = 512
        RNN_FC_dim = 256
        dropout_rnn = 0.5
        k = 6  # output_dims (XYZYPR)

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device(
            "cuda" if use_cuda else "cpu")  # use CPU or GPU

        # Create model
        self.cnn_encoder = EncoderCNN(img_x=img_x,
                                      img_y=img_y,
                                      input_channels=in_channels,
                                      fc_hidden1=CNN_fc_hidden1,
                                      fc_hidden2=CNN_fc_hidden2,
                                      drop_p=dropout_cnn,
                                      CNN_embed_dim=CNN_embed_dim).to(
                                          self.device)

        self.rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim,
                                      h_RNN_layers=RNN_hidden_layers,
                                      h_RNN=RNN_hidden_nodes,
                                      h_FC_dim=RNN_FC_dim,
                                      drop_p=dropout_rnn,
                                      output_dim=k).to(self.device)

        self.cnn_encoder.load_state_dict(
            torch.load(save_model_path + save_name + '_cnn_encoder_best.pth'))
        self.rnn_decoder.load_state_dict(
            torch.load(save_model_path + save_name + '_decoder_best.pth'))

    def callback_cart(self, data):
        p = data.pose.position
        o = data.pose.orientation
        cart_EGM_raw = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w],
                                dtype=np.float)
        r = (R.from_quat(cart_EGM_raw[3:])).as_matrix()
        cart_EGM_raw[:3] += r.dot(-self.TCP_offset)
        self.cart_EGM = cart_EGM_raw.copy()
        if self.restart_gtsam:
            self.data1.clear()
            self.data2.clear()
            self.cart_EGM_ = self.cart_EGM.copy()
            self.gtsam_graph.restart(self.cart_EGM, self.height_est,
                                     self.height_cov)
            self.height_est, self.height_cov = None, None
            self.restart_gtsam = False

    def call_back1(self, data):
        t = time.time()
        np_arr = np.fromstring(data.data, np.uint8)
        raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #raw_img = cv2.imread(
        #    '/home/mcube/sangwoon/data/FG_000_210423_slip_1/rectangle/1/g1_0.jpg'
        #)

        img = raw_img[int(self.m / 2) - self.pad_m:int(self.m / 2) +
                      self.pad_m,
                      int(self.n / 2) - self.pad_n:int(self.n / 2) +
                      self.pad_n, :]
        img = cv2.resize(img, (300, 218)).astype(np.float32)

        img = (img - self.g1_mean) / self.g1_std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(np.expand_dims(img, 0), 0)

        if self.restart1:
            self.X1_0 = torch.from_numpy(img).type(torch.cuda.FloatTensor)
            self.h_nc = None
            self.restart1 = False

        self.X1 = torch.from_numpy(img).type(torch.cuda.FloatTensor)

        self.cnn_encoder.eval()
        if not self.isRNN:
            self.fc_decoder.eval()
        else:
            self.rnn_decoder.eval()

        self.X1_0 = self.X1_0.to(self.device)
        self.X1 = self.X1.to(self.device)

        self.isfresh1 = True

        if self.isfresh1 and self.isfresh2:
            with torch.no_grad():
                if not self.isRNN:
                    nn_output = self.fc_decoder(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.device).detach().cpu().numpy()[0, 0, :]
                else:
                    nn_output, self.h_nc = self.rnn_decoder.forward_single(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.h_nc, self.device)
                    nn_output = nn_output.detach().cpu().numpy()[0, 0, :]
            nn_output *= self.xyzypr_limit
            nn_output[3:] *= 180 / np.pi
            self.nn_output *= self.ema_decay_
            self.nn_output += (1 - self.ema_decay_) * nn_output
            self.nn_output_ema *= self.ema_decay
            self.nn_output_ema += (1 - self.ema_decay) * nn_output
            self.isfresh1 = False
            self.isfresh2 = False

        #print(len(self.nn_output))
        if self.verbose:
            print(
                "{:+1.2f} {:+1.2f} {:+1.2f} {:+1.2f} {:+1.2f} {:+1.2f}".format(
                    self.nn_output[0], self.nn_output[1], self.nn_output[2],
                    self.nn_output[3], self.nn_output[4], self.nn_output[5]))

        d_rot = (R.from_quat(self.cart_EGM_[3:]).inv() *
                 R.from_quat(self.cart_EGM[3:])).as_rotvec()
        d_rot *= self.cha_length
        d_trn = self.cart_EGM[:3] - self.cart_EGM_[:3]
        delta = np.linalg.norm(np.hstack((d_rot, d_trn)))
        """
        if self.new_added2:
            self.data1.append(
                [raw_img, t, None, self.cart_EGM, self.nn_output])
            self.new_added2 = False
        el
        """
        if delta > self.new_thres:
            #print(delta)
            self.cart_EGM_ = self.cart_EGM.copy()
            self.data1.append(
                [raw_img, t, None, self.cart_EGM,
                 self.nn_output.copy()])
            self.new_added1 = True
            if self.gtsam_on:
                self.gtsam_graph.add_new(self.cart_EGM, self.nn_output.copy())
                #self.gtsam_graph.add_new(self.cart_EGM, self.nn_output_ema)

    def call_back2(self, data):
        t = time.time()
        np_arr = np.fromstring(data.data, np.uint8)
        raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #raw_img = cv2.imread(
        #    '/home/mcube/sangwoon/data/FG_000_210423_slip_1/rectangle/1/g2_0.jpg'
        #)

        img = raw_img[int(self.m / 2) - self.pad_m:int(self.m / 2) +
                      self.pad_m,
                      int(self.n / 2) - self.pad_n:int(self.n / 2) +
                      self.pad_n, :]
        img = cv2.resize(img, (300, 218)).astype(np.float32)

        img = (img - self.g2_mean) / self.g2_std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(np.expand_dims(img, 0), 0)

        if self.restart2:
            self.X2_0 = torch.from_numpy(img).type(torch.cuda.FloatTensor)
            self.restart2 = False

        self.X2 = torch.from_numpy(img).type(torch.cuda.FloatTensor)

        self.cnn_encoder.eval()
        if not self.isRNN:
            self.fc_decoder.eval()
        else:
            self.rnn_decoder.eval()

        self.X2_0 = self.X2_0.to(self.device)
        self.X2 = self.X2.to(self.device)

        self.isfresh2 = True

        if self.isfresh1 and self.isfresh2:
            with torch.no_grad():
                if not self.isRNN:
                    nn_output = self.fc_decoder(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.device).detach().cpu().numpy()[0, 0, :]
                else:
                    nn_output, self.h_nc = self.rnn_decoder.forward_single(
                        self.cnn_encoder(self.X1_0), self.cnn_encoder(self.X1),
                        self.cnn_encoder(self.X2_0), self.cnn_encoder(self.X2),
                        self.h_nc, self.device)
                    nn_output = nn_output.detach().cpu().numpy()[0, 0, :]
            nn_output *= self.xyzypr_limit
            nn_output[3:] *= 180 / np.pi
            self.nn_output *= self.ema_decay_
            self.nn_output += (1 - self.ema_decay_) * nn_output
            self.nn_output_ema *= self.ema_decay
            self.nn_output_ema += (1 - self.ema_decay) * nn_output
            self.isfresh1 = False
            self.isfresh2 = False

        d_rot = (R.from_quat(self.cart_EGM_[3:]).inv() *
                 R.from_quat(self.cart_EGM[3:])).as_rotvec()
        d_rot *= self.cha_length
        d_trn = self.cart_EGM[:3] - self.cart_EGM_[:3]
        delta = np.linalg.norm(np.hstack((d_rot, d_trn)))
        if self.new_added1:
            self.data2.append(
                [raw_img, t, None, self.cart_EGM,
                 self.nn_output.copy()])
            self.new_added1 = False
        """
        elif delta > self.new_thres:
            self.cart_EGM_ = self.cart_EGM.copy()
            self.data2.append(
                [raw_img, t, None, self.cart_EGM, self.nn_output])
            self.new_added1 = True
            if self.gtsam_on:
                self.gtsam_graph.add_new(self.cart_EGM, self.nn_output)
        """


def main():
    print("start")
    rospy.init_node('tactile_module', anonymous=True)
    while not rospy.is_shutdown():
        tactile_module = tactile_module(TCP_offset=np.array([0, -6.6, 12.]), verbose=True)
        rospy.tactile_module()


if __name__ == "__main__":
    main()