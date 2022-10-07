import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils import data
import cv2
import os
from os import path
from scipy.spatial.transform import Rotation as R
import time
import zipfile
import io


class Dataset_Fixed_Base_FC(data.Dataset):
    def __init__(self,
                 file_folder,
                 xyzypr_limit,
                 pairs_for_seq,
                 TCP_offset=np.array([0, -6.6, 12.])):

        self.r_convert = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.file_folder = file_folder
        self.xyzypr_limit = xyzypr_limit
        self.pairs_for_seq = pairs_for_seq
        self.TCP_offset = TCP_offset
        self.g1_mean = 82.97
        self.g1_std = 47.74
        self.g2_mean = 76.44
        self.g2_std = 48.14

    def __len__(self):
        return len(self.file_folder) * self.pairs_for_seq

    def read_data(self, gripper_num, range_list, data_path):
        img_seq = []
        m, n = 320, 427
        pad_m = 160 #145
        pad_n = 213 #200

        for i in range_list:
            if gripper_num == 1:
                img = cv2.imread(data_path + "g1_" + str(i) + ".jpg")
            elif gripper_num == 2:
                img = cv2.imread(data_path + "g2_" + str(i) + ".jpg")
            else:
                raise ("invalid gripper number")

            img = img[int(m / 2) - pad_m:int(m / 2) + pad_m,
                      int(n / 2) - pad_n:int(n / 2) + pad_n, :]
            img = cv2.resize(img, (300, 218)).astype(np.float32)

            img_temp = img.copy()
            if gripper_num == 1:
                img_temp = (img_temp - self.g1_mean) / self.g1_std
            elif gripper_num == 2:
                img_temp = (img_temp - self.g2_mean) / self.g2_std

            img_seq.append(img_temp.transpose(2, 0, 1))

        img_seq = np.array(img_seq)
        X = torch.from_numpy(img_seq).type(torch.FloatTensor)

        return X

    def load_data(self, data_path):
        cart_g1 = np.load(data_path + "cart_g1_rock.npy")
        cart_init = np.load(data_path + "cart_init.npy")

        i1, i2 = np.random.choice(range(0, len(cart_g1)), 2, replace=False)

        X11 = self.read_data(1, [i1], data_path)
        X12 = self.read_data(1, [i2], data_path)
        X21 = self.read_data(2, [i1], data_path)
        X22 = self.read_data(2, [i2], data_path)

        xyz_world = cart_g1[i2, :3] - cart_g1[i1, :3]
        r_g_1 = R.from_quat(cart_g1[[i1], 3:]) * self.r_convert
        r_g_2 = R.from_quat(cart_g1[[i2], 3:]) * self.r_convert
        xyz = r_g_1.inv().as_matrix().dot(xyz_world)
        ypr = (r_g_1.inv() * r_g_2).as_euler('zyx')
        gt = np.hstack((xyz, ypr))
        gt = gt / self.xyzypr_limit

        Y = torch.from_numpy(gt).type(torch.FloatTensor)
        return X11, X12, X21, X22, Y

    def load_sequence(self, data_path):
        cart_g1 = np.load(data_path + "cart_g1_rock.npy")
        cart_init = np.load(data_path + "cart_init.npy")

        X11 = self.read_data(1, [0]*len(cart_g1), data_path)
        X12 = self.read_data(1, list(range(len(cart_g1))), data_path)
        X21 = self.read_data(2, [0]*len(cart_g1), data_path)
        X22 = self.read_data(2, list(range(len(cart_g1))), data_path)

        xyz_world = cart_g1[:,:3] - cart_init[:3]
        r_g_init = R.from_quat(cart_init[3:]) * self.r_convert
        r_g = R.from_quat(cart_g1[:,3:]) * self.r_convert        
        xyz = r_g_init.inv().as_matrix().dot(xyz_world.T).T
        ypr = (r_g_init.inv() * r_g).as_euler('zyx')
        gt = np.hstack((xyz,ypr))
        gt = gt / self.xyzypr_limit

        Y = torch.from_numpy(gt).type(torch.FloatTensor)
        return X11, X12, X21, X22, Y

    def __getitem__(self, index):
        filename = self.file_folder[int(index / self.pairs_for_seq)]
        X11, X12, X21, X22, Y = self.load_data(filename)
        return X11, X12, X21, X22, Y


class Dataset_Fixed_Base_LSTM(data.Dataset):
    def __init__(self, file_folder, xyzypr_limit, sequence_len):
        self.r_convert = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        self.file_folder = file_folder
        self.xyzypr_limit = xyzypr_limit
        self.sequence_len = sequence_len
        self.g1_mean = 82.97
        self.g1_std = 47.74
        self.g2_mean = 76.44
        self.g2_std = 48.14

    def __len__(self):
        return len(self.file_folder)

    def read_data(self, gripper_num, actual_len, range_list, data_path):
        img_seq = []
        m, n = 320, 427
        pad_m = 145
        pad_n = 200

        for i in range_list:
            if gripper_num == 1:
                if i < actual_len:
                    img = cv2.imread(data_path + "g1_" + str(i) + ".jpg")
                else:
                    img = cv2.imread(data_path + "g1_" + str(actual_len - 1) +
                                     ".jpg")
            elif gripper_num == 2:
                if i < actual_len:
                    img = cv2.imread(data_path + "g2_" + str(i) + ".jpg")
                else:
                    img = cv2.imread(data_path + "g2_" + str(actual_len - 1) +
                                     ".jpg")
            else:
                raise ("invalid gripper number")

            img = img[int(m / 2) - pad_m:int(m / 2) + pad_m,
                      int(n / 2) - pad_n:int(n / 2) + pad_n, :]
            img = cv2.resize(img, (300, 218)).astype(np.float32)

            img_temp = img.copy()
            if gripper_num == 1:
                img_temp = (img_temp - self.g1_mean) / self.g1_std
            elif gripper_num == 2:
                img_temp = (img_temp - self.g2_mean) / self.g2_std

            img_seq.append(img_temp.transpose(2, 0, 1))

        img_seq = np.array(img_seq)
        X = torch.from_numpy(img_seq).type(torch.FloatTensor)

        return X

    def load_data(self, data_path):
        cart_g1 = np.load(data_path + "cart_g1_rock.npy")
        actual_len = len(cart_g1)
        cart_init = np.load(data_path + "cart_init.npy")
        if len(cart_g1) < self.sequence_len:
            cart_g1 = np.vstack(
                (cart_g1,
                 np.tile(cart_g1[[-1], :],
                         (self.sequence_len - len(cart_g1), 1))))

        X1 = self.read_data(1, actual_len, list(range(0, self.sequence_len)),
                            data_path)
        X2 = self.read_data(2, actual_len, list(range(0, self.sequence_len)),
                            data_path)

        xyz_world = cart_g1[:self.sequence_len, :3] - cart_init[:3]
        r_g_init = R.from_quat(cart_init[3:]) * self.r_convert
        r_g = R.from_quat(cart_g1[:self.sequence_len, 3:]) * self.r_convert
        xyz = r_g_init.inv().as_matrix().dot(xyz_world.T).T
        ypr = (r_g_init.inv() * r_g).as_euler('zyx')
        gt = np.hstack((xyz, ypr))
        gt = gt / self.xyzypr_limit

        Y = torch.from_numpy(gt).type(torch.FloatTensor)
        return X1, X2, Y

    def __getitem__(self, index):
        filename = self.file_folder[index]
        X1, X2, Y = self.load_data(filename)
        return X1, X2, Y


def data_selection(dataset_name,
                   xyzypr_limit,
                   sequence_len,
                   pairs_for_seq,
                   mode,
                   num_data=400):

    if 'gridsan' in os.getcwd():
        data_folder = '/home/gridsan/sangwoon/data/'
    elif 'mcube' in os.getcwd():
        data_folder = '/home/mcube/sangwoon/data/'
    else:
        data_folder = '/home/devicereal/projects/tactile_FG/data/'

    circle_folder = data_folder + dataset_name + '/circle/'
    hexagon_folder = data_folder + dataset_name + '/hexagon/'
    ellipse_folder = data_folder + dataset_name + '/ellipse/'
    rectangle_folder = data_folder + dataset_name + '/rectangle/'

    folder_list = [
        circle_folder, hexagon_folder, ellipse_folder, rectangle_folder
    ]

    print('start loading data .....')
    random.seed(0)
    train_folder = []
    valid_folder = []
    for folder in folder_list:
        all_folder_temp = []
        for i in range(num_data):
            if path.exists(folder + str(i)):
                path_2save = folder + str(i) + '/'
                all_folder_temp.append(path_2save)
        length = int(len(all_folder_temp) * 0.8)
        random.shuffle(all_folder_temp)
        train_folder += all_folder_temp[:length]
        valid_folder += all_folder_temp[length:]
    print('data loaded .....')

    num_of_train = len(train_folder)
    num_of_valid = len(valid_folder)
    print(num_of_train, num_of_valid)

    if mode == 'fc':
        train_set, valid_set = Dataset_Fixed_Base_FC(
            train_folder, xyzypr_limit,
            pairs_for_seq), Dataset_Fixed_Base_FC(valid_folder, xyzypr_limit,
                                                  pairs_for_seq)
    elif mode == 'lstm':
        train_set, valid_set = Dataset_Fixed_Base_LSTM(
            train_folder, xyzypr_limit,
            sequence_len), Dataset_Fixed_Base_LSTM(valid_folder, xyzypr_limit,
                                                   sequence_len)

    return train_set, valid_set, num_of_train, num_of_valid