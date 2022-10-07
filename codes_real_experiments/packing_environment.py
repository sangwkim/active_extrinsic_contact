#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage, JointState, ChannelFloat32
from std_msgs.msg import Bool, Float32MultiArray
import numpy as np
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from visualization_msgs.msg import *
from wsg_50_common.srv import *
from robot_comm.srv import *
import sys
sys.path = sys.path[::-1]
import rospy, math, cv2, os, pickle
from geometry_msgs.msg import PoseStamped
import json
from std_srvs.srv import *
import random
from scipy.spatial.transform import Rotation as R
import os
from tactile_module import tactile_module
import gtsam
from PIL import Image
import glob


class Robot_motion:
    def __init__(self, env_type='hole', stick=True, mode='default'):
        self.TCP_offset = np.array([0,-6,12.]) # This is the mismatch between the actual TCP and the measured cartesian
        self.env_type = env_type
        self.stick = stick

        if mode == 'default':
            self.initialCartesian = self.TCP_convert_inv([-22.78, -385.79, 180, 0.7071, 0.7071, 0, 0])
            self.cartesianOfCircle = self.TCP_convert_inv([-14.5, -384.5, 48.97, 0.7071, 0.7071, 0, 0])
            self.cartesianOfEllipse = self.TCP_convert_inv([-14.7, -469.5, 48.97, 0.7071, 0.7071, 0, 0])
            self.cartesianOfRectangle = self.TCP_convert_inv([-14.5, -217.15, 48.97, 0.7071, 0.7071, 0, 0])
            self.cartesianOfHexagon = self.TCP_convert_inv([-15.2, -299, 48.97, 0.7071, 0.7071, 0, 0])
        elif mode == 'gtcollect':
            self.initialCartesian = self.TCP_convert_inv([-66.76, -300, 200, 1, 0, 0, 0])
            self.cartesianOfCircle = self.TCP_convert_inv([-66.76, -255, 86.97, 1, 0, 0, 0])
            self.cartesianOfEllipse = self.TCP_convert_inv([-66.76, -330, 86.97, 1, 0, 0, 0])
            self.cartesianOfRectangle = self.TCP_convert_inv([-66.76, -255, 86.97, 1, 0, 0, 0])
            self.cartesianOfHexagon = self.TCP_convert_inv([-66.76, -330, 86.97, 1, 0, 0, 0])

        self.cartesianOfCircle_top = list(self.cartesianOfCircle)
        self.cartesianOfCircle_top[2] += 75
        self.cartesianOfEllipse_top = list(self.cartesianOfEllipse)
        self.cartesianOfEllipse_top[2] += 75
        self.cartesianOfRectangle_top = list(self.cartesianOfRectangle)
        self.cartesianOfRectangle_top[2] += 75
        self.cartesianOfHexagon_top = list(self.cartesianOfHexagon)
        self.cartesianOfHexagon_top[2] += 75

        self.objectCartesianDict = {'circle':[self.cartesianOfCircle,self.cartesianOfCircle_top],\
                                   'rectangle':[self.cartesianOfRectangle,self.cartesianOfRectangle_top],\
                                   'hexagon':[self.cartesianOfHexagon,self.cartesianOfHexagon_top],\
                                   'ellipse':[self.cartesianOfEllipse,self.cartesianOfEllipse_top],\
                                   'circle_tight':[self.cartesianOfCircle,self.cartesianOfCircle_top],\
                                   'ellipse_tight':[self.cartesianOfEllipse,self.cartesianOfEllipse_top],\
                                   'hexagon_tight':[self.cartesianOfHexagon,self.cartesianOfHexagon_top]}

        self.object_width = {'circle': 40.,\
                                    'rectangle': 40.,\
                                    'hexagon': 34.,\
                                    'ellipse': 40.,\
                                    'circle_tight': 40.,\
                                    'hexagon_tight': 34.,\
                                    'ellipse_tight': 40.}

        self.Start_EGM = rospy.ServiceProxy('/robot2_ActivateEGM',
                                            robot_ActivateEGM)
        self.Stop_EGM = rospy.ServiceProxy('/robot2_StopEGM', robot_StopEGM)
        self.setSpeed = rospy.ServiceProxy('/robot2_SetSpeed', robot_SetSpeed)
        self.command_pose_pub = rospy.Publisher('/robot2_EGM/SetCartesian',
                                                PoseStamped,
                                                queue_size=100,
                                                latch=True)
        self.clearBuffer = rospy.ServiceProxy('/robot2_ClearBuffer',
                                              robot_ClearBuffer)
        self.addBuffer = rospy.ServiceProxy('/robot2_AddBuffer',
                                            robot_AddBuffer)
        self.executeBuffer = rospy.ServiceProxy('/robot2_ExecuteBuffer',
                                                robot_ExecuteBuffer)
        self.setZone = rospy.ServiceProxy('/robot2_SetZone', robot_SetZone)
        self.mode = 0

        ################################single wall##########################################
        self.cartesianOfGap_onewall_Rectangle = self.TCP_convert_inv([115, -379, 75.0, 0.70803, 0.70618, -0.00185, -0.00185])
        self.cartesianOfGap_onewall_Circle = self.TCP_convert_inv([115, -379, 75, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_onewall_Hexagon = self.TCP_convert_inv([115, -379, 75, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_onewall_Ellipse = self.TCP_convert_inv([115, -379, 75, 0.71074, 0.70338, -0.00737, -0.00737])

        self.cartesianOfGap_onewall_Dict = {
            'rectangle': self.cartesianOfGap_onewall_Rectangle,
            'circle': self.cartesianOfGap_onewall_Circle,
            'hexagon': self.cartesianOfGap_onewall_Hexagon,
            'ellipse': self.cartesianOfGap_onewall_Ellipse
        }

        ###############################holes#######################################
        self.cartesianOfGap_hole_Rectangle = self.TCP_convert_inv([98.4, -317, 75.0, 0.70803, 0.70618, -0.00185, -0.00185])
        self.cartesianOfGap_hole_Circle = self.TCP_convert_inv([156.3, -344.85, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Hexagon = self.TCP_convert_inv([158.3, -289.8, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Ellipse = self.TCP_convert_inv([99, -317.9, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Circle_Tight = self.TCP_convert_inv([236.1, -276.8, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Hexagon_Tight = self.TCP_convert_inv([297.2, -278.1, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])
        self.cartesianOfGap_hole_Ellipse_Tight = self.TCP_convert_inv([295.6, -347.2, 75.0, 0.71074, 0.70338, -0.00737, -0.00737])

        self.cartesianOfGap_hole_Dict = {
            'rectangle': self.cartesianOfGap_hole_Rectangle,
            'circle': self.cartesianOfGap_hole_Circle,
            'hexagon': self.cartesianOfGap_hole_Hexagon,
            'ellipse': self.cartesianOfGap_hole_Ellipse,
            'circle_tight': self.cartesianOfGap_hole_Circle_Tight,
            'hexagon_tight': self.cartesianOfGap_hole_Hexagon_Tight,
            'ellipse_tight': self.cartesianOfGap_hole_Ellipse_Tight,
        }

    def TCP_convert(self, position):
        pos_conv = np.array(position)
        r = (R.from_quat(position[3:])).as_matrix()
        pos_conv[:3] += r.dot(self.TCP_offset)
        return pos_conv

    def TCP_convert_inv(self, position):
        pos_conv = np.array(position)
        r = (R.from_quat(position[3:])).as_matrix()
        pos_conv[:3] += r.dot(-self.TCP_offset)
        return pos_conv

    def move_cart_mm(self, position):
        position = self.TCP_convert(position)
        setCartRos = rospy.ServiceProxy('/robot2_SetCartesian',
                                        robot_SetCartesian)
        setCartRos(position[0], position[1], position[2], position[6],
                   position[3], position[4], position[5])

    def move_cart_series(self, cart_list, speed=[600, 200]):
        self.setZone(4)  # 0~4
        self.clearBuffer()
        self.setSpeed(speed[0], speed[1])
        for cart in cart_list:
            cart = self.TCP_convert(cart)
            self.addBuffer(cart[0], cart[1], cart[2], cart[6], cart[3],
                           cart[4], cart[5])
        self.executeBuffer()
        self.setZone(1)

    def move_cart_add(self, dx=0, dy=0, dz=0):
        #Define ros services
        getCartRos = rospy.ServiceProxy('/robot2_GetCartesian',
                                        robot_GetCartesian)
        setCartRos = rospy.ServiceProxy('/robot2_SetCartesian',
                                        robot_SetCartesian)
        #read current robot pose
        c = getCartRos()
        #move robot to new pose
        setCartRos(c.x + dx, c.y + dy, c.z + dz, c.q0, c.qx, c.qy, c.qz)

    def get_cart(self):
        getCartRos = rospy.ServiceProxy('/robot2_GetCartesian',
                                        robot_GetCartesian)
        c = getCartRos()
        c_array = self.TCP_convert_inv([c.x, c.y, c.z, c.qx, c.qy, c.qz, c.q0])
        return list(c_array)

    def close_gripper_f(self, grasp_speed=50, grasp_force=10, width=40.):
        grasp = rospy.ServiceProxy('/wsg_50_driver/grasp', Move)
        self.ack()
        self.set_grip_force(grasp_force)
        time.sleep(0.1)
        error = grasp(width=width, speed=grasp_speed)
        time.sleep(0.5)

    def home_gripper(self):
        self.ack()
        home = rospy.ServiceProxy('/wsg_50_driver/homing', Empty)
        try:
            error = home()
        except:
            pass
        time.sleep(0.5)
        # print('error', error)

    def open_gripper(self):
        self.ack()
        release = rospy.ServiceProxy('/wsg_50_driver/move', Move)
        release(68.0, 100)
        time.sleep(0.5)

    def set_grip_force(self, val=5):
        set_force = rospy.ServiceProxy('/wsg_50_driver/set_force', Conf)
        error = set_force(val)
        time.sleep(0.2)

    def ack(self):
        srv = rospy.ServiceProxy('/wsg_50_driver/ack', Empty)
        error = srv()
        time.sleep(0.5)

    def get_jointangle(self):
        getJoint = rospy.ServiceProxy('/robot2_GetJoints', robot_GetJoints)
        angle = getJoint()
        return [angle.j1, angle.j2, angle.j3, angle.j4, angle.j5, angle.j6]

    def set_jointangle(self, angle):
        setJoint = rospy.ServiceProxy('/robot2_SetJoints', robot_SetJoints)
        setJoint(angle[0], angle[1], angle[2], angle[3], angle[4], angle[5])

    def grasp_object(self, target_object, graspForce, inposition, random_pose):
        object_cart_info = list(self.objectCartesianDict[target_object])
        objectCartesian = np.array(object_cart_info[0]).copy()
        objectCartesian[:3] += random_pose[:3] #np.array([0, random_pose[1], random_pose[2]])
        objectCartesian[3:] = (R.from_euler('zyx',random_pose[3:]) * R.from_quat(objectCartesian[3:])).as_quat()

        objectCartesian_top = np.array(object_cart_info[1]).copy()
        objectCartesian_top[:3] += random_pose[:3] #np.array([0, random_pose[1], 0])
        objectCartesian_top[3:] = (R.from_euler('zyx',random_pose[3:]) * R.from_quat(objectCartesian_top[3:])).as_quat()

        self.setSpeed(600, 200)
        if not inposition:
            self.move_cart_mm(objectCartesian_top)
            time.sleep(0.5)
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)
        self.close_gripper_f(100, graspForce, self.object_width[target_object])
        time.sleep(0.5)

    def robot_reset(self):
        print('go to the initial position')
        self.move_cart_mm(self.initialCartesian)

    def tran_rotate_robot(self, targetCartesian, x, y, theta):
        relativeVector = np.array([0., 0., 0.])  # 0 12 380
        ratationMatrix1 = (R.from_quat(targetCartesian[3:])).as_matrix()
        ratationMatrix2 = (R.from_euler('z', -theta,
                                        degrees=False)).as_matrix()  #rotate theta
        targetQuaternion = R.from_matrix(
            ratationMatrix2.dot(ratationMatrix1)).as_quat()
        targetCartesian[:3] = np.array(
            targetCartesian[:3]) + ratationMatrix1.dot(
                relativeVector) - ratationMatrix2.dot(ratationMatrix1).dot(
                    relativeVector)
        targetCartesian[3:] = targetQuaternion
        targetCartesian[0] = targetCartesian[0] + x  # add
        targetCartesian[1] = targetCartesian[1] + y  # add
        return targetCartesian

    def pick_up_object(self, target_object, graspForce, inposition,
                       random_pose):
        object_cart_info = list(self.objectCartesianDict[target_object])
        objectCartesian = np.array(object_cart_info[0]).copy()
        objectCartesian[:3] += np.array([0, random_pose[1], random_pose[2]])
        objectCartesian_top = np.array(object_cart_info[1]).copy()
        objectCartesian_top[:3] += np.array([0, random_pose[1], 0])

        self.setSpeed(600, 200)
        if not inposition:
            self.move_cart_mm(objectCartesian_top)
            time.sleep(0.5)
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)
        self.close_gripper_f(100, graspForce, self.object_width[target_object])

        self.move_cart_mm(objectCartesian_top)
        time.sleep(0.5)

        print(self.env_type)

        if self.env_type == 'hole':
            cartesianOfGap = list(self.cartesianOfGap_hole_Dict[target_object])
        elif self.env_type == 'wall':
            cartesianOfGap = list(self.cartesianOfGap_onewall_Dict[target_object])

        cart_positon_top = list(cartesianOfGap)
        cart_positon_top[1] += random_pose[1]
        cart_positon_top[2] += random_pose[2] + 25.
        self.move_cart_mm(cart_positon_top)
        time.sleep(0.5)

        cartesianOfGap[:3] += np.array([0, random_pose[1], random_pose[2]])
        self.move_cart_mm(cartesianOfGap)
        time.sleep(0.5)

    def return_object(self, objectCartesian, objectCartesian_top, random_pose):

        objectCartesian = np.array(objectCartesian)
        objectCartesian_top = np.array(objectCartesian_top)

        self.setSpeed(600, 200)
        self.move_cart_add(0., 0., 100.)
        time.sleep(0.5)

        objectCartesian_top[1] += random_pose[1]
        self.move_cart_mm(objectCartesian_top)
        time.sleep(0.5)

        self.setSpeed(100, 50)

        objectCartesian[:3] += np.array([0, random_pose[1], random_pose[2]])
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)

        self.open_gripper()
        time.sleep(0.2)

    def OU_process_EGM(
            self,
            volatility,
            decay,
            limit,
            num_frame,
            ma_length):

        # Generate the wiggling sequence #
        xyzypr_traj = []
        for i in range(num_frame):
            if i == 0: xyzypr_traj.append(np.zeros((1, 6)))
            else:
                xyzypr_traj.append(decay * xyzypr_traj[-1] +
                                   np.random.normal(np.zeros(6), volatility))
        xyzypr_traj = np.squeeze(np.stack(xyzypr_traj))
        xyzypr_traj_sm = np.zeros((num_frame + ma_length - 1, 6))
        for j in range(6):
            xyzypr_traj_sm[:, j] = np.convolve(
                xyzypr_traj[:, j],
                np.ones(100) / 100,
            )
        xyzypr_traj = np.clip(xyzypr_traj_sm, -limit, limit)
        xyzypr_traj = xyzypr_traj[:num_frame, :]

        # start wiggling #
        xyzquat_init = self.get_cart()
        xyz_init = xyzquat_init[:3]
        R0 = R.from_quat(xyzquat_init[3:])

        print("entering EGM")
        ret = self.Start_EGM(False, 86400)
        time.sleep(.5)
        rate = rospy.Rate(100)

        r_convert = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        for i in range(num_frame):
            xyz = xyz_init + xyzypr_traj[i, :3].dot(r_convert)
            quat = (R0 * R.from_euler('ZXY', xyzypr_traj[i, 3:])).as_quat()
            xyzquat = self.TCP_convert(np.hstack((xyz,quat)))
            pose = PoseStamped()
            pose.pose.position.x = xyzquat[0]
            pose.pose.position.y = xyzquat[1]
            pose.pose.position.z = xyzquat[2]
            pose.pose.orientation.x = xyzquat[3]
            pose.pose.orientation.y = xyzquat[4]
            pose.pose.orientation.z = xyzquat[5]
            pose.pose.orientation.w = xyzquat[6]
            self.command_pose_pub.publish(pose)
            rate.sleep()

        time.sleep(.5)
        ret = self.Stop_EGM()
        print("stopping EGM")
        time.sleep(.5)

    def EGM_warmup(self, duration, rate):
        for i in range(duration):
            xyzquat = self.TCP_convert(self.cart)
            pose = PoseStamped()
            pose.pose.position.x = xyzquat[0]
            pose.pose.position.y = xyzquat[1]
            pose.pose.position.z = xyzquat[2]
            pose.pose.orientation.x = xyzquat[3]
            pose.pose.orientation.y = xyzquat[4]
            pose.pose.orientation.z = xyzquat[5]
            pose.pose.orientation.w = xyzquat[6]
            self.command_pose_pub.publish(pose)
            rate.sleep()

    # functions for contact line estimation 2D display #

    def display_init(self, tactile_module, object_name):
        plt.ion()
        self.plotnum = 0
        self.fig, self.ax = plt.subplots()

        if object_name == 'circle':
            objt = np.array(
                [[17.5 * np.cos(th), 17.5 * np.sin(th)]
                 for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
            hole = np.array([[19.75, 19.75], [-19.75, 19.75],
                                     [-19.75, -19.75], [19.75, -19.75]])
        elif object_name == 'hexagon':
            objt = np.array([[15.156, -8.75], [15.156, 8.75],
                                        [0, 17.5], [-15.156, 8.75],
                                        [-15.156, -8.75], [0, -17.5]])
            hole = np.array([[17.406, 19.75], [-17.406, 19.75],
                                      [-17.406, -19.75], [17.406, -19.75]])
        elif object_name == 'ellipse':
            objt = np.array(
                [[17.5 * np.cos(th), 25. * np.sin(th)]
                 for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
            hole = np.array([[19.75, 27.25], [-19.75, 27.25],
                                      [-19.75, -27.25], [19.75, -27.25]])
        elif object_name == 'rectangle':
            objt = np.array([[17.5, 25.], [-17.5, 25.],
                                          [-17.5, -25.], [17.5, -25.]])
            hole = np.array([[19.75, 27.25], [-19.75, 27.25],
                                        [-19.75, -27.25], [19.75, -27.25]])
        elif object_name == 'circle_tight':
            objt = np.array(
                [[17.5 * np.cos(th), 17.5 * np.sin(th)]
                 for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
            hole = np.array(
                [[19.75 * np.cos(th), 19.75 * np.sin(th)]
                 for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
        elif object_name == 'hexagon_tight':
            objt = np.array([[15.156, -8.75], [15.156, 8.75],
                                        [0, 17.5], [-15.156, 8.75],
                                        [-15.156, -8.75], [0, -17.5]])
            hole = (15.156+2.25)/15.156*np.array([[15.156, -8.75], [15.156, 8.75],
                                        [0, 17.5], [-15.156, 8.75],
                                        [-15.156, -8.75], [0, -17.5]])
        elif object_name == 'ellipse_tight':
            objt = np.array(
                [[17.5 * np.cos(th), 25. * np.sin(th)]
                 for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])
            hole = np.array(
                [[19.75 * np.cos(th), 27.25 * np.sin(th)]
                 for th in np.arange(0, 2 * np.pi, 2 * np.pi / 40)])

        if self.env_type == 'wall':
            hole = np.array([[-45,0.2], [45,0.2],
                                        [45, 35], [-45, 35]])

        dx, dy, th = self.pose_error
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        objt =  R.dot(objt.T).T + np.array([dx, dy])

        self.ax.fill(hole[:, 0], hole[:, 1], facecolor="none", edgecolor='black')
        self.ax.fill(objt[:, 0], objt[:, 1], facecolor="none", edgecolor='black')

        ctt = tactile_module.gtsam_graph.ctl.translation()
        ctp = R.dot(ctt[:2]) + np.array([dx, dy])
        cta = tactile_module.gtsam_graph.ctl.rotation().column(1)

        l = 100
        a = np.arctan2(cta[1], cta[0]) + th
        self.line, = self.ax.plot([0], [0], c='r')
        self.line_nominal, = self.ax.plot([0],[0], c='b')
        self.tilt_arrow, = self.ax.plot([0],[0], c='g')
        self.text = self.ax.text(0, 0, f'{ctt[0]:.2f}, {ctt[1]:.2f}, {ctt[2]:.2f}')
        self.line.set_xdata([ctp[0] - l * np.cos(a), ctp[0] + l * np.cos(a)])
        self.line.set_ydata([ctp[1] - l * np.sin(a), ctp[1] + l * np.sin(a)])
        self.ax.axis('equal')
        self.ax.grid()
        self.ax.set_xlim([-60, 60])
        self.ax.set_ylim([-40, 40])
        if False: # enable this to view display in realtime. It will slowdown the motion
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        plt.savefig(f'./plots/plot_{self.plotnum}.png')

    def display_update(self, tactile_module):
        self.plotnum += 1
        dx, dy, th = self.pose_error
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        ctt = tactile_module.gtsam_graph.ctl.translation()
        ctp = R.dot(ctt[:2]) + np.array([dx, dy])
        cta = tactile_module.gtsam_graph.ctl.rotation().column(1)
        cov = tactile_module.gtsam_graph.ctl_cov

        l = 100
        a = np.arctan2(cta[1], cta[0]) + th
        self.line.set_xdata([ctp[0] - l * np.cos(a), ctp[0] + l * np.cos(a)])
        self.line.set_ydata([ctp[1] - l * np.sin(a), ctp[1] + l * np.sin(a)])
        if tactile_module.gtsam_graph.ctl_nominal:
            ctt_nominal = tactile_module.gtsam_graph.ctl_nominal.translation()
            ctp_nominal = R.dot(ctt_nominal[:2]) + np.array([dx, dy])
            cta_nominal = tactile_module.gtsam_graph.ctl_nominal.rotation().column(1)
            a = np.arctan2(cta_nominal[1], cta_nominal[0]) + th
            self.line_nominal.set_xdata([ctp_nominal[0] - l * np.cos(a), ctp_nominal[0] + l * np.cos(a)])
            self.line_nominal.set_ydata([ctp_nominal[1] - l * np.sin(a), ctp_nominal[1] + l * np.sin(a)])
        self.tilt_arrow.set_xdata([ctp[0], ctp[0] + 1e2*self.tilt_max_vec[0]*np.cos(th-np.pi/2) - 1e2*self.tilt_max_vec[1]*np.sin(th-np.pi/2)])
        self.tilt_arrow.set_ydata([ctp[1], ctp[1] + 1e2*self.tilt_max_vec[0]*np.sin(th-np.pi/2) + 1e2*self.tilt_max_vec[1]*np.cos(th-np.pi/2)])
        self.text.set_text(f'{tactile_module.gtsam_graph.i}, {tactile_module.gtsam_graph.idx_window_begin}\n{ctt[0]:.2f}, {ctt[1]:.2f}, {ctt[2]:.2f}\n rotation var: {cov[1,1]:.3f}\n height var: {cov[4,4]:.1f}\n horizontal var: {cov[5,5]:.1f}')
        if False: # enable this to view display in realtime. It will slowdown the motion
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        plt.savefig(f'./plots/plot_{self.plotnum}.png')

    def display_save_as_gif(self):

        imgs = sorted(glob.glob(os.path.join('./plots', '*.png')), key=os.path.getmtime)
        frames = []
        for i in imgs:
            new_frame = Image.open(i)
            os.remove(i)
            frames.append(new_frame)

        return frames

    def pushdown(
            self,
            tactile_module,
            object_name,
            desired_z_deform=.3,
            l_approx=54.5,
            Kpz= 4e-2,
            z_clip = 0.15,
            Kppit=1e-4, #4e-4,
            Kprol=1e-4, #4e-4,
            Kperpend =2e-4,
            Kperpend_clip =2e-4,
            Kslip = 1e-2,
            Kslip_clip = 1e-2,
            pitch_magnify_p = 6., #9., #6.,
            pitch_magnify_n = 6., #6., #3.,
            max_tilt_deg = 8.,
            min_tilt_deg = 6.,
            EGM_rate = 100,
            frame_num = 1200,
            display_on = True,
            display_interval = 20,
            verbose = False,
            height_est = None,
            height_cov = None
    ):
        time.sleep(.5)

        self.cart_0 = self.get_cart()
        self.cart = self.get_cart()
        self.R0 = (R.from_quat(self.cart[3:])).as_matrix()
        self.wrist_axis_0 = self.R0.dot(np.array([0., 0., -1.]))
        self.wrist_axis_init = self.wrist_axis_0.copy()
        self.wrist_axis = self.R0.dot(np.array([0., 0., -1.]))
        self.wrist_y_0 = self.R0.dot(np.array([1., 0., 0.]))
        self.wrist_y = self.R0.dot(np.array([1., 0., 0.]))
        self.wrist_x_0 = self.R0.dot(np.array([0., 1., 0.]))
        self.wrist_x = self.R0.dot(np.array([0., 1., 0.]))
        self.tilt_origin = self.cart[:3] - l_approx * self.wrist_axis
        self.tilt_max_vec = np.zeros(3)

        pit_tilt = 0
        rol_tilt = 0
        slip_gain = 0
        slip_gain_cumul = 0
        dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema = 0, 0, 0, 0, 0, 0

        print("Entering EGM: Passive Pushdown")
        ret = self.Start_EGM(False, 86400)
        rate = rospy.Rate(EGM_rate)
        time.sleep(0.5)
        self.EGM_warmup(100, rate)
        tactile_module.height_est = height_est
        tactile_module.height_cov = height_cov
        tactile_module.restart_gtsam = True
        tactile_module.gtsam_on = False
        tactile_module.gtsam_graph.stick_on = False

        if display_on:
            self.display_init(tactile_module, object_name)

        time_1 = time.time()

        for i in range(frame_num):
            if tactile_module.gtsam_graph.terminate:
                self.display_update(tactile_module)
                break
            if i % display_interval == 0 and display_on: 
                self.display_update(tactile_module)
            [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output
            [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
            
            dz = -dz
            dyaw = -dyaw
            if dpit > 0: dpit *= pitch_magnify_p
            else: dpit *= pitch_magnify_n
            dz_ema = -dz_ema
            dyaw_ema = -dyaw_ema
            if dpit_ema > 0: dpit_ema *= pitch_magnify_p
            else: dpit_ema *= pitch_magnify_n

            z_dot = np.clip(- Kpz * (desired_z_deform - dz), - Kpz*z_clip, Kpz*z_clip)
            if np.abs(drol) > tactile_module.gtsam_graph.max_deform_limit[5] or np.abs(dpit)/pitch_magnify_p > tactile_module.gtsam_graph.max_deform_limit[4]:
                z_dot = np.clip(z_dot, 0, np.inf)
            
            if dz_ema > 0.03:
                rol_dot = -Kprol * drol
                pit_dot = -Kppit * dpit
            else:
                rol_dot = 0
                pit_dot = 0

            pit_tilt += pit_dot
            rol_tilt += rol_dot

            if tactile_module.gtsam_on == False and dz_ema > 0.03:
                tactile_module.gtsam_on = True
            
            if tactile_module.gtsam_graph.mode_detect_on == False and dz_ema > 0.24 and self.env_type=='hole':
                tactile_module.gtsam_graph.mode_detect_on = True

            self.tilt_origin += z_dot * self.wrist_axis

            tilt_magnitude = (pit_tilt**2 + rol_tilt**2)**0.5
            if tilt_magnitude * 180 / np.pi > max_tilt_deg:
                break
            tilt_dir = np.array([pit_tilt, rol_tilt, 0]) + 1e-5
            tilt_dir /= np.linalg.norm(tilt_dir)
            tilt_axis = self.R0.dot(tilt_dir)

            q_tilt = R.from_quat(
                np.hstack((np.sin(tilt_magnitude / 2) * tilt_axis,
                           np.cos(tilt_magnitude / 2))))

            self.wrist_axis = q_tilt.as_matrix().dot(self.wrist_axis_0)
            tilt_vec = np.cross(self.wrist_axis_init, self.wrist_axis)

            self.tilt_max_vec = tactile_module.gtsam_graph.tilt_max_vec.copy()

            self.wrist_x = q_tilt.as_matrix().dot(self.wrist_x_0)
            self.wrist_y = q_tilt.as_matrix().dot(self.wrist_y_0)
            self.cart[:3] = self.tilt_origin + l_approx * self.wrist_axis
            self.cart[3:] = (q_tilt * R.from_quat(self.cart_0[3:])).as_quat()
            xyzquat = self.TCP_convert(self.cart)
            pose = PoseStamped()
            pose.pose.position.x = xyzquat[0]
            pose.pose.position.y = xyzquat[1]
            pose.pose.position.z = xyzquat[2]
            pose.pose.orientation.x = xyzquat[3]
            pose.pose.orientation.y = xyzquat[4]
            pose.pose.orientation.z = xyzquat[5]
            pose.pose.orientation.w = xyzquat[6]
            self.command_pose_pub.publish(pose)
            rate.sleep()
            if i%10 == 0 and verbose:
                print(f"{dz:.4f}, {dz_ema:.4f}, {drol:.4f}, {drol_ema:.4f}, {dpit:.4f}, {dpit_ema:.4f}, {tactile_module.gtsam_graph.ctl.translation()[2]:.1f}, {tilt_magnitude*180/np.pi:.2f}")

        if dz_ema < 0.05:
            time_2 = None
            time.sleep(.5)
            ret = self.Stop_EGM()
            print("Stopping EGM: Object Inserted!")
            time.sleep(.5)
            Success = None
            return True, None, None, None, None

        if not tactile_module.gtsam_graph.got_nominal and not tactile_module.gtsam_graph.terminate:
            time_2 = None
            time.sleep(.5)
            ret = self.Stop_EGM()
            print("Stopping EGM: Passive Pushdown Failed")
            time.sleep(.5)
            Success = False
            return False, Success, None, time_1, time_2

        else:
            Success = True
            if tactile_module.gtsam_graph.terminate:
                time_2 = None
                time.sleep(.5)
                ret = self.Stop_EGM()
                print("Stopping EGM: Mode Change Detected During Passive Pushdown")
                time.sleep(.5)
                return False, Success, tilt_magnitude, time_1, time_2
            elif tilt_magnitude > min_tilt_deg / 180 * np.pi:
                time_2 = None
                time.sleep(.5)
                ret = self.Stop_EGM()
                print("Stopping EGM: Passive Pushdown Successful")
                time.sleep(.5)
                return False, Success, tilt_magnitude, time_1, time_2
            else:
                print("Continuing EGM: Passive Pushdown Successful, Active Tilting for More Tilting Angle")
                tactile_module.gtsam_on = True
                if self.stick:
                    tactile_module.gtsam_graph.stick_on = True
                self.EGM_warmup(100, rate)

                [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
                dz_ema = -dz_ema
                dyaw_ema = -dyaw_ema
                if dpit_ema > 0: dpit *= pitch_magnify_p
                else: dpit_ema *= pitch_magnify_n
                dz_set = dz_ema.copy()
                drol_set = drol_ema.copy()
                dpit_set = dpit_ema.copy()

                tilt_dot = max_tilt_deg/180*np.pi / frame_num

                tilt_axis = tactile_module.gtsam_graph.ctl.rotation().column(1)
                tilt_axis = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_axis)
                if tilt_axis.dot(self.tilt_max_vec) < 0:
                    tilt_dot *= -1

                time_2 = time.time()

                for i in range(int((min_tilt_deg/180*np.pi - tilt_magnitude)/abs(tilt_dot))):
                    if tactile_module.gtsam_graph.terminate:
                        self.display_update(tactile_module)
                        break
                    if i % display_interval == 0 and display_on: 
                        self.display_update(tactile_module)

                    tilt_axis = tactile_module.gtsam_graph.ctl.rotation().column(1)
                    tilt_axis = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_axis)
                        
                    [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output
                    [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
                    
                    dz = -dz
                    dyaw = -dyaw
                    if dpit > 0: dpit *= pitch_magnify_p
                    else: dpit *= pitch_magnify_n
                    dz_ema = -dz_ema
                    dyaw_ema = -dyaw_ema
                    if dpit_ema > 0: dpit_ema *= pitch_magnify_p
                    else: dpit_ema *= pitch_magnify_n

                    q_tilt = R.from_quat(
                        np.hstack((np.sin(tilt_dot / 2) * tilt_axis,
                                   np.cos(tilt_dot / 2))))

                    tilt_axis_r_g_init = tactile_module.gtsam_graph.ctl.rotation().column(1)
                    tilt_axis_perpend_r_g_init = R.from_rotvec(-np.pi/2*tactile_module.gtsam_graph.obj.rotation().column(2)).as_matrix().dot(tilt_axis_r_g_init)
                    tilt_axis_perpend = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_axis_perpend_r_g_init)

                    grp_rot = R.from_matrix(tactile_module.gtsam_graph.grp.rotation().matrix())
                    tilt_axis_grp = grp_rot.inv().as_matrix().dot(tilt_axis_r_g_init)
                    tilt_axis_perpend_grp = grp_rot.inv().as_matrix().dot(tilt_axis_perpend_r_g_init)

                    dperpend = tilt_axis_perpend_grp[1]*(dpit-dpit_set) + tilt_axis_perpend_grp[0]*(drol-drol_set)

                    tilt_dot_perpend = - np.clip(Kperpend * dperpend, -Kperpend_clip, Kperpend_clip)
                    q_tilt_perpend = R.from_quat(
                        np.hstack((np.sin(tilt_dot_perpend / 2) * tilt_axis_perpend,
                                   np.cos(tilt_dot_perpend / 2))))

                    tilt_origin = tactile_module.gtsam_graph.ctl.translation()
                    tilt_origin = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_origin)
                    tilt_origin += tactile_module.gtsam_graph.cart_init[:3]
                    origin_to_grp = self.cart[:3] - tilt_origin
                    origin_to_grp = (q_tilt_perpend*q_tilt).as_matrix().dot(origin_to_grp)
                    self.wrist_axis = (q_tilt_perpend*q_tilt).as_matrix().dot(self.wrist_axis)
                    tilt_vec = np.cross(self.wrist_axis_init, self.wrist_axis)
                    self.tilt_max_vec = tactile_module.gtsam_graph.tilt_max_vec.copy()
                    #if np.linalg.norm(self.tilt_max_vec) < np.linalg.norm(tilt_vec):
                    #    self.tilt_max_vec = tilt_vec
                    self.wrist_x = (q_tilt_perpend*q_tilt).as_matrix().dot(self.wrist_x)
                    self.wrist_y = (q_tilt_perpend*q_tilt).as_matrix().dot(self.wrist_y)

                    tilt_magnitude = np.arccos(self.wrist_axis.dot(self.wrist_axis_0))

                    self.cart[:3] = tilt_origin + origin_to_grp
                    slip_gain = np.clip(Kslip * ((dpit - dpit_set) * tilt_axis_perpend_grp[0] + (drol_set - drol) * tilt_axis_perpend_grp[1]), -Kslip_clip, Kslip_clip)
                    slip_gain_cumul += slip_gain
                    self.cart[:3] += np.clip(Kslip * ((dpit - dpit_set) * tilt_axis_perpend_grp[0] + (drol_set - drol) * tilt_axis_perpend_grp[1]) * tilt_axis_perpend_grp[0], -Kslip_clip, Kslip_clip) * self.wrist_x
                    self.cart[:3] += np.clip(Kslip * ((dpit - dpit_set) * tilt_axis_perpend_grp[0] + (drol_set - drol) * tilt_axis_perpend_grp[1]) * tilt_axis_perpend_grp[1], -Kslip_clip, Kslip_clip) * self.wrist_y
                    self.cart[:3] += -Kpz*(dz_set-dz)*self.wrist_axis
                    self.cart[3:] = (q_tilt_perpend * q_tilt * R.from_quat(self.cart[3:])).as_quat()

                    xyzquat = self.TCP_convert(self.cart)
                    pose = PoseStamped()
                    pose.pose.position.x = xyzquat[0]
                    pose.pose.position.y = xyzquat[1]
                    pose.pose.position.z = xyzquat[2]
                    pose.pose.orientation.x = xyzquat[3]
                    pose.pose.orientation.y = xyzquat[4]
                    pose.pose.orientation.z = xyzquat[5]
                    pose.pose.orientation.w = xyzquat[6]
                    self.command_pose_pub.publish(pose)
                    rate.sleep()

                    if i%10==0 and verbose:
                        print(f"{dz:.4f}, {dz_ema:.4f}, {drol:.4f}, {drol_ema:.4f}, {dpit:.4f}, {dpit_ema:.4f}, {tactile_module.gtsam_graph.ctl.translation()[2]:.1f}, {tilt_magnitude*180/np.pi:.2f}")
        
                time.sleep(.5)

                if tactile_module.gtsam_graph.terminate:
                    print("Stopping EGM: Mode Change Detected during Active Tilting")
                else:
                    print("Stopping EGM: Passive Tilting and Active Tilting Successful")
                ret = self.Stop_EGM()
                time.sleep(.5)

                return False, Success, tilt_magnitude, time_1, time_2

    def pivoting(self,
            tactile_module,
            is_tilted,
            tilt_magnitude = 6 / 180 * np.pi,
            max_toggle_num = 3,
            Kpz=4e-2, 
            Kperpend =2e-4,
            Kperpend_clip =2e-4, 
            Kslip = 1e-2,
            Kslip_clip = 1e-2,
            pitch_magnify_p = 6.,
            pitch_magnify_n = 3.,
            EGM_rate = 100,
            frame_num = 500,
            display_on = True,
            display_interval = 20,
            verbose = False
    ):
        tactile_module.gtsam_graph.ctl_nominal_ = tactile_module.gtsam_graph.ctl
        tactile_module.gtsam_graph.ctl_cov_nominal_ = tactile_module.gtsam_graph.ctl_cov

        tactile_module.gtsam_on = True
        if self.stick: 
            tactile_module.gtsam_graph.stick_on = True

        tilt_magnitude_ = np.array(tilt_magnitude).copy()
        print("entering EGM: Wiggling")
        ret = self.Start_EGM(False, 86400)
        rate = rospy.Rate(EGM_rate)

        slip_gain_cumul = 0
        time_6 = time.time()

        for f in range(2*max_toggle_num):

            tactile_module.gtsam_on = True
            if self.stick:
                tactile_module.gtsam_graph.stick_on = True
            self.EGM_warmup(100, rate)

            [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
            dz_ema = -dz_ema
            dyaw_ema = -dyaw_ema
            if dpit_ema > 0: dpit_ema *= pitch_magnify_p
            else: dpit_ema *= pitch_magnify_n
            dz_set = dz_ema.copy()
            drol_set = drol_ema.copy()
            dpit_set = dpit_ema.copy()

            tilt_dot = 0.8 * tilt_magnitude_ / frame_num
            tilt_axis = tactile_module.gtsam_graph.ctl.rotation().column(1)
            tilt_axis = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_axis)
            if tilt_axis.dot(self.tilt_max_vec) > 0 and (f+is_tilted)%2!=0:
                tilt_dot *= -1
            elif tilt_axis.dot(self.tilt_max_vec) < 0 and (f+is_tilted)%2==0:
                tilt_dot *= -1    

            for i in range(frame_num):
                if tactile_module.gtsam_graph.terminate:
                    self.display_update(tactile_module)
                    break
                if i % display_interval == 0 and display_on: 
                    self.display_update(tactile_module)

                tilt_axis = tactile_module.gtsam_graph.ctl.rotation().column(1)
                tilt_axis = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_axis)
                  
                [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output
                [dx_ema, dy_ema, dz_ema, dyaw_ema, dpit_ema, drol_ema] = tactile_module.nn_output_ema
                
                dz = -dz
                dyaw = -dyaw
                if dpit > 0: dpit *= pitch_magnify_p
                else: dpit *= pitch_magnify_n
                dz_ema = -dz_ema
                dyaw_ema = -dyaw_ema
                if dpit_ema > 0: dpit_ema *= pitch_magnify_p
                else: dpit_ema *= pitch_magnify_n

                q_tilt = R.from_quat(
                    np.hstack((np.sin(tilt_dot / 2) * tilt_axis,
                               np.cos(tilt_dot / 2))))

                tilt_axis_r_g_init = tactile_module.gtsam_graph.ctl.rotation().column(1)
                tilt_axis_perpend_r_g_init = R.from_rotvec(-np.pi/2*tactile_module.gtsam_graph.obj.rotation().column(2)).as_matrix().dot(tilt_axis_r_g_init)
                tilt_axis_perpend = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_axis_perpend_r_g_init)

                grp_rot = R.from_matrix(tactile_module.gtsam_graph.grp.rotation().matrix())
                tilt_axis_grp = grp_rot.inv().as_matrix().dot(tilt_axis_r_g_init)
                tilt_axis_perpend_grp = grp_rot.inv().as_matrix().dot(tilt_axis_perpend_r_g_init)

                dperpend = tilt_axis_perpend_grp[1]*(dpit-dpit_set) + tilt_axis_perpend_grp[0]*(drol-drol_set)
                #dperpend = tilt_axis_perpend_grp[1]*dpit + tilt_axis_perpend_grp[0]*drol

                tilt_dot_perpend = - np.clip(Kperpend * dperpend, -Kperpend_clip, Kperpend_clip)
                q_tilt_perpend = R.from_quat(
                    np.hstack((np.sin(tilt_dot_perpend / 2) * tilt_axis_perpend,
                               np.cos(tilt_dot_perpend / 2))))

                tilt_origin = tactile_module.gtsam_graph.ctl.translation()
                tilt_origin = tactile_module.gtsam_graph.r_g_init.as_matrix().dot(tilt_origin)
                tilt_origin += tactile_module.gtsam_graph.cart_init[:3]
                origin_to_grp = self.cart[:3] - tilt_origin
                origin_to_grp = (q_tilt_perpend*q_tilt).as_matrix().dot(origin_to_grp)
                self.wrist_axis = (q_tilt_perpend*q_tilt).as_matrix().dot(self.wrist_axis)
                self.wrist_x = (q_tilt_perpend*q_tilt).as_matrix().dot(self.wrist_x)
                self.wrist_y = (q_tilt_perpend*q_tilt).as_matrix().dot(self.wrist_y)

                tilt_magnitude = np.arccos(self.wrist_axis.dot(self.wrist_axis_0))

                self.cart[:3] = tilt_origin + origin_to_grp
                slip_gain = np.clip(Kslip * ((dpit - dpit_set) * tilt_axis_perpend_grp[0] + (drol_set - drol) * tilt_axis_perpend_grp[1]), -Kslip_clip, Kslip_clip)
                slip_gain_cumul += slip_gain
                self.cart[:3] += np.clip(Kslip * ((dpit - dpit_set) * tilt_axis_perpend_grp[0] + (drol_set - drol) * tilt_axis_perpend_grp[1]) * tilt_axis_perpend_grp[0], -Kslip_clip, Kslip_clip) * self.wrist_x
                self.cart[:3] += np.clip(Kslip * ((dpit - dpit_set) * tilt_axis_perpend_grp[0] + (drol_set - drol) * tilt_axis_perpend_grp[1]) * tilt_axis_perpend_grp[1], -Kslip_clip, Kslip_clip) * self.wrist_y
                self.cart[:3] += -Kpz*(dz_set-dz)*self.wrist_axis
                self.cart[3:] = (q_tilt_perpend * q_tilt * R.from_quat(self.cart[3:])).as_quat()

                xyzquat = self.TCP_convert(self.cart)
                pose = PoseStamped()
                pose.pose.position.x = xyzquat[0]
                pose.pose.position.y = xyzquat[1]
                pose.pose.position.z = xyzquat[2]
                pose.pose.orientation.x = xyzquat[3]
                pose.pose.orientation.y = xyzquat[4]
                pose.pose.orientation.z = xyzquat[5]
                pose.pose.orientation.w = xyzquat[6]
                self.command_pose_pub.publish(pose)
                rate.sleep()

                if i%10==0 and verbose:
                    print(f"{dz:.4f}, {dz_ema:.4f}, {drol:.4f}, {drol_ema:.4f}, {dpit:.4f}, {dpit_ema:.4f}, {tactile_module.gtsam_graph.ctl.translation()[2]:.1f}, {tilt_magnitude*180/np.pi:.2f}, {slip_gain_cumul:.2f}")

            if tactile_module.gtsam_graph.terminate:
                
                time.sleep(.5)
                ret = self.Stop_EGM()
                if tactile_module.gtsam_graph.mode_detected:
                    print("stopping EGM: mode detected")
                else:
                    print("stopping EGM: mode not detected")
                time.sleep(.5)

                return time_6
            
        time.sleep(.5)
        ret = self.Stop_EGM()
        if tactile_module.gtsam_graph.mode_detected:
            print("stopping EGM: mode detected")
        else:
            print("stopping EGM: mode not detected")
        time.sleep(.5)

        return time_6


    def rocking(self,
            tactile_module,
            desired_z_deform=.3,
            l_approx=54.5,
            Kpz_rock=0,
            Kpz_release=1e-2,
            Kppit=2e-4,
            Kprol=2e-4,
            pitch_magnify_p = 4.,#4.,
            pitch_magnify_n = 3.,#2.,
            max_angle=4.,
            EGM_rate = 100,
            frame_num = 600,
            display_on = True,
            display_interval = 20,
            verbose = False,
            desired_rol = 1.,#.6,
            desired_pit = .6,#.6,
            frame_num_tilt = 500,
            frame_num_rock = 2000
    ):
        tactile_module.gtsam_graph.nominal_thres = [1.5,2.]
        tactile_module.gtsam_graph.set_current_pose_as_origin()

        time.sleep(1)
        self.cart_0 = self.get_cart()
        self.cart = self.get_cart()
        self.R0 = (R.from_quat(self.cart[3:])).as_matrix()
        self.R = (R.from_quat(self.cart[3:])).as_matrix()
        pit_tilt = 0
        rol_tilt = 0
        self.wrist_axis_0 = self.R0.dot(np.array([0., 0., -1.]))
        self.wrist_axis = self.R.dot(np.array([0., 0., -1.]))
        self.tilt_origin = self.cart[:3] - l_approx * self.wrist_axis

        print("entering EGM: Rocking")
        ret = self.Start_EGM(False, 86400)
        rate = rospy.Rate(EGM_rate)
        self.EGM_warmup(100, rate)

        tactile_module.gtsam_on = True
        tactile_module.gtsam_graph.stick_on = False
        if self.env_type == 'hole':
            tactile_module.gtsam_graph.mode_detect_on = True

        time_3 = time.time()
        print("             Tilting for Rocking")

        for i in range(frame_num_tilt):
            if tactile_module.gtsam_graph.terminate:
                self.display_update(tactile_module)
                break
            if i % display_interval == 0 and display_on: 
                self.display_update(tactile_module)
            desired_roll_deform = desired_rol * i / frame_num_tilt
            desired_pitch_deform = 0
            [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output            
            dz = -dz
            dyaw = -dyaw
            if dpit > 0: dpit *= pitch_magnify_p
            else: dpit *= pitch_magnify_n

            z_dot = -Kpz_rock * (desired_z_deform - dz)
            rol_dot = Kprol * (desired_roll_deform - drol)
            pit_dot = Kppit * (desired_pitch_deform - dpit)

            self.tilt_origin += z_dot * self.wrist_axis

            pit_tilt += pit_dot
            rol_tilt += rol_dot
            pit_tilt = np.clip(pit_tilt, -max_angle / 180. * np.pi,
                               max_angle / 180. * np.pi)
            rol_tilt = np.clip(rol_tilt, -max_angle / 180. * np.pi,
                               max_angle / 180. * np.pi)
            tilt_magnitude = (
                pit_tilt**2 + rol_tilt**2
            )**0.5
            tilt_dir = np.array([pit_tilt, rol_tilt, 0]) + 1e-5
            tilt_dir /= np.linalg.norm(tilt_dir)
            tilt_axis = self.R0.dot(tilt_dir)

            q_tilt = R.from_quat(
                np.hstack((np.sin(tilt_magnitude / 2) * tilt_axis,
                           np.cos(tilt_magnitude / 2))))

            self.wrist_axis = q_tilt.as_matrix().dot(self.wrist_axis_0)
            tilt_vec = np.cross(self.wrist_axis_init, self.wrist_axis)
            self.tilt_max_vec = tactile_module.gtsam_graph.tilt_max_vec.copy()
            #if np.linalg.norm(self.tilt_max_vec) < np.linalg.norm(tilt_vec):
            #    self.tilt_max_vec = tilt_vec
            self.cart[:3] = self.tilt_origin + l_approx * self.wrist_axis
            self.cart[3:] = (q_tilt * R.from_quat(self.cart_0[3:])).as_quat()
            xyzquat = self.TCP_convert(self.cart)
            pose = PoseStamped()
            pose.pose.position.x = xyzquat[0]
            pose.pose.position.y = xyzquat[1]
            pose.pose.position.z = xyzquat[2]
            pose.pose.orientation.x = xyzquat[3]
            pose.pose.orientation.y = xyzquat[4]
            pose.pose.orientation.z = xyzquat[5]
            pose.pose.orientation.w = xyzquat[6]
            self.command_pose_pub.publish(pose)
            if i%10==0 and verbose:
                print(f"{dz:.4f}, {desired_z_deform:.4f}, {drol:.4f}, {desired_roll_deform:.4f}, {dpit:.4f}, {desired_pitch_deform:.4f}, {tilt_magnitude*180/np.pi:.2f}")

            rate.sleep()
        
        self.EGM_warmup(50, rate)
        time_4 = time.time()
        print("             Rocking")

        tactile_module.gtsam_graph.obj_tilt_buffer.clear()
        tactile_module.gtsam_graph.grp_tilt_buffer.clear()

        for i in range(int(1.25*frame_num_rock)):

            if i == frame_num_rock:
                tactile_module.gtsam_graph.check_nominal_after_rock()

            if tactile_module.gtsam_graph.terminate:
                self.display_update(tactile_module)
                break
            if i % display_interval == 0 and display_on: 
                self.display_update(tactile_module)
            desired_roll_deform = desired_rol * np.cos(
                i / float(frame_num_rock) * 2 * np.pi)
            desired_pitch_deform = desired_pit * np.sin(
                i / float(frame_num_rock) * 2 * np.pi)

            [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output            
            dz = -dz
            dyaw = -dyaw
            if dpit > 0: dpit *= pitch_magnify_p
            else: dpit *= pitch_magnify_n

            z_dot = -Kpz_rock * (desired_z_deform - dz)
            rol_dot = Kprol * (desired_roll_deform - drol)
            pit_dot = Kppit * (desired_pitch_deform - dpit)

            self.tilt_origin += z_dot * self.wrist_axis

            pit_tilt += pit_dot
            rol_tilt += rol_dot
            pit_tilt = np.clip(pit_tilt, -max_angle / 180. * np.pi,
                               max_angle / 180. * np.pi)
            rol_tilt = np.clip(rol_tilt, -max_angle / 180. * np.pi,
                               max_angle / 180. * np.pi)
            tilt_magnitude = (
                pit_tilt**2 + rol_tilt**2
            )**0.5
            tilt_dir = np.array([pit_tilt, rol_tilt, 0]) + 1e-5
            tilt_dir /= np.linalg.norm(tilt_dir)
            tilt_axis = self.R0.dot(tilt_dir)

            q_tilt = R.from_quat(
                np.hstack((np.sin(tilt_magnitude / 2) * tilt_axis,
                           np.cos(tilt_magnitude / 2))))

            self.wrist_axis = q_tilt.as_matrix().dot(self.wrist_axis_0)
            tilt_vec = np.cross(self.wrist_axis_init, self.wrist_axis)
            self.tilt_max_vec = tactile_module.gtsam_graph.tilt_max_vec.copy()
            #if np.linalg.norm(self.tilt_max_vec) < np.linalg.norm(tilt_vec):
            #    self.tilt_max_vec = tilt_vec
            self.cart[:3] = self.tilt_origin + l_approx * self.wrist_axis
            self.cart[3:] = (q_tilt * R.from_quat(self.cart_0[3:])).as_quat()
            xyzquat = self.TCP_convert(self.cart)
            pose = PoseStamped()
            pose.pose.position.x = xyzquat[0]
            pose.pose.position.y = xyzquat[1]
            pose.pose.position.z = xyzquat[2]
            pose.pose.orientation.x = xyzquat[3]
            pose.pose.orientation.y = xyzquat[4]
            pose.pose.orientation.z = xyzquat[5]
            pose.pose.orientation.w = xyzquat[6]
            self.command_pose_pub.publish(pose)
            if i%10==0 and verbose:
                print(f"{dz:.4f}, {desired_z_deform:.4f}, {drol:.4f}, {desired_roll_deform:.4f}, {dpit:.4f}, {desired_pitch_deform:.4f}, {tilt_magnitude*180/np.pi:.2f}")
            rate.sleep()
        
        self.EGM_warmup(50, rate)
        time_5 = time.time()
        print("             Release Tilting")

        for i in range(frame_num_tilt):
            if tactile_module.gtsam_graph.terminate:
                self.display_update(tactile_module)
                break
            if i % display_interval == 0 and display_on: 
                self.display_update(tactile_module)
            desired_roll_deform = 0
            desired_pitch_deform = (frame_num_tilt-i)/frame_num_tilt * desired_pit
            [dx, dy, dz, dyaw, dpit, drol] = tactile_module.nn_output            
            dz = -dz
            dyaw = -dyaw
            if dpit > 0: dpit *= pitch_magnify_p
            else: dpit *= pitch_magnify_n

            z_dot = -Kpz_release * (desired_z_deform - dz)
            rol_dot = Kprol * (desired_roll_deform - drol)
            pit_dot = Kppit * (desired_pitch_deform - dpit)

            self.tilt_origin += z_dot * self.wrist_axis

            pit_tilt += pit_dot
            rol_tilt += rol_dot
            pit_tilt = np.clip(pit_tilt, -max_angle / 180. * np.pi,
                               max_angle / 180. * np.pi)
            rol_tilt = np.clip(rol_tilt, -max_angle / 180. * np.pi,
                               max_angle / 180. * np.pi)
            tilt_magnitude = (
                pit_tilt**2 + rol_tilt**2
            )**0.5
            tilt_dir = np.array([pit_tilt, rol_tilt, 0]) + 1e-5
            tilt_dir /= np.linalg.norm(tilt_dir)
            tilt_axis = self.R0.dot(tilt_dir)

            q_tilt = R.from_quat(
                np.hstack((np.sin(tilt_magnitude / 2) * tilt_axis,
                           np.cos(tilt_magnitude / 2))))

            self.wrist_axis = q_tilt.as_matrix().dot(self.wrist_axis_0)
            tilt_vec = np.cross(self.wrist_axis_init, self.wrist_axis)
            self.tilt_max_vec = tactile_module.gtsam_graph.tilt_max_vec.copy()
            #if np.linalg.norm(self.tilt_max_vec) < np.linalg.norm(tilt_vec):
            #    self.tilt_max_vec = tilt_vec
            self.cart[:3] = self.tilt_origin + l_approx * self.wrist_axis
            self.cart[3:] = (q_tilt * R.from_quat(self.cart_0[3:])).as_quat()
            xyzquat = self.TCP_convert(self.cart)
            pose = PoseStamped()
            pose.pose.position.x = xyzquat[0]
            pose.pose.position.y = xyzquat[1]
            pose.pose.position.z = xyzquat[2]
            pose.pose.orientation.x = xyzquat[3]
            pose.pose.orientation.y = xyzquat[4]
            pose.pose.orientation.z = xyzquat[5]
            pose.pose.orientation.w = xyzquat[6]
            self.command_pose_pub.publish(pose)
            if i%10==0 and verbose:
                print(f"{dz:.4f}, {desired_z_deform:.4f}, {drol:.4f}, {desired_roll_deform:.4f}, {dpit:.4f}, {desired_pitch_deform:.4f}, {tilt_magnitude*180/np.pi:.2f}")
            rate.sleep()

        time.sleep(.5)

        ret = self.Stop_EGM()

        if tactile_module.gtsam_graph.got_nominal or tactile_module.gtsam_graph.terminate:
            Success = True
            if tactile_module.gtsam_graph.terminate:
                print("Stopping EGM: Rocking Successful, but Mode Change Detected during Rocking")
            else:
                print("Stopping EGM: Rocking Successful, no Mode Change Detected")
        else:
            Success = False
            if tactile_module.gtsam_graph.terminate:
                print("Stopping EGM: Rocking Failed, Mode Change Detected during Rocking")
            else:
                print("Stopping EGM: Rocking Failed, no Mode Change Detected")

        time.sleep(.5)

        return Success, time_3, time_4, time_5


class Packing_env:
    def __init__(self, data_folder, env_type='hole', stick=True, mode='default'):
        self.env_type = env_type
        self.stick = stick
        self.robot = Robot_motion(mode=mode, stick=stick, env_type=env_type)
        self.robot.data_folder = data_folder
        self.tactile_module = tactile_module(self.robot.TCP_offset, env_type=env_type, verbose=False)
        self.object_name_list = ['circle', 'hexagon', 'ellipse', 'rectangle']
        self.target_object = None

    def parse_data(self):

        image_g1, image_g2, time_g1, time_g2, cart_g1, cart_g2, tact_g1, tact_g2 = [], [], [], [], [], [], [], []

        for i in range(min(len(self.data1),len(self.data2))):
            image_g1.append(self.data1[i][0])
            image_g2.append(self.data2[i][0])
            time_g1.append(self.data1[i][1])
            time_g2.append(self.data2[i][1])
            cart_g1.append(self.data1[i][3])
            cart_g2.append(self.data2[i][3])
            tact_g1.append(self.data1[i][4])
            tact_g2.append(self.data2[i][4])

        image_g1 = np.array(image_g1)
        image_g2 = np.array(image_g2)
        time_g1 = np.array(time_g1)
        time_g2 = np.array(time_g2)
        cart_g1 = np.array(cart_g1)
        cart_g2 = np.array(cart_g2)
        cart_g1 = cart_g1.astype(np.float)
        cart_g2 = cart_g2.astype(np.float)
        tact_g1 = np.array(tact_g1)
        tact_g2 = np.array(tact_g2)

        return image_g1, image_g2, time_g1, time_g2, cart_g1, cart_g2, tact_g1, tact_g2

    def step(self, pose_error, rand_pose, max_toggle_num=1, max_tilt_deg=8., min_tilt_deg=6., height_est=None, height_cov=None):

        self.robot.pose_error = pose_error

        if height_est:
            l_approx = (54.5 * height_cov + height_est * 200) / (height_cov + 200)
        else:
            l_approx = 54.5

        self.robot.setSpeed(80, 40)

        if self.env_type == 'hole':
            cartesianOfGap = list(self.robot.cartesianOfGap_hole_Dict[self.target_object])
        elif self.env_type == 'wall':
            cartesianOfGap = list(self.robot.cartesianOfGap_onewall_Dict[self.target_object])
        cartesianOfGap[:3] += np.array([0, rand_pose[1], rand_pose[2]])
        targetCartesian = self.robot.tran_rotate_robot(np.array(cartesianOfGap),
                                                       pose_error[0],
                                                       pose_error[1],
                                                       -pose_error[2])
        
        while True:

            self.robot.move_cart_mm(targetCartesian)
            time.sleep(1.)

            self.tactile_module.data1.clear()
            self.tactile_module.data2.clear()
            self.tactile_module.restart1 = True
            self.tactile_module.restart2 = True
            time.sleep(.5)
            
            cart_init = self.robot.get_cart()

            inserted, Success, tilt_magnitude, time_1, time_2 = self.robot.pivoting(self.tactile_module, self.target_object, l_approx=l_approx, height_est=height_est, height_cov=height_cov, verbose=False, max_tilt_deg=max_tilt_deg, min_tilt_deg=min_tilt_deg)

            self.data1 = list(self.tactile_module.data1)[:]
            self.data2 = list(self.tactile_module.data2)[:]
            data_pushdown = self.parse_data()
            
            if Success or self.tactile_module.gtsam_graph.terminate or inserted:
                is_tilted = True
                data_rocking = None
                time_3, time_4, time_5 = None, None, None
            else:
                is_tilted = False
                self.tactile_module.data1.clear()
                self.tactile_module.data2.clear()
                Success, time_3, time_4, time_5 = self.robot.rocking(self.tactile_module, l_approx=l_approx, verbose=False)
                tilt_magnitude = min_tilt_deg/180*np.pi
                self.data1 = list(self.tactile_module.data1)[:]
                self.data2 = list(self.tactile_module.data2)[:]
                data_rocking = self.parse_data()

            if Success and not self.tactile_module.gtsam_graph.terminate:
                self.tactile_module.data1.clear()
                self.tactile_module.data2.clear()
                time_6 = self.robot.pivoting(self.tactile_module, is_tilted, tilt_magnitude, max_toggle_num=max_toggle_num, verbose=False)
                self.data1 = list(self.tactile_module.data1)[:]
                self.data2 = list(self.tactile_module.data2)[:]
                data_pivoting = self.parse_data()
            else:
                time_6 = None
                data_pivoting = None

            self.robot.move_cart_add(0., 0., 6.)
            time.sleep(0.5)

            plt.close('all')

            if self.tactile_module.gtsam_graph.error_raise == False:
                break

        if Success:
            height_est = -gtsam.Pose3.between(self.tactile_module.gtsam_graph.grp, self.tactile_module.gtsam_graph.ctl).translation()[2]
            height_cov = self.tactile_module.gtsam_graph.ctl_cov[4,4]
            ctt = self.tactile_module.gtsam_graph.ctl.translation()
            cta = self.tactile_module.gtsam_graph.ctl.rotation().column(3)
            r = np.linalg.norm(ctt[:2])
            th = np.arctan2(cta[1],cta[0])
            if ctt[:2].dot(cta[:2]) < 0:
                th += np.pi
                cta *= -1
            if cta[0]*self.robot.tilt_max_vec[1] - cta[1]*self.robot.tilt_max_vec[0] > 0:
                tilt_side = -1
            else:
                tilt_side = +1
            mode_detected = 2*self.tactile_module.gtsam_graph.mode_detected - 1
            state = np.array([r, np.cos(th), np.sin(th), tilt_side, mode_detected])
        else:
            height_est = None
            height_cov = None
            state = np.array([0,0,0,0,0])

        return self.robot.display_save_as_gif(), inserted, Success, height_est, height_cov, state, data_pushdown, data_rocking, data_pivoting, cart_init, [time_1, time_2, time_3, time_4, time_5, time_6]

    def run_OU_process(
            self,
            volatility,
            decay,
            limit,
            num_frame=800,
            ma_length=100):

        time.sleep(.5)
        self.tactile_module.data1.clear()
        self.tactile_module.data2.clear()

        cart_init = self.robot.get_cart()

        self.robot.OU_process_EGM(volatility, decay, limit, num_frame,
                                  ma_length)
        
        self.data1 = list(self.tactile_module.data1)[:]
        self.data2 = list(self.tactile_moduletactile_module.data2)[:]

        image2save_g1, image2save_g2, time_g1, time_g2, cart_g1, cart_g2 = self.select_image(
            cart_init=cart_init)

        return np.array(image2save_g1), np.array(image2save_g2), [
            time_g1, time_g2, cart_g1, cart_g2, cart_init
        ]

    def select_image(self, cart_init):

        image_g1, image_g2, time_g1, time_g2, cart_g1, cart_g2 = [], [], [], [], [], []

        for i in range(min(len(self.data1),len(self.data2))):
            image_g1.append(self.data1[i][0])
            image_g2.append(self.data2[i][0])
            time_g1.append(self.data1[i][1])
            time_g2.append(self.data2[i][1])
            cart_g1.append(self.data1[i][3])
            cart_g2.append(self.data2[i][3])

        image_g1 = np.array(image_g1)
        image_g2 = np.array(image_g2)
        time_g1 = np.array(time_g1)
        time_g2 = np.array(time_g2)
        cart_g1 = np.array(cart_g1)
        cart_g2 = np.array(cart_g2)
        cart_g1 = cart_g1.astype(np.float)
        cart_g2 = cart_g2.astype(np.float)

        i_start = np.argmin(
            np.linalg.norm(cart_g1[:, :3] - cart_init[:3], axis=1))
        i_end = len(cart_g1)

        idx_g1 = np.arange(i_start, i_end)
        idx_g2 = np.array([np.argmin(np.abs(time_g2 - time_g1[i])) for i in idx_g1])

        image2save_g1 = image_g1[idx_g1]
        image2save_g2 = image_g2[idx_g2]
        time_g1 = time_g1[idx_g1]
        time_g2 = time_g2[idx_g2]
        cart_g1 = cart_g1[idx_g1]
        cart_g2 = cart_g2[idx_g2]

        return image2save_g1, image2save_g2, time_g1, time_g2, cart_g1, cart_g2