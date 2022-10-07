#!/usr/bin/env python
"""
This code collects the training data for the tactile module.
It grasp the objects with random pose and force then wiggle the gripper to get the training sequence.
The tactile module takes the tactile images as input and outputs the gripper-object relative SE(3) pose (x,y,z,yaw,pitch,roll).
"""

import os, shutil, scipy, rospy, cv2, os.path, random, time, threading
import numpy as np
from packing_environment import Robot_motion, Packing_env

dataset_name = "tactile_module_training_set"
data_folder = "../data/" + dataset_name

grasp_per_object = 100
object_list = ['hexagon', 'rectangle', 'hexagon', 'rectangle']


def save_data(image_g1, image_g2, time_cart_rock, folder_num, object_name):

    for i in range(len(image_g1)):
        save_folder = data_folder + '/' + object_name + '/' + str(folder_num)
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
        if not os.path.isdir(data_folder + '/' + object_name):
            os.mkdir(data_folder + '/' + object_name)
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        cv2.imwrite(save_folder + '/g1_' + str(i) + '.jpg',
                    image_g1[i, :, :, :])
        cv2.imwrite(save_folder + '/g2_' + str(i) + '.jpg',
                    image_g2[i, :, :, :])

    np.save(save_folder + '/' + 'time_g1_rock.npy', time_cart_rock[0])
    np.save(save_folder + '/' + 'time_g2_rock.npy', time_cart_rock[1])
    np.save(save_folder + '/' + 'cart_g1_rock.npy', time_cart_rock[2])
    np.save(save_folder + '/' + 'cart_g2_rock.npy', time_cart_rock[3])
    np.save(save_folder + '/' + 'cart_init.npy', time_cart_rock[4])


def main():

    mode = 'gtcollect'
    env = Packing_env(mode=mode)
    robot = Robot_motion(mode=mode)

    robot.setSpeed(600, 200)
    robot.move_cart_add(0., 0., 100.)
    robot.robot_reset()
    robot.open_gripper()

    for object_name in object_list:
        env.target_object = object_name

        for i in range(grasp_per_object):
            rand_pose = np.array([
                np.random.uniform(-4, 4),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-5, 10),
                np.random.uniform(-1.3 / 180 * np.pi, 1.3 / 180 * np.pi),
                np.random.uniform(-4 / 180 * np.pi, 4 / 180 * np.pi),
                np.random.uniform(-1.3 / 180 * np.pi, 1.3 / 180 * np.pi)
            ])
            graspForce = np.random.uniform(10, 20)
            if i == 0: inposition = False
            else: inposition = True
            robot.grasp_object(env.target_object, graspForce, inposition,
                               rand_pose)
            image_g1, image_g2, time_cart = env.run_OU_process(
                volatility=np.array([1e-2, 2.5e-2, 2.5e-2, 1e-4, 3.5e-4,
                                     7e-4]),
                decay=np.array([0.999, 0.999, 0.999, 0.999, 0.999, 0.999]),
                limit=np.array([
                    .25, .5, .5, .2 / 180 * np.pi, .6 / 180 * np.pi,
                    1.2 / 180 * np.pi
                ]),
                num_frame=800)

            robot.open_gripper()
            thread = threading.Thread(target=save_data,
                                      args=(image_g1, image_g2, time_cart, i,
                                            object_name))
            thread.start()

        robot.move_cart_add(0., 0., 100.)

if __name__ == '__main__':
    rospy.init_node('FG_host', anonymous=True)
    main()