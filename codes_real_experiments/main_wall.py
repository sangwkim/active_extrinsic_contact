#!/usr/bin/env python
"""
This code tests the active extrinsic contact sensing module on a single edge wall
"""

import numpy as np
import os, rospy, threading, torch, argparse
from packing_environment import Robot_motion, Packing_env
from insertion_simulator import insertion_simulator
from TD3_utils import TD3

def save_data(giffy, data_pushdown,
    data_rocking,
    data_back_n_forth,
    cart_init,
    times,
    rand_pose,
    misalign,
    step_num,
    object_name,
    save_image=False):

    if data_rocking == None:
        data_rocking = [
            np.empty((0, data_pushdown[0].shape[1], data_pushdown[0].shape[2],
                      data_pushdown[0].shape[3])),
            np.empty((0, data_pushdown[1].shape[1], data_pushdown[1].shape[2],
                      data_pushdown[1].shape[3])),
            np.empty((0)),
            np.empty((0)),
            np.empty((0, data_pushdown[4].shape[1])),
            np.empty((0, data_pushdown[5].shape[1])),
            np.empty((0, data_pushdown[6].shape[1])),
            np.empty((0, data_pushdown[7].shape[1]))
        ]
    if data_back_n_forth == None:
        data_back_n_forth = [
            np.empty((0, data_pushdown[0].shape[1], data_pushdown[0].shape[2],
                      data_pushdown[0].shape[3])),
            np.empty((0, data_pushdown[1].shape[1], data_pushdown[1].shape[2],
                      data_pushdown[1].shape[3])),
            np.empty((0)),
            np.empty((0)),
            np.empty((0, data_pushdown[4].shape[1])),
            np.empty((0, data_pushdown[5].shape[1])),
            np.empty((0, data_pushdown[6].shape[1])),
            np.empty((0, data_pushdown[7].shape[1]))
        ]

    image_g1 = np.vstack(
        (data_pushdown[0], data_rocking[0], data_back_n_forth[0]))
    image_g2 = np.vstack(
        (data_pushdown[1], data_rocking[1], data_back_n_forth[1]))
    time_g1 = np.hstack(
        (data_pushdown[2], data_rocking[2], data_back_n_forth[2]))
    time_g2 = np.hstack(
        (data_pushdown[3], data_rocking[3], data_back_n_forth[3]))
    cart_g1 = np.vstack(
        (data_pushdown[4], data_rocking[4], data_back_n_forth[4]))
    cart_g2 = np.vstack(
        (data_pushdown[5], data_rocking[5], data_back_n_forth[5]))
    tact_g1 = np.vstack(
        (data_pushdown[6], data_rocking[6], data_back_n_forth[6]))
    tact_g2 = np.vstack(
        (data_pushdown[7], data_rocking[7], data_back_n_forth[7]))

    save_folder = data_folder + '/' + object_name + '/' + str(step_num)
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    if not os.path.isdir(data_folder + '/' + object_name):
        os.mkdir(data_folder + '/' + object_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    giffy[0].save(data_folder + '/' + object_name+ '/'+f"{step_num}.gif", format='GIF',
                       append_images=giffy[1:],
                       save_all=True,
                       duration=40, loop=0)

    if save_image:
        for i in range(len(image_g1)):
            cv2.imwrite(save_folder + '/g1_' + str(i) + '.jpg',
                        image_g1[i, :, :, :])
            cv2.imwrite(save_folder + '/g2_' + str(i) + '.jpg',
                        image_g2[i, :, :, :])

    np.save(save_folder + '/' + 'time_g1_rock.npy', time_g1)
    np.save(save_folder + '/' + 'time_g2_rock.npy', time_g2)
    np.save(save_folder + '/' + 'cart_g1_rock.npy', cart_g1)
    np.save(save_folder + '/' + 'cart_g2_rock.npy', cart_g2)
    np.save(save_folder + '/' + 'tact_g1_rock.npy', tact_g1)
    np.save(save_folder + '/' + 'tact_g2_rock.npy', tact_g2)
    np.save(save_folder + '/' + 'cart_init.npy', cart_init)
    np.save(save_folder + '/' + 'rand_pose.npy', rand_pose)
    np.save(save_folder + '/' + 'misalign.npy', misalign)
    np.save(save_folder + '/' + 'timeflag.npy', times)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--stick', type=str, default='on') # turn on sitcking factor
    parser.add_argument('--dataset_name', type=str, default='single_wall')
    args = parser.parse_args()
    stick = vars(args)['stick']
    if stick == 'on': stick = True
    else: stick = False
    dataset_name = vars(args)['dataset_name']

    global data_folder
    data_folder = "/home/mcube/sangwoon/data/" + dataset_name

    min_tilt_deg = 5.
    max_tilt_deg = 6.

    env_type = 'wall'
    env = Packing_env(data_folder=data_folder, env_type=env_type, stick=stick)
    robot = Robot_motion(env_type=env_type, stick=stick)

    robot.open_gripper()
    robot.move_cart_add(0., 0., 10.)

    robot.setSpeed(600, 200)
    robot.robot_reset()

    grasp_per_object = 100
    object_list = ['circle', 'rectangle', 'hexagon', 'ellipse']

    for object_name in object_list:

        env.target_object = object_name

        for i in range(grasp_per_object):

            rand_pose = np.random.uniform((0, 0, -5), (0, 0, 10))
            print(f"height: {52+rand_pose[2]}")
            graspForce = np.random.uniform(10, 15)

            robot.pick_up_object(env.target_object, graspForce, False,
                                 rand_pose)

            state_full = np.random.uniform((0,-12,-0.5*np.pi),(0,12,0.5*np.pi))

            giffy, inserted, Success, height_est, height_cov, state, data_pushdown, data_rocking, data_back_n_forth, cart_init, times = env.step(state_full, rand_pose, max_toggle_num=4, max_tilt_deg=max_tilt_deg, min_tilt_deg=min_tilt_deg)

            thread = threading.Thread(target=save_data,
                                      args=(giffy, data_pushdown, data_rocking,
                                            data_back_n_forth, cart_init,
                                            times, rand_pose.copy(), state_full.copy(), i,
                                            object_name, False))
            thread.start()

            object_cart_info = list(robot.objectCartesianDict[object_name])
            robot.return_object(object_cart_info[0], object_cart_info[1],
                                rand_pose)

if __name__ == '__main__':
    rospy.init_node('FG_host', anonymous=True)
    main()