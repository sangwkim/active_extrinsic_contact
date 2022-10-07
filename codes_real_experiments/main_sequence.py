#!/usr/bin/env python
"""
This code is the main code that demonstrate the peg-in-hole insertion with trained policy.
"""

import numpy as np
import os, rospy, threading, torch
from packing_environment import Robot_motion, Packing_env
from insertion_simulator import insertion_simulator
from TD3_utils import TD3

dataset_name = "insertion_test"
data_folder = "../data/" + dataset_name

def save_data(giffy, data_pushdown,
              data_rocking,
              data_pivoting,
              cart_init,
              times,
              rand_pose,
              misalign,
              epi_num,
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
    if data_pivoting == None:
        data_pivoting = [
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
        (data_pushdown[0], data_rocking[0], data_pivoting[0]))
    image_g2 = np.vstack(
        (data_pushdown[1], data_rocking[1], data_pivoting[1]))
    time_g1 = np.hstack(
        (data_pushdown[2], data_rocking[2], data_pivoting[2]))
    time_g2 = np.hstack(
        (data_pushdown[3], data_rocking[3], data_pivoting[3]))
    cart_g1 = np.vstack(
        (data_pushdown[4], data_rocking[4], data_pivoting[4]))
    cart_g2 = np.vstack(
        (data_pushdown[5], data_rocking[5], data_pivoting[5]))
    tact_g1 = np.vstack(
        (data_pushdown[6], data_rocking[6], data_pivoting[6]))
    tact_g2 = np.vstack(
        (data_pushdown[7], data_rocking[7], data_pivoting[7]))

    save_folder = data_folder + '/' + object_name + '/' + str(epi_num) + '/' + str(step_num)
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    if not os.path.isdir(data_folder + '/' + object_name):
        os.mkdir(data_folder + '/' + object_name)
    if not os.path.isdir(data_folder + '/' + object_name + '/' + str(epi_num)):
        os.mkdir(data_folder + '/' + object_name + '/' + str(epi_num))
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    giffy[0].save(data_folder + '/' + object_name + '/' + str(epi_num) + '/'+f"{step_num}.gif", format='GIF',
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

    episode_per_object = 50
    max_step = 15
    object_list = ['circle_tight', 'ellipse_tight', 'hexagon_tight', 'circle', 'rectangle', 'hexagon', 'ellipse']
    max_error = np.array([12, 12, 15/180.*np.pi])

    max_action = max_error.copy()
    max_action[:2] *= np.cos(max_action[2]) + np.sin(max_action[2])
    max_action *= 1.05

    env_name = 'insertion_simulation'
    random_seed = 0
    directory = "../TD3_model/{}".format(env_name)
    if not os.path.isdir(directory): raise("the model doesn't exist")
    filename = "TD3_{}_{}".format(env_name, random_seed)
    policy = TD3(0, 0, max_action)
    policy.load(directory, filename)

    env = Packing_env(data_folder=data_folder)
    robot = Robot_motion()

    robot.open_gripper()
    robot.move_cart_add(0., 0., 10.)
    robot.setSpeed(600, 200)
    robot.robot_reset()

    for object_name in object_list:

        env.target_object = object_name
        simulator = insertion_simulator([object_name], max_step, isActive=True, noise_on=False, max_error=max_error)
        data_list = []

        for i in range(episode_per_object):

            rand_pose = np.random.uniform((0, 0, 0), (0, 0, 7.5))
            graspForce = np.random.uniform(10, 15)
            height_gt = rand_pose[2] + 52.

            robot.pick_up_object(env.target_object, graspForce, False,
                                 rand_pose)

            num_trial = 0
            ep_reward = 0
            Success_log, state_simul_log, state_log, action_log, state_full_log, height_log = [], [], [], [], [], []

            state_simul, state_full, _ = simulator.reset() # samples the initial misalignment
            giffy, inserted, Success, height_est, height_cov, state, data_pushdown, data_rocking, data_pivoting, cart_init, times = env.step(state_full, rand_pose, None, None)
            height_log.append(np.array([height_est, height_cov]))
            policy.h_nc_actor = None

            action = np.zeros(3)
            
            thread = threading.Thread(target=save_data,
                                      args=(giffy, data_pushdown, data_rocking,
                                            data_pivoting, cart_init,
                                            times, rand_pose.copy(), state_full.copy(), i, 0,
                                            object_name, False))
            thread.start()

            for t in range(max_step):

                state_simul_log.append(state_simul.copy())
                state_log.append(state.copy())
                state_full_log.append(state_full.copy())
                state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                action = torch.from_numpy(action/max_action).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                action = policy.select_action(state, action)

                print(f"state: {state}")
                print(f"state_gt: {state_simul}")
                print(f"state_full: {state_full}")
                print(f"action: {action}")

                state_simul, state_full, _, _ = simulator.step(action)

                giffy, inserted, Success, height_est, height_cov, state, data_pushdown, data_rocking, data_pivoting, cart_init, times = env.step(state_full, rand_pose, height_est, height_cov)
                height_log.append(np.array([height_est, height_cov]))
                action_log.append(action)
                Success_log.append(float(inserted))

                thread = threading.Thread(target=save_data,
                                      args=(giffy, data_pushdown, data_rocking,
                                            data_pivoting, cart_init,
                                            times, rand_pose.copy(), state_full.copy(), i, t+1,
                                            object_name, False))
                thread.start()

                if inserted: break

            data_list.append((Success_log, state_simul_log, state_log, action_log, state_full_log, height_log, height_gt))

            object_cart_info = list(robot.objectCartesianDict[object_name])
            robot.return_object(object_cart_info[0], object_cart_info[1],
                                rand_pose)
            np.save(f'{data_folder}/{object_name}/{i}/epi_log.npy', (Success_log, state_simul_log, state_log, action_log, state_full_log, height_log, height_gt))
                        
        robot.open_gripper()
        np.save(f'{data_folder}/{object_name}/combined_log.npy', data_list)

if __name__ == '__main__':
    rospy.init_node('FG_host', anonymous=True)
    main()