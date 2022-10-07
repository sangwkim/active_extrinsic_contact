import os
import os.path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
from TD3_utils_devel import TD3
from insertion_simulator_devel import insertion_simulator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    env_name = 'insertion_simulation_devel'
    random_seed = 0
    max_timesteps = 15 #5  # max timesteps in one episode

    episodes = 1000
    show_plot = False
    test_object = None

    success_reward = 2.
    
    state_dim = 5
    action_dim = 3
    max_error = np.array([12, 12, 15/180.*np.pi])
    max_action = max_error.copy()
    max_action[:2] *= np.cos(max_action[2]) + np.sin(max_action[2])
    max_action *= 1.05

    directory = "../TD3_model/{}".format(env_name)  # save trained models
    if not os.path.isdir(directory): raise("the model doesn't exist")
    filename = "TD3_{}_{}".format(env_name, random_seed)
    
    object_list = None
    env = insertion_simulator(object_list, max_timesteps, noise_on=True, max_error=max_error, success_reward = success_reward, isActive=True)

    policy = TD3(0, 0, max_action)
    policy.load(directory, filename)
    
    logger = []
    
    ep_reward_list = []
    ep_success_list = []
    ep_trialnum_list = []
    ep_init_error_list = []
    
    for epi in range(episodes):
        
        state_log, action_log, reward_log, done_log, state_full_log = [], [], [], [], []
    
        state, state_full, object_name = env.reset(object_name=test_object)
        ep_init_error_list.append(state_full.copy())
        if show_plot: env.plot_hole_object(state_full)
        policy.h_nc_actor = None
        action = np.zeros(3)
    
        for t in range(max_timesteps):
            
            state_log.append(state)
            state_full_log.append(state_full.copy())
    
            state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = torch.from_numpy(action/max_action).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = policy.select_action(state, action)
            
            state, state_full, reward, done = env.step(action)
            if show_plot: env.plot_hole_object(state_full, verbose=False)
            
            action_log.append(action/max_action)
            reward_log.append(reward)
            done_log.append(float(done))
    
            if done: 
                ep_reward_list.append(np.sum(reward_log))
                ep_success_list.append(np.sum(reward_log)>0)
                ep_trialnum_list.append(t+1)
                logger.append((state_log, action_log, reward_log, done_log, state_full_log))
                break
            
    ep_init_error = np.array(ep_init_error_list)
    ep_success = np.array(ep_success_list)
    ep_init_error_success = ep_init_error[ep_success]
    ep_init_error_failure = ep_init_error[~ep_success]
    
    plt.figure()
    plt.scatter(ep_init_error_success[:,0], ep_init_error_success[:,1], s=10)
    plt.scatter(ep_init_error_failure[:,0], ep_init_error_failure[:,1], s=10)
    plt.show()