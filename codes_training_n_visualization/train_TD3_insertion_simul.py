import os
import os.path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import numpy as np
from TD3_utils_devel import TD3, ReplayBuffer
from insertion_simulator_devel import insertion_simulator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env_name = 'insertion_simulation_devel'
    log_interval = 50  # print avg reward after interval
    random_seed = 0
    gamma = 1.  # discount for future rewards
    batch_size = 128  # num of transitions sampled from replay buffer
    lr_a = 4e-3 #4e-4
    lr_c = 1e-2 #4e-3

    exploration_noise = 0.4
    exploration_noise_min = 0.05
    exploration_decay = 0.99975
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 4  # delayed policy updates parameter
    max_episodes = 100000  # max num of episodes
    max_timesteps = 15 #5  # max timesteps in one episode

    success_reward = 2.
    
    state_dim = 5
    action_dim = 3
    max_error = np.array([12, 12, 15/180.*np.pi])
    max_action = max_error.copy()
    max_action[:2] *= np.cos(max_action[2]) + np.sin(max_action[2])
    max_action *= 1.05

    directory = "../TD3_model/{}".format(env_name)  # save trained models
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename = "TD3_{}_{}".format(env_name, random_seed)

    object_list = None
    env = insertion_simulator(object_list, max_timesteps, noise_on=True, max_error=max_error, success_reward = success_reward, isActive=True)

    policy = TD3(lr_a, lr_c, max_action)
    replay_buffer = ReplayBuffer()
    print('buffer size', replay_buffer.size)
            
    avg_reward = 0
    ep_reward_list = []
    ep_success_list = []
    ep_trialnum_list = []
    critic_loss = []
    actor_loss = []
    success_sign = False
    
    ep_reward_list_test_circle = []
    ep_success_list_test_circle = []
    ep_trialnum_list_test_circle = []
    ep_reward_list_test_rectangle = []
    ep_success_list_test_rectangle = []
    ep_trialnum_list_test_rectangle = []
    
    max_reward = - np.inf

    for episode in range(0, max_episodes):
        #print('###########################')
        exploration_noise *= exploration_decay
        num_trial = 0
        ep_reward = 0

        state_log, action_log, reward_log, done_log, state_full_log = [], [], [], [], []

        state, state_full, object_name = env.reset()
        policy.h_nc_actor = None
        action = np.zeros(3)

        for t in range(max_timesteps):

            num_trial += 1
            state_log.append(state)
            state_full_log.append(state_full.copy())
            state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = torch.from_numpy(action/max_action).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = policy.select_action(state, action)

            action = action + np.random.normal(
                0, max(exploration_noise, exploration_noise_min),
                size=action_dim) * max_action
            action = action.clip(-max_action, max_action)
            #print(action)

            state, state_full, reward, done = env.step(action)

            action_log.append(action/max_action)
            reward_log.append(reward)
            done_log.append(float(done))

            ep_reward += reward

            # if episode is done then update policy:
            if done:
                for _ in range(max_timesteps-1-t):
                    state_log.append(np.zeros(state_dim))
                    state_full_log.append(np.zeros(3))
                    action_log.append(np.zeros(3))
                    reward_log.append(0)
                    done_log.append(1.)
                replay_buffer.add((state_log, action_log, reward_log, done_log, state_full_log))
                break

        if replay_buffer.size > batch_size:
            policy.update(replay_buffer, batch_size, gamma,
                          polyak, policy_noise, noise_clip,
                          policy_delay)
            #print('NN updated')
            actor_loss.append(policy.actor_loss.detach().cpu().numpy())
            critic_loss.append(policy.loss_Q1.detach().cpu().numpy())
            np.save(directory + '/actor_loss.npy', actor_loss)
            np.save(directory + '/critic_loss.npy', critic_loss)

        if ep_reward > 0:
            success_sign = True
        else:
            success_sign = False
            
        avg_reward += ep_reward

        ep_reward_list.append(ep_reward)
        ep_success_list.append(success_sign)
        ep_trialnum_list.append(num_trial)
        np.save(directory + '/reward_log.npy', ep_reward_list)
        np.save(directory + '/success_log.npy', ep_success_list)
        np.save(directory + '/trialnum_log.npy', ep_trialnum_list)
        
        #############################
        num_trial = 0
        ep_reward = 0

        state_log, action_log, reward_log, done_log, state_full_log = [], [], [], [], []

        state, state_full, object_name = env.reset(object_name='circle')
        policy.h_nc_actor = None
        action = np.zeros(3)

        for t in range(max_timesteps):

            num_trial += 1
            state_log.append(state)
            state_full_log.append(state_full.copy())
            state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = torch.from_numpy(action/max_action).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = policy.select_action(state, action)
            
            state, state_full, reward, done = env.step(action)

            action_log.append(action/max_action)
            reward_log.append(reward)
            done_log.append(float(done))

            ep_reward += reward
            
            if done: break
            
        if ep_reward > 0:
            success_sign = True
        else:
            success_sign = False
            
        ep_reward_list_test_circle.append(ep_reward)
        ep_success_list_test_circle.append(success_sign)
        ep_trialnum_list_test_circle.append(num_trial)
        np.save(directory + '/reward_log_test_cirle.npy', ep_reward_list_test_circle)
        np.save(directory + '/success_log_test_circle.npy', ep_success_list_test_circle)
        np.save(directory + '/trialnum_log_test_circle.npy', ep_trialnum_list_test_circle)
        ################################
        
        #############################
        num_trial = 0
        ep_reward = 0

        state_log, action_log, reward_log, done_log, state_full_log = [], [], [], [], []

        state, state_full, object_name = env.reset(object_name='rectangle')
        policy.h_nc_actor = None
        action = np.zeros(3)

        for t in range(max_timesteps):

            num_trial += 1
            state_log.append(state)
            state_full_log.append(state_full.copy())
            state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = torch.from_numpy(action/max_action).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            action = policy.select_action(state, action)
            
            state, state_full, reward, done = env.step(action)

            action_log.append(action/max_action)
            reward_log.append(reward)
            done_log.append(float(done))

            ep_reward += reward
            
            if done: break
            
        if ep_reward > 0:
            success_sign = True
        else:
            success_sign = False
            
        ep_reward_list_test_rectangle.append(ep_reward)
        ep_success_list_test_rectangle.append(success_sign)
        ep_trialnum_list_test_rectangle.append(num_trial)
        np.save(directory + '/reward_log_test_rectangle.npy', ep_reward_list_test_rectangle)
        np.save(directory + '/success_log_test_rectangle.npy', ep_success_list_test_rectangle)
        np.save(directory + '/trialnum_log_test_rectangle.npy', ep_trialnum_list_test_rectangle)
        ################################
        

        if episode % 100 == 0:
            if np.mean(ep_reward_list_test_rectangle[-500:]) > max_reward:
                policy.save(directory, filename)
                max_reward = np.mean(ep_reward_list_test_rectangle[-500:])

        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = avg_reward / log_interval
            print("Episode: {}\tAverage Reward: {:1.2f}".format(
                episode, avg_reward))
            avg_reward = 0
            plt.plot(actor_loss)
            plt.savefig(directory + '/actor_loss.png')
            plt.clf()
            plt.plot(critic_loss)
            plt.savefig(directory + '/critic_loss.png')
            plt.clf()
            plt.plot(np.convolve(ep_reward_list, np.ones(500)/500, mode='valid'))
            plt.plot(np.convolve(ep_reward_list_test_circle, np.ones(500)/500, mode='valid'))
            plt.plot(np.convolve(ep_reward_list_test_rectangle, np.ones(500)/500, mode='valid'))
            plt.legend(['train','test_circle','test_rectangle'])
            plt.savefig(directory + '/reward_log.png')
            plt.clf()
            plt.plot(np.convolve(ep_success_list, np.ones(500)/500, mode='valid'))
            plt.plot(np.convolve(ep_success_list_test_circle, np.ones(500)/500, mode='valid'))
            plt.plot(np.convolve(ep_success_list_test_rectangle, np.ones(500)/500, mode='valid'))
            plt.legend(['train','test_circle','test_rectangle'])
            plt.savefig(directory + '/success_log.png')
            plt.clf()
            plt.plot(np.convolve(ep_trialnum_list, np.ones(500)/500, mode='valid'))
            plt.plot(np.convolve(ep_trialnum_list_test_circle, np.ones(500)/500, mode='valid'))
            plt.plot(np.convolve(ep_trialnum_list_test_rectangle, np.ones(500)/500, mode='valid'))
            plt.legend(['train','test_circle','test_rectangle'])
            plt.savefig(directory + '/trialnum_log.png')
            plt.clf()