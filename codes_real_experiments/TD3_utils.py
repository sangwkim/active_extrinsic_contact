import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self,
                 state_dim=5,
                 LSTM_layers=3, #3,
                 LSTM_nodes=64, #64,
                 FC_dim = 32, #32,
                 #drop_p=0.,
                 action_dim=3,
                 max_action=np.array([7,7,10/180.*np.pi])):
        super(Actor, self).__init__()
        
        self.action_dim = action_dim
        self.max_action = torch.from_numpy(max_action).type(torch.FloatTensor).to(device)

        self.LSTM = nn.LSTM(
            input_size=state_dim+action_dim,
            hidden_size=LSTM_nodes,
            num_layers=LSTM_layers,
            batch_first=True)

        self.fc1 = nn.Linear(LSTM_nodes, FC_dim)
        self.fc2 = nn.Linear(FC_dim, action_dim)
        #self.drop = nn.Dropout(drop_p)

    def forward(self, states, actions, h_nc=None):
        
        states_ = states.clone()
        actions_ = actions.clone()
        
        states_[:,:,0] /= self.max_action[0]
        
        if states_.shape[1] == actions_.shape[1] + 1:
            actions_ = torch.cat((torch.zeros(actions_.shape[0],1,self.action_dim).to(device),actions_),1)

        self.LSTM.flatten_parameters()

        RNN_out, h_nc = self.LSTM(torch.cat((states_, actions_),2), h_nc)
        #x = self.drop(F.relu(self.fc1(RNN_out)))
        x = F.relu(self.fc1(RNN_out[:,[-1],:]))
        x = torch.tanh(self.fc2(x)) * self.max_action

        return x, h_nc

class Critic(nn.Module):
    def __init__(self,
                 state_dim=3,
                 action_dim=3):
        super(Critic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim+action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, state, action):

        x = self.model(torch.cat((state, action), 2))

        return x[:,0,0]

class ReplayBuffer:
    def __init__(self, max_size=1e7):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, sequence):
        self.size += 1
        # sequence is tuple of (states, actions, rewards, dones, state_fulls)
        self.buffer.append(sequence)

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, dones, state_fulls  = [], [], [], [], []

        for i in indexes:
            s, a, r, d, sf = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))
            state_fulls.append(np.array(sf, copy=False))

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(state_fulls)

class TD3:
    def __init__(self, lr_a, lr_c, max_action):

        self.actor = Actor(max_action=max_action).to(device)
        self.actor_target = Actor(max_action=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)

        self.critic_1 = Critic().to(device)
        self.critic_1_target = Critic().to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_c)

        self.critic_2 = Critic().to(device)
        self.critic_2_target = Critic().to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_c)
        
        self.iter = 0
        
        self.h_nc_actor = None

        self.max_action = torch.from_numpy(max_action).type(torch.FloatTensor).to(device)

    def select_action(self, state, prev_action):

        state = state.to(device)
        prev_action = prev_action.to(device)
        action, self.h_nc_actor = self.actor(state, prev_action, self.h_nc_actor)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size, gamma, polyak,
               policy_noise, noise_clip, policy_delay):

        states, actions, rewards, dones, state_fulls = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        state_fulls = torch.FloatTensor(state_fulls).to(device)

        self.loss_Q1, self.loss_Q2 = 0, 0
        count = 0

        for t in range(1,states.shape[1]):

            noise = torch.FloatTensor(actions[:,[t],:].to('cpu')).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = self.actor_target(states[:,:t+1,:], actions[:,:t,:])[0] / self.max_action + noise
            next_action = next_action.clamp(-1, 1)

            #target_Q1 = self.critic_1_target(state_fulls[:,[t],:] / self.max_action, next_action)
            #target_Q2 = self.critic_1_target(state_fulls[:,[t],:] / self.max_action, next_action)
            target_Q1 = self.critic_1_target(state_fulls[:,[0],:] / self.max_action, next_action)
            target_Q2 = self.critic_1_target(state_fulls[:,[0],:] / self.max_action, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards[:,t-1] + ((1 - dones[:,t-1]) * gamma * target_Q).detach()

            #current_Q1 = self.critic_1(state_fulls[:,[t-1],:] / self.max_action, actions[:,[t-1],:])
            #current_Q2 = self.critic_2(state_fulls[:,[t-1],:] / self.max_action, actions[:,[t-1],:])
            current_Q1 = self.critic_1(state_fulls[:,[0],:] / self.max_action, actions[:,[t-1],:])
            current_Q2 = self.critic_2(state_fulls[:,[0],:] / self.max_action, actions[:,[t-1],:])

            if t == 1:
                self.loss_Q1 += F.mse_loss(current_Q1, target_Q, reduction='sum')
                self.loss_Q2 += F.mse_loss(current_Q2, target_Q, reduction='sum')
                count += states.shape[0]
            else:
                self.loss_Q1 += F.mse_loss(current_Q1 * (1-dones[:,t-2]), target_Q * (1-dones[:,t-2]), reduction='sum')
                self.loss_Q2 += F.mse_loss(current_Q2 * (1-dones[:,t-2]), target_Q * (1-dones[:,t-2]), reduction='sum')
                count += torch.sum(1-dones[:,t-2])

        self.loss_Q1 /= count
        self.loss_Q2 /= count
        #print('Q1 loss', self.loss_Q1)
        #print('Q2 loss', loss_Q2)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        self.loss_Q1.backward()
        self.loss_Q2.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.iter % policy_delay == 0:

            self.actor_loss = 0
            count = 0

            for t in range(1,states.shape[1]+1):
                #critic_out = self.critic_1(state_fulls[:,[t-1],:] / self.max_action, self.actor(states[:,:t,:], actions[:,:t-1,:])[0] / self.max_action)
                critic_out = self.critic_1(state_fulls[:,[0],:] / self.max_action, self.actor(states[:,:t,:], actions[:,:t-1,:])[0] / self.max_action)
                if t == 1:
                    count += states.shape[0]
                if t > 1:
                    critic_out *= 1 - dones[:,t-2]
                    count += torch.sum(1-dones[:,t-2])
                
                self.actor_loss += - critic_out.sum()

            self.actor_loss /= count
            #print('actor loss', self.actor_loss)
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak averaging update:
            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) +
                                        ((1 - polyak) * param.data))

            for param, target_param in zip(
                    self.critic_1.parameters(),
                    self.critic_1_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) +
                                        ((1 - polyak) * param.data))

            for param, target_param in zip(
                    self.critic_2.parameters(),
                    self.critic_2_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) +
                                        ((1 - polyak) * param.data))
                
        self.iter += 1

    def save(self, directory, name):
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(),
                   '%s/%s_actor_target.pth' % (directory, name))

        torch.save(self.critic_1.state_dict(),
                   '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(),
                   '%s/%s_critic_1_target.pth' % (directory, name))

        torch.save(self.critic_2.state_dict(),
                   '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(),
                   '%s/%s_critic_2_target.pth' % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(
            torch.load('%s/%s_crtic_1.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(
            torch.load('%s/%s_critic_1_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(
            torch.load('%s/%s_crtic_2.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(
            torch.load('%s/%s_critic_2_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

    def load_actor(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

    def print_param(self):
        for i, param in enumerate(self.actor.parameters()):
            print(param.requires_grad)

        for i, param in enumerate(self.actor_target.parameters()):
            print(param.requires_grad)