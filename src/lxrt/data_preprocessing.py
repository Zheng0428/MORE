import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

'''
Usage:

traj_dataset = MiniGridDataset(dataset_path='minigrid.pkl')

traj_data_loader = DataLoader(traj_dataset,
						batch_size=batch_size,
						shuffle=True,
						pin_memory=True,
						drop_last=True)

data_iter = iter(traj_data_loader)
'''

def extend_tensor(tensor1, tensor2):
    if tensor2 is None:
        print('tensor 2 None')
    if tensor1 is None:
        return tensor2
    else:
        return torch.cat((tensor1, tensor2), 0)

class MiniGridDataset(Dataset):
    def __init__(self, dataset_path, max_length=1000, reward_with_timestep=False):

        self.reward_with_timestep = reward_with_timestep
        self.max_length = max_length
        self.observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.instructions = []
        self.episode_idxs = None
        self.episode_lengths = None
        self.rtg = None

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        for env in self.trajectories:
            print(env)

            self.observations = extend_tensor(self.observations, torch.as_tensor(self.trajectories[env]['observations']))
            self.instructions.extend(self.trajectories[env]['instructions'])
            self.actions = extend_tensor(self.actions,torch.as_tensor(self.trajectories[env]['actions']))
            self.rewards = extend_tensor(self.rewards, torch.as_tensor(self.trajectories[env]['rewards']))
            self.dones = extend_tensor(self.dones, torch.as_tensor(self.trajectories[env]['dones']))
            episode_idxs, episode_lengths = self.get_episode_infos(env)
            self.episode_idxs = extend_tensor(self.episode_idxs, episode_idxs)
            self.episode_lengths = extend_tensor(self.episode_lengths, episode_lengths)
            self.rtg = extend_tensor(self.rtg, self.get_rtg(env))

        '''
        Discounts to go -> option of having 1/0 for all return-to-go for an episode,
        or have return-to-go decrease with each timestep
        '''

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        '''
        returns the rest of the given episode from the indexed time step
        '''
        episode_length = self.episode_lengths[index]
        episode_end_idx = self.episode_lengths[index]
        padding_length = self.max_length - episode_length

        states = self.observations[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
        states = torch.cat([states,
                            torch.zeros(([padding_length] + list(states.shape[1:])),
                            dtype=states.dtype)],
                            dim=0)

        actions = self.actions[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
        actions = torch.cat([actions,
                            torch.zeros(([padding_length] + list(actions.shape[1:])),
                            dtype=actions.dtype)],
                            dim=0)

        rtg = self.rtg[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
        rtg = torch.cat([rtg,
                            torch.zeros(([padding_length] + list(rtg.shape[1:])),
                            dtype=states.dtype)],
                            dim=0)

        instructions = self.instructions[episode_end_idx + 1 - episode_length:episode_end_idx + 1]
        instructions.extend([0 for i in range(padding_length)])

        timesteps = torch.arange(start=0, end=self.max_length, step=1)

        traj_mask = torch.cat([torch.ones(episode_length, dtype=torch.long),
                                torch.zeros(padding_length, dtype=torch.long)],
                                dim=0)
        return  timesteps, states, actions, rtg, traj_mask #instructions


    def get_non_zero_idx(self, env):
        end_idxs_lst = torch.nonzero(torch.as_tensor(self.trajectories[env]['dones']))
        if len(end_idxs_lst) == 0:
            end_idxs_lst = torch.as_tensor([self.max_length * (i+1) - 1 for i in range(3)])
        return end_idxs_lst

    def get_episode_infos(self, env):
        episode_idx = None
        episode_lengths = None
        end_idxs_lst = self.get_non_zero_idx(env)
        for val in end_idxs_lst:
            if episode_idx is None:
                episode_length = val + 1
            else:
                episode_length = val + 1 - episode_idx.shape[0]
            episode_idx = extend_tensor(episode_idx, torch.as_tensor([self.observations.shape[0] + val - len(self.trajectories[env]['dones']) for i in range(episode_length)]))
            episode_lengths = extend_tensor(episode_lengths, torch.as_tensor([episode_length for i in range(episode_length)]))
        return episode_idx, episode_lengths

    def get_rtg(self, env):
        rewards = self.trajectories[env]['rewards']
        rtg = None
        end_idxs_lst = self.get_non_zero_idx(env)
        for val in end_idxs_lst:
            if rtg is None:
                episode_length = val + 1
            else:
                episode_length = val + 1 - rtg.shape[0]
            curr_episode = [rewards[val] for i in range(episode_length)]
            if self.reward_with_timestep:
                reward = rewards[val]
                curr_episode[:] = [reward / episode_length * i for i in range(curr_episode)]
            rtg = extend_tensor(rtg, torch.as_tensor(curr_episode))
        return rtg
