from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
import os
import sys

# 确保能导入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 【修改】导入 DStarLite
import utils.dstarlite as dstarlite

np.set_printoptions(threshold=np.inf)

""" Deep Q Network """
class Net(nn.Module):
    def __init__(self, num_inputs=3, num_actions=4, map_xdim=9, map_ydim=10):
        super(Net, self).__init__()
        # 针对 9x10 小地图优化的网络结构
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc_input_dim = 64 * map_xdim * map_ydim
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class Agent:
    def __init__(self, policy_net, target_net):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = policy_net.to(self.device)
        self.target_network = target_net.to(self.device)
        
        self.epsilon_start = 0.8
        self.epsilon = self.epsilon_start
        self.epsilon_end = 1
        self.epsilon_count = 0
        
        self.replay_mem = deque()
        self.memory_size = 50000 
        self.start_training_info_number = 100
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 100 
        self.batch_size = 32 
        self.GAMMA = 0.95 
        self.memory = Memory(self.memory_size)
        
        self.lr_start = 0.01
        self.lr = self.lr_start
        self.lr_end = 0.01
        self.lr_count = 0
        
        self.optim = torch.optim.Adagrad(self.policy_network.parameters(), self.lr)
        self.loss_function = torch.nn.SmoothL1Loss()
        self.loss_value = []

    def choose_action(self, obs, current_place, target_place, valid_path_matrix, matrix_padding=0, use_astar=False):
        if np.random.uniform() < self.epsilon:
            state = torch.from_numpy(obs).float().unsqueeze(0)
            state = state.to(self.device)
            t_s = time.time()
            actions_value = self.policy_network.forward(state)
            t_e = time.time()
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
        else:
            t_s = time.time()
            # 【修改】使用 D* Lite 指导探索 (这里参数名仍叫 use_astar 保持兼容，但实际逻辑已改) 
            # if use_astar:
            #     action = self.find_action_dstarlite(valid_path_matrix, current_place, target_place)
            # else:
            #     action = np.random.randint(0, 4)
            action = np.random.randint(0, 4)
            action = np.array([action])
            t_e = time.time()
        t_ = t_e-t_s
        return action, t_

    def choose_action_as(self, current_place, target_place, valid_path_matrix, matrix_padding=0):
        action = self.find_action_dstarlite(valid_path_matrix, current_place, target_place)
        action = np.array([action])
        return action

    # 【修改】改用 D* Lite
    def find_action_dstarlite(self, matrix_valid_map, current_position, target_position):
        # 1-based -> 0-based
        start_node = (current_position[0]-1, current_position[1]-1)
        end_node = (target_position[0]-1, target_position[1]-1)
        
        dsl = dstarlite.DStarLite(matrix_valid_map, start_node, end_node)
        
        # 获取下一步的动作方向字符串
        action_str = dsl.get_next_action()
        
        return self.get_value(action_str)

    def get_value(self, direction):
        if direction == 'UP': return 0.
        if direction == 'RIGHT': return 1.
        if direction == 'DOWN': return 2.
        if direction == 'LEFT': return 3.
        if direction == 'STOP': return 4.

    def store_transition(self, s, a, r, s_, is_done):
        state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
        target = self.policy_network.forward(state).cpu()
        a = int(a)
        old_val = target[0][a]
        
        state = torch.from_numpy(s_).float().unsqueeze(0).to(self.device)
        target_val = self.policy_network.forward(state)
        
        if is_done == 1:
            new_val = r
        else:
            new_val = r + self.GAMMA * torch.max(target_val)
            
        error = abs(old_val - new_val).cpu().detach().numpy()
        self.memory.add(error, (np.array(s), a, r, np.array(s_), is_done))
        
        if self.memory.tree.n_entries >= self.start_training_info_number:
            self.update_network()

    def update_network(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        self.learn_step_counter += 1
        
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        batch_s, batch_a, batch_r, batch_n, batch_is_done = zip(*batch)
        
        batch_s = torch.from_numpy(np.stack(batch_s)).float().to(self.device)
        batch_r = torch.Tensor(batch_r).unsqueeze(1).to(self.device)
        batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
        batch_n = torch.from_numpy(np.stack(batch_n)).float().to(self.device)
        batch_is_done = torch.LongTensor(batch_is_done).unsqueeze(1).to(self.device)
        
        state_action_values = self.policy_network(batch_s).gather(1, batch_a)
        
        next_state_values = self.target_network(batch_n).detach().max(1)[0].unsqueeze(1)
        next_state_values_ = self.target_network(batch_n)
        next_state_values__ = self.policy_network(batch_n)
        next_state_values___ = next_state_values_.gather(1, torch.max(next_state_values__, 1)[1].unsqueeze(1))
        
        expected_state_action_values = (next_state_values___ * self.GAMMA) * (1 - batch_is_done) + batch_r
        
        loss = self.loss_function(state_action_values, expected_state_action_values)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        if loss.item() >= 0.5:
            self.loss_value.append(0.5)
        else:
            self.loss_value.append(loss.item())

    def change_learning_rate(self, times):
        if self.lr_count == times:
            print("the value of current learning rate is {}".format(self.lr))
        if self.lr_count > times:
            return
        else:
            self.lr = self.lr - (self.lr_start - self.lr_end)/times
        self.lr_count += 1

    def change_explore_rate(self, times):
        if self.epsilon_count >= times:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon = self.epsilon + (self.epsilon_end - self.epsilon_start)/times
        self.epsilon_count += 1
        if self.epsilon_count == times:
            print("exploring rate is 1.")