import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import sys

# 将上级目录加入路径，以便能导入 utils 包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 【修改】导入 DStarLite
from utils import dstarlite 

from DQN_structure.DQN import Net as Net
from DQN_structure.DQN import Agent as Agent

class DQNAgentController:
    """
        a link between environment and algorithm
        """
    # state_number要修改
    # marix_padding要删除
    def __init__(self, rmfs_scene, map_xdim, map_ydim, max_task, control_mode=1, state_number=4):
        print("start simulation with DQN algorithm")
        print("map_xdim:", map_xdim, "map_ydim:", map_ydim, "state_number:", state_number)

        '''received parameters'''
        self.control_mode = control_mode
        self.state_number = state_number

        '''--------【A*/D* Lite 开关】--------'''
        self.use_shaping = True   # 开关：是否开启奖励塑造
        self.use_astar_guidance = True  
        self.shaping_factor = 1.0 # 缩放因子
        self.gamma = self.agent.GAMMA if hasattr(self, 'agent') else 0.95
        

        '''get RMFS object'''
        self.rmfs_model = rmfs_scene

        '''create/load neural network_picture'''
        policy_net, target_net = None, None
        if self.control_mode == "train_NN":
            print("create NN")
            policy_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
            target_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
        elif self.control_mode == "use_NN":
            print("load NN")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'network_picture', 'RMFS_DQN_policy_net.pt')
            policy_net = torch.load(model_path)
            model_path1 = os.path.join(current_dir, 'network_picture', 'RMFS_DQN_target_net.pt')
            target_net = torch.load(model_path1)

        '''create Agent object'''
        self.agent = Agent(policy_net, target_net)
        # 重新绑定gamma，确保一致
        self.gamma = self.agent.GAMMA 

        '''training parameters'''
        self.simulation_times = 5000
        self.max_value = max_task*3
        self.max_value_times = 0
        self.duration_times = 60
        self.interupt_num = 0
        self.interupt_times = 0
        #############################################
        self.acc_max = 0
        self.acc_max_val = 0  # 40

        self.lr_start_decay = False

        self.lifelong_reward = []
        self.action_length_record = 0
        self.time_list = []

        """":parameter"""
        self.reward_acc = 0
        self.veh_group = []
        self.logs = []

    def self_init(self):
        self.reward_acc = 0
        self.veh_group = []
        self.action_length_record = 0
        self.time_list = []

    def model_run(self):  # mainloop for training/running
        import pygame # 引入 pygame
        print("model is controlled by neural network")
        for i_episode in range(self.simulation_times):
            self.self_init()
            self.rmfs_model.init()
            # print("i_episode", i_episode)
            """"transfer the controller to the model"""
            """the model runs once"""
            running_time = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self)
            pygame.quit()
            self.lifelong_reward.append(self.reward_acc)
            log = 'i_episode {}\treward_accu: {} \taction_length: {}'.format(i_episode, self.reward_acc, self.action_length_record)
            self.logs.append(log)
            print(log)
            # print("self.time_list", np.array(self.time_list).sum())
            # print("running_time", running_time)
            # 改变探索率
            if self.lr_start_decay:
                self.agent.change_learning_rate(times=200)
            # 改变lr
                self.agent.change_explore_rate(times=200)

            if i_episode % 100 == 0:
                self.save_neural_network(auto=True)
            if i_episode > 0 and i_episode % 50 == 0:
                print(f"Drawing pictures at episode {i_episode}...")
                # 绘制奖励曲线
                self.draw_picture(self.lifelong_reward, 
                                p_title="Cumulative Reward", 
                                p_xlabel="training episodes", 
                                p_ylabel="cumulative reward", 
                                p_color="g")
                # 绘制 Loss 曲线
                self.draw_picture(self.agent.loss_value, 
                                p_title="Loss Value", 
                                p_xlabel="Training steps", 
                                p_ylabel="Loss value", 
                                p_color="k")
                # 保存日志
                self.save_log()          

        self.save_neural_network(auto=False)
        self.save_log()
        self.draw_picture(self.lifelong_reward, p_title="Cumulative Reward", p_xlabel="training episodes", p_ylabel="cumulative reward", p_color="g")
        self.draw_picture(self.agent.loss_value, p_title="Loss Value", p_xlabel="Training steps", p_ylabel="Loss value", p_color="k")
        plt.show()

    # --- 【修改函数】使用 D* Lite 计算势能 ---
    def get_dstarlite_potential(self, current_place, target_place, valid_matrix):
        """
        利用 D* Lite 计算当前位置到目标的距离的负数作为势能
        Phi(s) = -Distance(s, target)
        """
        # 坐标转换：1-based -> 0-based
        start_node = (current_place[0] - 1, current_place[1] - 1)
        end_node = (target_place[0] - 1, target_place[1] - 1)
        
        # 实例化 D* Lite
        # 在动态环境中，valid_matrix 会因为动态障碍物的移动而改变
        dsl = dstarlite.DStarLite(valid_matrix, start_node, end_node)
        
        # 获取路径长度
        path_len = dsl.get_path_length()
        
        # 如果不可达，给予大惩罚
        if path_len >= 1000:
            return -30.0 
        
        return -1.0 * path_len

    def choose_action(self, all_info, this_veh):  # all_infor=[layout  , current_place, target_place
        """build a VehObj to store information"""
        veh_found = False
        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_found = True
                veh_obj = veh
                break
        if not veh_found:
            veh_obj = VehObj(this_veh)
            self.veh_group.append(veh_obj)

        """get observation and other info"""
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.create_state(all_info, this_veh)
        
        # --- 【计算势能】 ---
        if self.use_shaping:
            current_potential = self.get_dstarlite_potential(this_veh_cp, this_veh_tp, valid_path_matrix)
            veh_obj.last_potential = current_potential # 暂存，供store_info计算奖励差分
        # -----------------

        """get action"""
        veh_obj.obs_current = obs
        veh_obj.obs_valid_matrix = valid_path_matrix
        # 注意：Agent 的 choose_action 内部如果也调用了寻路逻辑，可能需要对应修改，
        # 但通常 DQN 的 choose_action 是基于神经网络输出的，这里的 use_astar 参数可能只是备选策略
        action_l, t_ = self.agent.choose_action(obs, current_place=this_veh_cp, target_place=this_veh_tp,
                                                valid_path_matrix=valid_path_matrix,
                                                use_astar=self.use_astar_guidance)   
        action = action_l[0]
        action = self.check_action(this_veh_cp, this_veh_tp, valid_path_matrix, action, this_veh)
        """record info"""
        self.time_list.append(t_)
        veh_obj.action.append(action)
        self.action_length_record += 1
        return action

    def check_action(self, this_veh_cp, veh_tp, valid_path_matrix, action, this_veh):
        # 可以在此添加防撞墙逻辑
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        self.reward_acc += reward
        if self.control_mode == "use_NN":
            return

        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_obj = veh
                break
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.create_state(all_info, this_veh)

        total_reward = reward 
        
        try:
            # 1. 计算当前（动作后）的势能 (Next Potential)
            next_potential = self.get_dstarlite_potential(this_veh_cp, this_veh_tp, valid_path_matrix)
            
            # 2. 获取上一步（动作前）的势能 (Prev Potential)
            if not hasattr(veh_obj, 'last_potential'):
                veh_obj.last_potential = next_potential 
            prev_potential = veh_obj.last_potential
            
            # 3. 计算势能差分奖励: F = gamma * Phi(next) - Phi(prev)
            if is_end:
                shaping_reward = next_potential - prev_potential
            else:
                # 避免原地不动的刷分
                if next_potential == prev_potential:
                    shaping_reward = 0.0 
                else:
                    shaping_reward = (self.gamma * next_potential) - prev_potential
            
            # 4. 更新势能
            veh_obj.last_potential = next_potential
            
            # 5. 叠加奖励
            total_reward = reward + (self.shaping_factor * shaping_reward)
            
        except Exception as e:
            print(f"Shaping Error: {e}")
            total_reward = reward

        veh_obj.obs_next, veh_obj.reward = obs, total_reward

        is_done = 1 if is_end else 0
        self.agent.store_transition(veh_obj.obs_current, veh_obj.action[-1], veh_obj.reward, veh_obj.obs_next, is_done)

    def create_state(self, all_info, this_veh):
        layout = all_info[0]
        occupied_place = []
        occupied_target = []
        current_place = 0
        target_place = 0
        veh_loaded = False
        """obtain information about current_place, target_place, occupied_place, occupied_target"""
        for i in range(1, len(all_info)):
            one_veh = all_info[i]
            veh_name_, current_place_, target_place_, veh_loaded_ = one_veh[0], one_veh[1], one_veh[2], one_veh[3]
            if veh_name_ == this_veh:  # target_veh
                current_place, target_place, veh_loaded = current_place_, target_place_, veh_loaded_
            else:
                occupied_place.append(current_place_)
                occupied_target.append(target_place_)
        """"format observations"""
        valid_path_matrix, forbidden_path_matrix, basic_matrix_array = \
            self.create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place)
        current_position_matrix, target_position_matrix, occupied_position_matrix\
            = self.create_position_matrix(layout, current_place, target_place, occupied_place, occupied_target)

        state = np.array((current_position_matrix, target_position_matrix, valid_path_matrix))
        return state, current_place, target_place, valid_path_matrix

    def create_path_matrix(self, layout, veh_loaded, current_place, target_place, occupied_place):
        # valid_path_matrix, forbidden_path_matrix
        valid_path, valid_path_one_line = [], []
        forbidden_path, forbidden_path_one_line = [], []

        # 制作原始的valid_path和forbidden_path
        # 注意：这里需要正确处理 Scene.py 中设置的动态障碍物 (代号 2)
        for map_one_line in layout:
            for one_cell in map_one_line:
                if one_cell == 0: # 空地
                    valid_path_one_line.append(1.)
                    forbidden_path_one_line.append(0.)
                elif one_cell == 1: # 货架
                    if veh_loaded == 0: # 空载可穿过
                        valid_path_one_line.append(1.)
                        forbidden_path_one_line.append(0.)
                    else: # 负载不可穿过
                        valid_path_one_line.append(0.)
                        forbidden_path_one_line.append(1.)
                elif one_cell == 2: # 障碍物 或 动态障碍物
                    valid_path_one_line.append(0.)
                    forbidden_path_one_line.append(1.)
                else:
                    # 其他未知代码，默认为障碍
                    valid_path_one_line.append(0.)
                    forbidden_path_one_line.append(1.)

            valid_path.append(valid_path_one_line)
            valid_path_one_line = []
            forbidden_path.append(forbidden_path_one_line)
            forbidden_path_one_line = []

        valid_path_matrix = np.array(valid_path)
        forbidden_path_matrix = np.array(forbidden_path)

        # 调整当前位置和目标位置为可通行 (防止AGV出生在障碍点导致的逻辑错误)
        valid_path_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0  
        forbidden_path_matrix[current_place[1] - 1][current_place[0] - 1] = 0.0
        valid_path_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0  
        forbidden_path_matrix[target_place[1] - 1][target_place[0] - 1] = 0.0

        # 其他车辆对道路的占用
        if occupied_place:
            for o_place in occupied_place:
                valid_path_matrix[o_place[1] - 1][o_place[0] - 1] = 0.0  
                forbidden_path_matrix[o_place[1] - 1][o_place[0] - 1] = 1.0
                
        basic_matrix = self.create_basic_matrix(layout)
        basic_matrix_array = np.array(basic_matrix)

        # 这里原本有一些对周围环境探测的逻辑，保留原样
        current_p_x, current_p_y = current_place[0] - 1, current_place[1] - 1
        up, right, down, left = (0, -1), (1, 0), (0, 1), (-1, 0)
        four_dict = [up, right, down, left]

        if valid_path_matrix[current_p_y][current_p_x] != 0:
            basic_matrix_array[current_p_y][current_p_x] = 1.0
            for one_direction in four_dict:
                pos = [current_p_x+one_direction[0], current_p_y+one_direction[1]]
                if pos[0] < 0 or pos[1] < 0 or pos[0] >= len(valid_path_matrix[0]) or pos[1] >= len(valid_path_matrix):
                    continue
                else:
                    if valid_path_matrix[pos[1]][pos[0]] != 0:
                        basic_matrix_array[pos[1]][pos[0]] = 1.0
                    elif valid_path_matrix[pos[1]][pos[0]] == 0:
                        basic_matrix_array[pos[1]][pos[0]] = -1.0

        return valid_path_matrix, forbidden_path_matrix, basic_matrix_array

    def create_basic_matrix(self, layout):
        basic_matrix, basic_matrix_one_line = [], []
        for map_one_line in layout:
            for one_cell in map_one_line:
                basic_matrix_one_line.append(0.)
            basic_matrix.append(basic_matrix_one_line)
            basic_matrix_one_line = []
        return basic_matrix

    def create_position_matrix(self, layout, current_place, target_place, occupied_place, occupied_target):
        basic_matrix = self.create_basic_matrix(layout)
        basic_matrix_array = np.array(basic_matrix)

        current_position_matrix = basic_matrix_array.copy()
        current_position_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0

        target_position_matrix = basic_matrix_array.copy()
        target_position_matrix[target_place[1]-1][target_place[0]-1] = 1.0

        occupied_position_matrix = basic_matrix_array.copy()
        if occupied_place:
            for occupied_ in occupied_place:
                occupied_position_matrix[occupied_[1]-1][occupied_[0]-1] = 1.0

        return current_position_matrix, target_position_matrix, occupied_position_matrix

    def draw_picture(self, p_data, p_title="NoTitle", p_xlabel="xlabel", p_ylabel="ylabel", p_color="g"):
        import os
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 9))
        plt.title(p_title)
        plt.xlabel(p_xlabel)
        plt.ylabel(p_ylabel)
        plt.plot(p_data, color=p_color)
        plt.tight_layout()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, 'network_picture')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not p_title.endswith('.png'):
            file_name = p_title + ".png"
        else:
            file_name = p_title
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def check_determination(self, reward_accu):
        if reward_accu >= self.max_value-1:
            self.acc_max += 1
            self.max_value_times = self.max_value_times+1
            self.lr_start_decay = True
            if self.interupt_times > self.interupt_num:
                self.interupt_times = 0
        else:
            if self.interupt_times >= self.interupt_num:
                self.max_value_times = 0
            else:
                pass
            self.interupt_times += 1

        if self.max_value_times == self.duration_times:
            return True
        else:
            return False

    def save_neural_network(self, auto=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, 'network_picture')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if auto:
            print("neural network auto-save")
            torch.save(self.agent.policy_network, os.path.join(save_dir, 'RMFS_DQN_policy_net_auto.pt'))
            torch.save(self.agent.target_network, os.path.join(save_dir, 'RMFS_DQN_target_net_auto.pt'))
        else:
            torch.save(self.agent.policy_network, os.path.join(save_dir, 'RMFS_DQN_policy_net.pt'))
            torch.save(self.agent.target_network, os.path.join(save_dir, 'RMFS_DQN_target_net.pt'))
            
    def save_log(self):
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, 'network_picture')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, 'logs.txt')
        with open(file_path, 'w') as f:
            for one_log in self.logs:
                f.write(one_log)
                f.write("\r\n")

class VehObj:
    """veh object"""
    def __init__(self, this_veh):
        self.veh_name = this_veh
        self.obs_current = 0
        self.obs_next = 0
        self.obs_valid_matrix = 0
        self.action = []
        self.reward = 0
        self.is_end = False
        self.last_state = 0
        self.last_state_store = False
        # 新增 last_potential 属性，在Controller运行时会被赋值