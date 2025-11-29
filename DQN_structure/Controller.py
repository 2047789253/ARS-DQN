import matplotlib
matplotlib.use('Agg')  # 【关键】强制使用无界面后端，防止段错误
import matplotlib.pyplot as plt
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import sys

# --- 【新增代码 Start】 ---
# 将上级目录加入路径，以便能导入 utils 包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import astar # 导入 A* 算法模块
# --- 【新增代码 End】 ---

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

        '''--------【A* 开关】--------'''
        # True = 使用 A* 
        # False = 纯 DQN
        self.use_astar_guidance = False  

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

        # --- 【新增代码 Start】 ---
        # 奖励塑造参数
        self.use_shaping = True   # 开关：是否开启奖励塑造
        self.shaping_factor = 1.0 # 缩放因子：这一项越大，老师的引导作用越强
        self.gamma = self.agent.GAMMA # 必须和 DQN 的 gamma 一致 (通常是 0.95)
        # --- 【新增代码 End】 ---

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
                # 保存日志 (既然都存图了，顺便把日志也存一下，防止中途断电)
                self.save_log()          
            # check whether determination condition meets
            # if self.check_determination(self.reward_acc):
            #     break
        self.save_neural_network(auto=False)
        self.save_log()
        self.draw_picture(self.lifelong_reward, p_title="Cumulative Reward", p_xlabel="training episodes", p_ylabel="cumulative reward", p_color="g")
        self.draw_picture(self.agent.loss_value, p_title="Loss Value", p_xlabel="Training steps", p_ylabel="Loss value", p_color="k")
        plt.show()

    # --- 【新增函数】 ---
    def get_astar_potential(self, current_place, target_place, valid_matrix):
        """
        利用 A* 计算当前位置到目标的距离的负数作为势能
        Phi(s) = -Distance(s, target)
        """
        # A* 需要的是 (x, y) 坐标，且代码中通常是 0-based索引，而 current_place 可能是 1-based
        # 根据你之前的 Explorer.py 代码，path_founder 需要 (x, y)
        # 注意：这里假设 current_place 是 [x, y] 格式
        start_node = (current_place[0] - 1, current_place[1] - 1)
        end_node = (target_place[0] - 1, target_place[1] - 1)
        
        # 调用 A* 寻路
        path_founder = astar.FindPathAstar(valid_matrix, start_node, end_node)
        find_target, path_list, _, _ = path_founder.run_astar_method()
        
        if not find_target:
            # 如果 A* 发现根本走不到终点（被堵死了），给一个非常大的惩罚势能
            return -30.0 
        
        # 势能 = 负的路径长度 (距离越短，势能越大)
        return -1.0 * len(path_list)

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
        
        # --- 【新增代码 Start】 ---
        # 在做决定前，先算出“我现在离终点有多远（势能）”
        if self.use_shaping:
            current_potential = self.get_astar_potential(this_veh_cp, this_veh_tp, valid_path_matrix)
            veh_obj.last_potential = current_potential # 存在小车对象里，等会算奖励用
        # --- 【新增代码 End】 ---

        """get action"""
        veh_obj.obs_current = obs
        veh_obj.obs_valid_matrix = valid_path_matrix
        action_l, t_ = self.agent.choose_action(obs, current_place=this_veh_cp, target_place=this_veh_tp,
                                                valid_path_matrix=valid_path_matrix,
                                                use_astar=self.use_astar_guidance)   # state should be formatted as array
        action = action_l[0]
        action = self.check_action(this_veh_cp, this_veh_tp, valid_path_matrix, action, this_veh)
        """record info"""
        self.time_list.append(t_)
        veh_obj.action.append(action)
        self.action_length_record += 1
        return action

    def check_action(self, this_veh_cp, veh_tp, valid_path_matrix, action, this_veh):
        # 1. 预测 AGV 执行动作后的新位置 (veh_cp)
        veh_cp = this_veh_cp.copy()
        if action == 0:
            veh_cp[1] -= 1
        if action == 1:
            veh_cp[0] += 1
        if action == 2:
            veh_cp[1] += 1
        if action == 3:
            veh_cp[0] -= 1

        # 2. 判断动作是否非法 (越界 或 撞墙)
        is_invalid = False
        
        # 检查是否越出地图边界
        if veh_cp[0] <= 0 or veh_cp[1] <= 0 or veh_cp[0] > len(valid_path_matrix[0]) or veh_cp[1] > len(valid_path_matrix):
            is_invalid = True
        # 检查是否撞到障碍物 (valid_path_matrix 中值为 0 的地方)
        elif valid_path_matrix[veh_cp[1]-1][veh_cp[0]-1] == 0:
            is_invalid = True

        # 3. 根据开关决定是否修正
        if is_invalid:
            # 只有在开启 A* 指导时，才帮它修正
            if self.use_astar_guidance:
                import random
                action = random.randint(0, 3)
            #action = self.agent.choose_action_as(current_place=this_veh_cp, target_place=veh_tp, valid_path_matrix=valid_path_matrix)[0]
            else:
                # 纯 DQN 模式：不修正，直接返回错误的 action。
                # 这样环境中的 Explorer 就会执行这个动作，检测到碰撞，返回 -1 奖励。
                pass 
        
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        self.reward_acc += reward
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return

        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_obj = veh
                break
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.create_state(all_info, this_veh)

        total_reward = reward # 默认就是原始奖励
        
        try:
            # 1. 计算当前（动作后）的势能
            # 注意：这里传入的 matrix 应该是处理过的，不过在单AGV下不影响
            next_potential = self.get_astar_potential(this_veh_cp, this_veh_tp, valid_path_matrix)
            
            # 2. 获取上一步（动作前）的势能
            # 如果是第一步，没有 last_potential，就假设它和当前一样，避免第一步产生巨大的奖励波动
            if not hasattr(veh_obj, 'last_potential'):
                veh_obj.last_potential = next_potential 
            prev_potential = veh_obj.last_potential
            
            # 3. 计算 PBRS 附加奖励: F = gamma * Phi(next) - Phi(prev)
            if is_end:
                # 如果到达终点，不需要打折，直接算差值
                shaping_reward = next_potential - prev_potential
            else:
                # === 【关键修改点：堵住漏洞】 ===
                # 检查势能是否相等（意味着原地不动，或撞墙未移动）
                if next_potential == prev_potential:
                    shaping_reward = 0.0  # 强制归零，防止 +0.5 的刷分奖励
                else:
                    # 只有真正移动了，才计算折扣奖励
                    shaping_reward = (self.agent.GAMMA * next_potential) - prev_potential
            
            # 4. 更新势能记录，供下一步使用
            veh_obj.last_potential = next_potential
            
            # 5. 组合总奖励 
            # 建议系数设为 0.5 或 1.0，不要太大，以免掩盖原本的任务奖励（如到达终点的+10）
            total_reward = reward + (0.5 * shaping_reward)
            
        except Exception as e:
            # 如果 A* 报错，降级为普通奖励，保证程序不崩
            print(f"Shaping Error: {e}")
            total_reward = reward

        # 【关键】将修改后的 total_reward 存入对象
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
        """neural network uses state to make decision"""
        """astar algorithm uses current_place, target_place, valid_path_matrix to make decision"""
        return state, current_place, target_place, valid_path_matrix

    def create_path_matrix(self, layout, veh_loaded, current_place, target_place, occupied_place):
        # valid_path_matrix, forbidden_path_matrix
        valid_path, valid_path_one_line = [], []
        forbidden_path, forbidden_path_one_line = [], []

        # 制作原始的valid_path和forbidden_path
        for map_one_line in layout:
            for one_cell in map_one_line:
                if one_cell == 0:
                    valid_path_one_line.append(1.)
                    forbidden_path_one_line.append(0.)
                elif one_cell == 1:
                    if veh_loaded == 0:
                        valid_path_one_line.append(1.)
                        forbidden_path_one_line.append(0.)
                    else:
                        valid_path_one_line.append(0.)
                        forbidden_path_one_line.append(1.)
                elif one_cell == 2:
                    valid_path_one_line.append(0.)
                    forbidden_path_one_line.append(1.)
                else:
                    print("create_path_matrix:wrong matrix")
            valid_path.append(valid_path_one_line)
            valid_path_one_line = []
            forbidden_path.append(forbidden_path_one_line)
            forbidden_path_one_line = []

        valid_path_matrix = np.array(valid_path)
        forbidden_path_matrix = np.array(forbidden_path)

        valid_path_matrix_o = valid_path_matrix.copy()
        forbidden_path_matrix_o = forbidden_path_matrix.copy()

        """调整valid_path_matrix和forbidden_path_matrix"""
        # 根据current_position和target_position调整
        valid_path_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0  # current_place_array样式[[x],[y]]
        forbidden_path_matrix[current_place[1] - 1][current_place[0] - 1] = 0.0
        valid_path_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0  # current_place_array样式[[x],[y]]
        forbidden_path_matrix[target_place[1] - 1][target_place[0] - 1] = 0.0

        # useless?
        valid_path_matrix_o[current_place[1] - 1][current_place[0] - 1] = 2.0  # current_place_array样式[[x],[y]]
        valid_path_matrix_o[target_place[1] - 1][target_place[0] - 1] = 3.0  # current_place_array样式[[x],[y]]

        # 其他车辆对道路的占用
        if occupied_place:
            for o_place in occupied_place:
                valid_path_matrix[o_place[1] - 1][o_place[0] - 1] = 0.0  # current_place_array样式[[x],[y]]
                forbidden_path_matrix[o_place[1] - 1][o_place[0] - 1] = 1.0
                valid_path_matrix_o[o_place[1] - 1][o_place[0] - 1] = 4.0  # current_place_array样式[[x],[y]]

        """"无效"""
        # 赋予更高权重
        basic_matrix = self.create_basic_matrix(layout)
        basic_matrix_array = np.array(basic_matrix)

        #
        current_p_x, current_p_y = current_place[0] - 1, current_place[1] - 1
        up, right, down, left = (0, -1), (1, 0), (0, 1), (-1, 0)
        four_dict = [up, right, down, left]
        direction_length = 3

        if valid_path_matrix[current_p_y][current_p_x] != 0:
            basic_matrix_array[current_p_y][current_p_x] = 1.0  # current_place_array样式[[x],[y]]
            for one_direction in four_dict:
                pos = [current_p_x+one_direction[0], current_p_y+one_direction[1]]
                pos_further = [current_p_x+one_direction[0]*2, current_p_y+one_direction[1]*2]
                if pos[0] < 0 or pos[1] < 0 or pos[0] >= len(valid_path_matrix[0]) or pos[1] >= len(valid_path_matrix):
                    continue
                else:
                    if valid_path_matrix[pos[1]][pos[0]] != 0:
                        basic_matrix_array[pos[1]][pos[0]] = 1.0  # current_place_array样式[[x],[y]]
                        # 当前位置有效，探索更远一格位置
                    elif valid_path_matrix[pos[1]][pos[0]] == 0:
                        basic_matrix_array[pos[1]][pos[0]] = -1.0  # current_place_array样式[[x],[y]]

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

        # 构建current_position_matrix
        current_position_matrix = basic_matrix_array.copy()
        current_position_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0

        # 构建target_position_matrix
        target_position_matrix = basic_matrix_array.copy()
        target_position_matrix[target_place[1]-1][target_place[0]-1] = 1.0

        occupied_position_matrix = basic_matrix_array.copy()
        if occupied_place:
            for occupied_ in occupied_place:
                occupied_position_matrix[occupied_[1]-1][occupied_[0]-1] = 1.0

        return current_position_matrix, target_position_matrix, occupied_position_matrix

    def draw_picture(self, p_data, p_title="NoTitle", p_xlabel="xlabel", p_ylabel="ylabel", p_color="g"):
        import os
        import matplotlib.pyplot as plt # 确保导入了 plt

        plt.figure(figsize=(16, 9))  # 调整长宽比
        plt.title(p_title)
        plt.xlabel(p_xlabel)
        plt.ylabel(p_ylabel)
        plt.plot(p_data, color=p_color)
        plt.tight_layout()  # 去除白边
        
        # ---【修改开始】使用绝对路径保存 ---
        # 1. 获取当前脚本所在的绝对目录 (.../DQN_structure)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 拼接保存文件夹路径 (.../DQN_structure/network_picture)
        save_dir = os.path.join(base_dir, 'network_picture')
        
        # 3. 如果文件夹不存在，强制创建！
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 4. 拼接完整的文件名 (注意加上 .png 后缀，防止有些系统不识别)
        # 如果 p_title 只有名字没有后缀，最好加上后缀
        if not p_title.endswith('.png'):
            file_name = p_title + ".png"
        else:
            file_name = p_title
            
        save_path = os.path.join(save_dir, file_name)
        
        plt.savefig(save_path, dpi=300)  # 使用生成的绝对路径保存
        # ---【修改结束】---
        
        plt.close() # 画完图后关闭，释放内存，防止循环画图导致内存溢出

    def check_determination(self, reward_accu):
        # check whether the determination meets
        if reward_accu >= self.max_value-1:
            self.acc_max += 1
            self.max_value_times = self.max_value_times+1
            self.lr_start_decay = True  # lr start to decay
            if self.interupt_times > self.interupt_num:
                self.interupt_times = 0
        else:
            if self.interupt_times >= self.interupt_num:
                self.max_value_times = 0
            else:
                pass
            self.interupt_times += 1

            # self.interupt_num = 1
            # self.interupt_times = 0

        if self.max_value_times == self.duration_times:
            return True
        else:
            return False

    def save_neural_network(self, auto=False):
        # 获取 Controller.py 所在的文件夹路径 (即 .../DQN_structure)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接出保存目录 (.../DQN_structure/network_picture)
        save_dir = os.path.join(base_dir, 'network_picture')

        # 确保目录存在，如果不存在则创建（防止其他路径错误）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if auto:
            print("neural network auto-save")
            # 使用 join 拼接完整路径
            torch.save(self.agent.policy_network, os.path.join(save_dir, 'RMFS_DQN_policy_net_auto.pt'))
            torch.save(self.agent.target_network, os.path.join(save_dir, 'RMFS_DQN_target_net_auto.pt'))
        else:
            torch.save(self.agent.policy_network, os.path.join(save_dir, 'RMFS_DQN_policy_net.pt'))
            torch.save(self.agent.target_network, os.path.join(save_dir, 'RMFS_DQN_target_net.pt'))
    def save_log(self):
        import os
        # 1. 获取当前脚本所在的绝对路径 (.../multiAGV_Env/DQN_structure)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 拼接出保存目录 (.../multiAGV_Env/DQN_structure/network_picture)
        save_dir = os.path.join(base_dir, 'network_picture')

        # 3. 关键步骤：如果文件夹不存在，自动创建它！
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 4. 拼接完整的文件路径
        file_path = os.path.join(save_dir, 'logs.txt')

        # 5. 写入文件
        with open(file_path, 'w') as f:
            for one_log in self.logs:
                f.write(one_log)
                f.write("\r\n")


class VehObj:
    """veh object"""
    def __init__(self, this_veh):
        self.veh_name = this_veh
        """"逐个检查"""
        # self.obs_list = []
        self.obs_current = 0
        self.obs_next = 0
        self.obs_forbidden_matrix = 0
        self.obs_valid_matrix = 0
        self.action = []
        self.reward = 0
        self.is_end = False
        self.last_state = 0
        self.last_state_store = False

