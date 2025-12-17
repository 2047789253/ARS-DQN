import sys
import pygame
from utils.utils import Direction as Dir
from utils.utils import ColorBox as ColorBox
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

sys.path.append(os.path.dirname(__file__))


class Scene:
    def __init__(self, layout, explorer_group):
        """"all parameters about drawing scene"""
        self.layout = layout
        self.control_pattern = ""
        self.clock = None
        self.FPS = 30 
        self.x_width = self.layout.scene_x_width
        self.y_width = self.layout.scene_y_width
        # other parameters related to scene
        self.border_width = 30
        self.line_width = 2
        self.color_box = ColorBox()
        # size of main interface
        self.cell_width = 36
        self.interface_width = self.x_width * self.cell_width - (self.x_width - 1) * self.line_width
        self.interface_height = self.y_width * self.cell_width - (self.y_width - 1) * self.line_width
        self.interface_start_x = self.border_width
        self.interface_start_y = self.border_width
        # size of sidebar
        self.sidebar_width = 200
        self.sidebar_height = self.interface_height + 2 * self.border_width
        self.sidebar_start_x = self.interface_width + 2 * self.border_width
        self.sidebar_start_y = 0
        # size of screen
        self.screen_width = self.interface_width + 2 * self.border_width + self.sidebar_width
        self.screen_height = self.interface_height + 2 * self.border_width
        # parameters related to AGV
        self.AGV_icon_scale = 0.9
        self.explorer_group = explorer_group
        if len(self.explorer_group) == 0:
            print("WARNING: the number of veh is zero")
            return
        # all surfaces
        self.screen = None
        self.interface = None
        self.sidebar = None
        """"all parameters about training"""
        self.dir = Dir()
        self.action_number = self.dir.action_num()
        self.smart_controller = None

        # 【修改】动态障碍物参数
        # 障碍物在 x=4 (中间通道) 上运动
        self.dyn_obs_x = 4 
        self.dyn_obs_y_min = 0
        self.dyn_obs_y_max = 8
        self.dyn_obs_current_pos = [4, 0] # 记录当前位置用于清除

    def _load_fonts(self):
        """每轮游戏开始时重新加载字体"""
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Times New Roman", 30)
        self.font_agv = pygame.font.SysFont("Times New Roman", 15)
        self.font_author = pygame.font.SysFont("Times New Roman", 15)
        self.font_scale = pygame.font.SysFont("Times New Roman", 12)

    def init(self):
        self.layout.init()
        for explorer in self.explorer_group:
            explorer.init()
        # 重置障碍物位置
        self.dyn_obs_current_pos = [4, 0]

    # 【修改】根据时间步 t 更新障碍物位置
    def update_dynamic_obstacles(self, t):
        layout_grid = self.layout.layout_original
        
        # 1. 清除上一时刻的障碍物 (设回 0: 空地)
        old_x, old_y = self.dyn_obs_current_pos
        if 0 <= old_y < self.y_width and 0 <= old_x < self.x_width:
             layout_grid[old_y][old_x] = 0 

        # 2. 计算新位置 (Pos = f(t))
        # 使用往复运动 (Ping-Pong) 逻辑
        # 周期长度 T = 2 * (max - min)
        # 例如 0->8->0，距离是8，来回是16步一个周期
        span = self.dyn_obs_y_max - self.dyn_obs_y_min
        cycle = 2 * span
        
        t_mod = t % cycle
        if t_mod <= span:
            # 正向移动: 0, 1, ..., 8
            new_y = self.dyn_obs_y_min + t_mod
        else:
            # 反向移动: 8, 7, ..., 0
            # 这里的逻辑是: max - (t_mod - max) = 2*max - t_mod
            new_y = self.dyn_obs_y_max - (t_mod - span)
        
        new_x = self.dyn_obs_x
        self.dyn_obs_current_pos = [new_x, new_y]

        # 3. 在地图上标记新障碍物
        # 使用 '2' (picking station代码) 或自定义代码，确保 Controller 能识别为障碍
        if 0 <= new_y < self.y_width and 0 <= new_x < self.x_width:
            layout_grid[new_y][new_x] = 2 

    def run_game(self, control_pattern="manual", smart_controller=None):
        pygame.init()
        self._load_fonts()
        pygame.display.set_caption('multiAGV World')
        self.control_pattern = control_pattern
        self.explorer_group[0].create_explorer()
        
        # screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(self.color_box.GRAY_Color)
        self.refresh_screen(self.explorer_group[0])

        self.clock = pygame.time.Clock()

        if self.control_pattern == "manual":
            self.run_mode_manual()
        if self.control_pattern in ["A_star", "auto"]:
            self.run_mode_auto()
        if self.control_pattern == "intelligent":
            self.run_mode_smart(smart_controller)

    def run_mode_smart(self, smart_controller):
        self.smart_controller = smart_controller
        running_time = 0
        while True:
            running_time += 1 # 这就是系统时间步 t
            self.clock.tick(self.FPS)

            # 【关键修改】每一帧都更新障碍物，完全跟随 t
            self.update_dynamic_obstacles(running_time)

            """standard code: exit game"""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.create_interface()  # update interface
            self.create_sidebar()  # update sidebar
            for explorer in self.explorer_group:
                if not explorer.has_created:
                    break
                if explorer.all_assigned:
                    self.patch_agv_icon(explorer)
                    continue
                if explorer.Working:
                    if explorer.time_counting == explorer.Working_Time[explorer.working_type]:
                        explorer.continue_working()
                        if self.layout.task_finished:
                            return running_time
                        if explorer.all_assigned:
                            self.patch_agv_icon(explorer)
                            continue
                    else:
                        explorer.time_counting += 1
                        self.patch_agv_icon(explorer)
                        continue

                """collect infos and make decision"""
                # 此时 layout 已经包含了时刻 t 的动态障碍物
                all_info = self.create_info()
                
                # Agent 根据当前状态 (包含动态障碍物) 做决策
                input_action = self.smart_controller.choose_action(all_info, explorer.explorer_name)

                """execute action"""
                reward, is_end, all_info_ = explorer.execute_action(input_action, all_info)
                self.patch_agv_icon(explorer)

                if self.layout.task_finished:
                    is_end = True
                self.smart_controller.store_info(all_info_, reward, is_end, explorer.explorer_name)
                if is_end:
                    print("running_time", running_time)
                    return running_time
            
            flags = self.check_new_veh()
            if flags != 0:
                explorer = self.explorer_group[flags]
                self.patch_agv_icon(explorer)

            self.screen.blit(self.interface, (self.interface_start_x, self.interface_start_y))
            self.screen.blit(self.sidebar, (self.sidebar_start_x, self.sidebar_start_y))

            pygame.display.flip()

    # ... (其余方法 run_mode_manual, run_mode_auto, refresh_screen, patch_agv_icon 保持不变) ...

    def run_mode_manual(self):
        print("Control by manual mode")
        if len(self.explorer_group) > 1:
            print("WARNING: manual mode can only control one AGV")
            return
        while True:
            input_action = ""
            self.clock.tick(self.FPS)
            explorer = self.explorer_group[0]
            if explorer.Working:
                if explorer.time_counting == explorer.Working_Time[explorer.working_type]:
                    explorer.continue_working()
                    self.refresh_screen(explorer)
                else:
                    explorer.time_counting += 1
                    continue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        input_action = self.dir.value_str[0]
                    if event.key == pygame.K_DOWN:
                        input_action = self.dir.value_str[2]
                    if event.key == pygame.K_LEFT:
                        input_action = self.dir.value_str[3]
                    if event.key == pygame.K_RIGHT:
                        input_action = self.dir.value_str[1]
            if input_action != "":
                explorer.execute_action(input_action)
                self.refresh_screen(explorer)

    def run_mode_auto(self):
        print("Control by auto mode")
        while True:
            input_action = ""
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.create_interface()
            self.create_sidebar()

            for explorer in self.explorer_group:
                if not explorer.has_created:
                    break
                if explorer.all_assigned:
                    self.patch_agv_icon(explorer)
                    continue
                if explorer.Working:
                    if explorer.time_counting == explorer.Working_Time[explorer.working_type]:
                        explorer.continue_working()
                        if self.layout.task_finished:
                            return
                        if explorer.all_assigned:
                            self.patch_agv_icon(explorer)
                            continue
                    else:
                        explorer.time_counting += 1
                        self.patch_agv_icon(explorer)
                        continue

                if not self.layout.task_finished:
                    input_action = explorer.find_path_astar(self.explorer_group)
                if input_action != "":
                    explorer.execute_action(input_action)
                    self.patch_agv_icon(explorer)

            flags = self.check_new_veh()
            if flags != 0:
                explorer = self.explorer_group[flags]
                self.patch_agv_icon(explorer)

            self.screen.blit(self.interface, (self.interface_start_x, self.interface_start_y))
            self.screen.blit(self.sidebar, (self.sidebar_start_x, self.sidebar_start_y))
            pygame.display.flip()

    def refresh_screen(self, explorer):
        self.create_interface()
        self.patch_agv_icon(explorer)
        self.create_sidebar()
        self.screen.blit(self.interface, (self.interface_start_x, self.interface_start_y))
        self.screen.blit(self.sidebar, (self.sidebar_start_x, self.sidebar_start_y))
        pygame.display.flip()

    def patch_agv_icon(self, explore_group):
        if not isinstance(explore_group, list):
            explore_group = [explore_group]
        for explore in explore_group:
            agv_image = pygame.image.load(explore.icon_path)
            agv_image = pygame.transform.scale(agv_image, (
            self.cell_width * self.AGV_icon_scale, self.cell_width * self.AGV_icon_scale))
            agv_position = self.position_rectify(explore.current_place[0], explore.current_place[1])
            self.interface.blit(agv_image, agv_position)

    def create_interface(self):
        # interface
        self.interface = pygame.Surface((self.interface_width, self.interface_height), flags=pygame.HWSURFACE)
        self.interface.fill(color=self.color_box.WHITE_COLOR)
        # draw blocks
        for y_dim in range(self.y_width):
            for x_dim in range(self.x_width):
                pygame.draw.rect(self.interface, self.color_box.BLACK_COLOR, (
                x_dim * (self.cell_width - self.line_width), y_dim * (self.cell_width - self.line_width),
                self.cell_width, self.cell_width), self.line_width)
                
                if (x_dim + 1, y_dim + 1) in self.layout.picking_station_list:
                    self.draw_block(self.interface, self.color_box.BLACK_COLOR, x_dim, y_dim)
                if (x_dim + 1, y_dim + 1) in self.layout.storage_station_list:
                    if self.layout.layout[y_dim][x_dim] == 1.8:
                        self.draw_block(self.interface, self.color_box.PINK_COLOR, x_dim, y_dim)
                    elif self.layout.layout[y_dim][x_dim] == 1.3:
                        self.draw_block(self.interface, self.color_box.GREEN_COLOR, x_dim, y_dim)
                    else:
                        self.draw_block(self.interface, self.color_box.RED_COLOR, x_dim, y_dim)
                if x_dim == 0:
                    self.draw_scale(self.screen, float(y_dim), "y")
                if y_dim == 0:
                    self.draw_scale(self.screen, float(x_dim), "x")
        
        # 【修改】绘制动态障碍物 (蓝色)
        d_x, d_y = self.dyn_obs_current_pos
        if 0 <= d_x < self.x_width and 0 <= d_y < self.y_width:
             self.draw_block(self.interface, (0, 0, 255), d_x, d_y)

    def create_sidebar(self):
        # sidebar
        self.sidebar = pygame.Surface((self.sidebar_width, self.sidebar_height), flags=pygame.HWSURFACE)
        self.sidebar.fill(color=self.color_box.GRAY_Color)
        # title
        title = self.font_title.render(str("RMFS World"), True, self.color_box.BLACK_COLOR)
        title_rect = title.get_rect()
        self.sidebar.blit(title, (self.sidebar_width / 2 - title_rect.width / 2, self.sidebar_height / 15))
        # title
        font_agv = pygame.font.SysFont("Times New Roman", 15)
        t_l = self.font_agv.render(str("target_: " + str(self.explorer_group[0].target_position)), True,
                              self.color_box.BLACK_COLOR)
        c_l = self.font_agv.render(str("current: " + str(self.explorer_group[0].current_place)), True,
                              self.color_box.BLACK_COLOR)
        l_l = self.font_agv.render(str("last___: " + str(self.explorer_group[0].last_place)), True,
                              self.color_box.BLACK_COLOR)
        act = self.font_agv.render(str("action_: " + str(self.explorer_group[0].action_str)), True,
                              self.color_box.BLACK_COLOR)
        r_s = self.font_agv.render(str("state__: " + str(self.explorer_group[0].running_state)), True,
                              self.color_box.BLACK_COLOR)
        self.sidebar.blit(t_l, (20, self.sidebar_height / 3))
        self.sidebar.blit(c_l, (20, self.sidebar_height / 3 + 20))
        self.sidebar.blit(l_l, (20, self.sidebar_height / 3 + 40))
        self.sidebar.blit(act, (20, self.sidebar_height / 3 + 60))
        self.sidebar.blit(r_s, (20, self.sidebar_height / 3 + 80))
        # title
        font_author = pygame.font.SysFont("Times New Roman", 15)
        author_detail = self.font_author.render(str("Author: Stone"), True, self.color_box.BLACK_COLOR)
        self.sidebar.blit(author_detail, (20, 5 * self.sidebar_height / 6))

    def draw_scale(self, screen, value, axis):
        font = pygame.font.SysFont("Times New Roman", 12)
        rect = self.font_scale.render(str(value + 1), True, self.color_box.BLACK_COLOR)
        if axis == "x":
            screen.blit(rect, (value * (
                        self.cell_width - self.line_width) + self.interface_start_x - rect.get_width() / 2 + self.cell_width / 2,
                               self.border_width / 3))
        elif axis == "y":
            screen.blit(rect, (self.border_width / 4, value * (
                        self.cell_width - self.line_width) + self.interface_start_y - rect.get_height() / 2 + self.cell_width / 2))

    def draw_block(self, interface, color, x_dim, y_dim):
        pygame.draw.rect(interface, color, (x_dim * (self.cell_width - self.line_width) + self.line_width,
                                            y_dim * (self.cell_width - self.line_width) + self.line_width,
                                            self.cell_width - self.line_width,
                                            self.cell_width - self.line_width))

    def position_rectify(self, x_dim, y_dim):
        x_position = (x_dim - 1) * (self.cell_width - self.line_width) + self.line_width
        y_position = (y_dim - 1) * (self.cell_width - self.line_width) + self.line_width
        position = (x_position, y_position)
        return position

    def create_info(self):
        """infos for reinforcement learning"""
        layout = self.layout.layout_original
        all_info = [layout]
        for explorer in self.explorer_group:
            if explorer.has_created:
                one_explorer = [explorer.explorer_name, explorer.current_place, explorer.target_position,
                                explorer.loaded]
                all_info.append(one_explorer)
            else:
                break
        return all_info

    def check_new_veh(self):
        init_pos_occupy = False
        flags = 0
        for explore_num in range(len(self.explorer_group)):
            if self.explorer_group[explore_num].has_created:
                if self.explorer_group[explore_num].current_place == [1, 1]:
                    init_pos_occupy = True
            else:
                if not init_pos_occupy:
                    self.explorer_group[explore_num].create_explorer()
                    flags = explore_num
                    break
        return flags