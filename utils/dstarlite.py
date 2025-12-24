import heapq
import numpy as np

class DStarLite:
    def __init__(self, map_matrix, s_start, s_goal):
        """
        :param map_matrix: 1为通路，0为障碍物
        :param s_start: (x, y) 0-based
        :param s_goal: (x, y) 0-based
        """
        self.map = np.array(map_matrix)
        self.x_limit = self.map.shape[1]
        self.y_limit = self.map.shape[0]
        self.s_start = s_start
        self.s_goal = s_goal
        
        self.g = {}
        self.rhs = {}
        self.U = [] 
        self.km = 0
        
        # D* Lite 反向搜索：从 Goal 到 Start
        # 初始化
        for y in range(self.y_limit):
            for x in range(self.x_limit):
                self.g[(x, y)] = float('inf')
                self.rhs[(x, y)] = float('inf')
        
        self.rhs[self.s_goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.s_goal), self.s_goal))

    def calculate_key(self, s):
        min_val = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (min_val + self.heuristic(self.s_start, s) + self.km, min_val)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbours(self, s):
        list_n = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: 
            x2, y2 = s[0] + dx, s[1] + dy
            if 0 <= x2 < self.x_limit and 0 <= y2 < self.y_limit:
                # 假设 map_matrix 中非0值为可通行
                if self.map[y2][x2] != 0: 
                    list_n.append((x2, y2))
        return list_n

    def update_vertex(self, u):
        if u != self.s_goal:
            min_rhs = float('inf')
            for s_prime in self.get_neighbours(u):
                min_rhs = min(min_rhs, self.g.get(s_prime, float('inf')) + 1)
            self.rhs[u] = min_rhs
        
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.U:
            k_old, u = self.U[0]
            k_new = self.calculate_key(u)
            
            if k_old >= self.calculate_key(self.s_start) and \
               self.rhs.get(self.s_start, float('inf')) == self.g.get(self.s_start, float('inf')):
                break
            
            heapq.heappop(self.U)
            
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self.get_neighbours(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbours(u):
                    self.update_vertex(s)

    def get_path_length(self):
        self.compute_shortest_path()
        dist = self.g.get(self.s_start, float('inf'))
        if dist == float('inf'):
            return 1000.0 
        return dist

    def get_next_action(self):
        """
        计算并返回从 Start 到 Goal 的第一步动作方向字符串。
        用于替代 A* 的 action_list[0]
        """
        self.compute_shortest_path()
        
        if self.g.get(self.s_start, float('inf')) == float('inf'):
            return "STOP"

        # 贪婪搜索：在邻居中找 g 值最小的（即离目标最近的）
        best_neighbor = None
        min_cost = float('inf')
        
        current_x, current_y = self.s_start
        
        for next_node in self.get_neighbours(self.s_start):
            # 代价 = 1 + g(next_node)
            cost = 1 + self.g.get(next_node, float('inf'))
            if cost < min_cost:
                min_cost = cost
                best_neighbor = next_node
        
        if best_neighbor:
            nx, ny = best_neighbor
            if nx > current_x: return "RIGHT"
            if nx < current_x: return "LEFT"
            if ny > current_y: return "DOWN"
            if ny < current_y: return "UP"
            
        return "STOP"