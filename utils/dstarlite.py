import heapq
import numpy as np

class DStarLite:
    def __init__(self, map_matrix, s_start, s_goal):
        """
        :param map_matrix: 0为障碍物，1为通路
        :param s_start: (x, y)
        :param s_goal: (x, y)
        """
        self.map = np.array(map_matrix)
        self.x_limit = self.map.shape[1]
        self.y_limit = self.map.shape[0]
        self.s_start = s_start
        self.s_goal = s_goal
        
        self.g = {}
        self.rhs = {}
        self.U = [] # 优先队列
        self.km = 0
        
        # 初始化 g 和 rhs 为无穷大
        for x in range(self.x_limit):
            for y in range(self.y_limit):
                self.g[(x, y)] = float('inf')
                self.rhs[(x, y)] = float('inf')
        
        self.rhs[self.s_goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.s_goal), self.s_goal))

    def calculate_key(self, s):
        min_val = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (min_val + self.heuristic(self.s_start, s) + self.km, min_val)

    def heuristic(self, a, b):
        # 使用曼哈顿距离或欧几里得距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbours(self, s):
        list_n = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 4邻域
            x2, y2 = s[0] + dx, s[1] + dy
            if 0 <= x2 < self.x_limit and 0 <= y2 < self.y_limit:
                # 只有当该位置不是障碍物时才视为邻居
                # 注意：D* Lite通常处理边权变化，这里简化为节点是否可达
                if self.map[y2][x2] == 1: 
                    list_n.append((x2, y2))
        return list_n

    def update_vertex(self, u):
        if u != self.s_goal:
            min_rhs = float('inf')
            for s_prime in self.get_neighbours(u):
                # 代价默认为1
                min_rhs = min(min_rhs, self.g.get(s_prime, float('inf')) + 1)
            self.rhs[u] = min_rhs
        
        # 从堆中移除 u (如果存在) - Python heapq 不支持直接移除，通常通过 lazy remove 或重新 push 处理
        # 这里简化处理：直接 push 新值，pop 时检查有效性
        
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.U:
            k_old, u = self.U[0]
            k_new = self.calculate_key(u)
            
            if self.calculate_key(u) >= self.calculate_key(self.s_start) and \
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
        """
        运行 D* Lite 并返回从 start 到 goal 的路径长度。
        如果无解返回大数值。
        """
        self.compute_shortest_path()
        if self.g.get(self.s_start, float('inf')) == float('inf'):
            return 1000.0 # 无解
        return self.g[self.s_start]