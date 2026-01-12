import numpy as np
import math
import heapq

class PathPlanner:
    def __init__(self, env):
        self.env = env
        self.resolution = 0.5
        
        #  预先计算所有动作及其对应的移动代价 ---
        # 格式: (dx, dy, dz, cost)
        self.motions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    # 这里直接算出移动成本 (1.0, 1.414, 或 1.732)
                    cost = math.sqrt(dx**2 + dy**2 + dz**2) * self.resolution
                    self.motions.append((dx, dy, dz, cost))

    def plan(self, start_pos, goal_pos):
        """
        Optimized A* Algorithm implementation.
        """
        start_node = self._pos_to_index(start_pos)
        goal_node = self._pos_to_index(goal_pos)

        # 优先队列 (f_score, h_score, current_node)
        # 将 h_score 放入元组是为了在 f_score 相同时，优先选离终点近的点
        open_set = []
        heapq.heappush(open_set, (0, 0, start_node))
        
        #  使用字典存储 g_score，访问速度快 ---
        g_score = {start_node: 0}
        came_from = {}
        
        #  使用 Set 记录已处理节点 (Closed Set)，查找 O(1) ---
        closed_set = set()

        # 设定到达目标的容差距离 (米²)
        goal_tolerance = 4.0
        #  使用平方距离阈值
        goal_tolerance_sq = (goal_tolerance / self.resolution) ** 2

        final_node = None

        while open_set:
            # 取出 f值最小的节点
            current_f, current_h, current_index = heapq.heappop(open_set)

            # 如果该节点已经被处理过更优的路径，跳过
            if current_index in closed_set:
                continue
            
            # 标记为已处理
            closed_set.add(current_index)

            # --- 优化 5: 终点检查使用平方和，代替 sqrt ---
            dist_sq = (current_index[0] - goal_node[0])**2 + \
                      (current_index[1] - goal_node[1])**2 + \
                      (current_index[2] - goal_node[2])**2
            
            if dist_sq <= goal_tolerance_sq:
                final_node = current_index
                break

            # 遍历邻居
            for dx, dy, dz, move_cost in self.motions:
                neighbor_index = (current_index[0] + dx, 
                                  current_index[1] + dy, 
                                  current_index[2] + dz)
                
                # 如果已经在关闭列表中，直接跳过 (这一步极大提升效率)
                if neighbor_index in closed_set:
                    continue

                # 越界检查 (快速失败)
                # 将索引转回实际坐标进行物理检测
                # 为了效率，可以先粗略判断索引是否在框内，再进行精确 float 转换
                neighbor_pos = self._index_to_pos(neighbor_index)

                if self.env.is_outside(neighbor_pos):
                    continue
                if self.env.is_collide(neighbor_pos):
                    continue

                # 计算新的 G 值 (使用预计算的 move_cost)
                tentative_g = g_score[current_index] + move_cost

                # 如果发现了更短的路径
                if neighbor_index not in g_score or tentative_g < g_score[neighbor_index]:
                    came_from[neighbor_index] = current_index
                    g_score[neighbor_index] = tentative_g
                    
                    # 计算 H 值
                    h = self._heuristic_sq(neighbor_index, goal_node)
                    
                    # F = G + H
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, h, neighbor_index))

        # 回溯路径
        path = []
        curr = final_node if final_node else current_index # 容错
        
        while curr in came_from:
            path.append(self._index_to_pos(curr))
            curr = came_from[curr]
            
        path.append(start_pos)
        path.reverse()
        
        # 加上终点
        if path[-1] != goal_pos:
            path.append(goal_pos)
        
        return np.array(path)

    def _pos_to_index(self, pos):

        return (int(round(pos[0] / self.resolution)),
                int(round(pos[1] / self.resolution)),
                int(round(pos[2] / self.resolution)))

    def _index_to_pos(self, idx):
        return (idx[0] * self.resolution,
                idx[1] * self.resolution,
                idx[2] * self.resolution)

    def _heuristic_sq(self, node1, node2):

        # 优化后的启发函数：使用平方距离 (Squared Euclidean Distance)。
        # 优点：计算极快，搜索节点极少。
        # 缺点：不保证完全最短路径（但在无复杂迷宫的障碍环境下通常效果很好）。

        dx = node1[0] - node2[0]
        dy = node1[1] - node2[1]
        dz = node1[2] - node2[2]
        
        # 返回平方和
        # 乘以 resolution 是为了保持量级与 G 值有一定关联，防止 H 权重过大导致路径完全贴着障碍物走
        return (dx**2 + dy**2 + dz**2) * self.resolution
