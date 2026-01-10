"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
            

import numpy as np
import math
import heapq

class PathPlanner:
    def __init__(self, env):
        self.env = env
        # 设定网格分辨率，例如0.5米一个格子
        self.resolution = 0.5 
        # 定义26个运动方向（3D空间）
        self.motions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.motions.append((dx, dy, dz))

    def plan(self, start_pos, goal_pos):
        """
        A* Algorithm implementation.
        """
        # 将实际坐标转换为网格索引
        start_node = self._pos_to_index(start_pos)
        goal_node = self._pos_to_index(goal_pos)

        # 优先队列 (cost, x, y, z)
        open_set = []
        heapq.heappush(open_set, (0, start_node, [])) # (priority, current_node, path_so_far)
        
        # 记录也就是 cost_so_far
        g_score = {start_node: 0}
        
        # 记录父节点以便回溯路径 
        came_from = {}

        # 重新初始化open_set只存节点
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        while open_set:
            current_priority, current_index = heapq.heappop(open_set)

            # 如果到达目标点附近
            if self._dist(current_index, goal_node) < 2.0: # 允许一定的网格误差，最后修正
                break

            for dx, dy, dz in self.motions:
                neighbor_index = (current_index[0] + dx, 
                                  current_index[1] + dy, 
                                  current_index[2] + dz)
                
                # 检查是否越界或碰撞
                neighbor_pos = self._index_to_pos(neighbor_index)
                
                # 先检查是否越界，再检查碰撞
                if self.env.is_outside(neighbor_pos):
                    continue
                if self.env.is_collide(neighbor_pos):
                    continue

                # 计算新的G值
                step_cost = math.sqrt(dx**2 + dy**2 + dz**2) * self.resolution
                new_g = g_score[current_index] + step_cost

                if neighbor_index not in g_score or new_g < g_score[neighbor_index]:
                    g_score[neighbor_index] = new_g
                    priority = new_g + self._heuristic(neighbor_index, goal_node)
                    heapq.heappush(open_set, (priority, neighbor_index))
                    came_from[neighbor_index] = current_index

        # 回溯路径
        path = []
        # 如果找不到路径，current_index 可能不是目标。需处理异常，这里假设一定能找到
        curr = current_index
        while curr != start_node:
            path.append(self._index_to_pos(curr))
            curr = came_from.get(curr)
            if curr is None: # 防止死循环
                break
        path.append(start_pos)
        path.reverse()
        
        # 强制将终点替换为精确的目标点
        path[-1] = goal_pos
        
        return np.array(path)

    def _pos_to_index(self, pos):
        return (int(round(pos[0] / self.resolution)),
                int(round(pos[1] / self.resolution)),
                int(round(pos[2] / self.resolution)))

    def _index_to_pos(self, idx):
        return (idx[0] * self.resolution,
                idx[1] * self.resolution,
                idx[2] * self.resolution)

    def _heuristic(self, node1, node2):
        # 欧几里得距离作为启发函数
        return math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 + (node1[2]-node2[2])**2) * self.resolution

    def _dist(self, node1, node2):
        return math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2 + (node1[2]-node2[2])**2)









