"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self):
        pass

    def generate_trajectory(self, path, avg_speed=2.0):
        """
        Generate a smooth trajectory from discrete path points using Cubic Spline.
        
        Args:
            path: Nx3 numpy array of path points.
            avg_speed: Average speed to estimate time allocation.
        Returns:
            trajectory_func: A function or object that returns (x,y,z) given t.
            time_points: The time stamps for the original path points.
        """
        path = np.array(path)
        x = path[:, 0]
        y = path[:, 1]
        z = path[:, 2]

        # 1. 分配时间戳
        # 根据点之间的欧氏距离分配时间
        dists = np.diff(path, axis=0)
        dists = np.sqrt((dists**2).sum(axis=1))
        # 时间 = 距离 / 速度
        dt = dists / avg_speed
        
        times = np.zeros(len(path))
        times[1:] = np.cumsum(dt)
        self.total_time = times[-1]

        # 2. 三次样条插值
        # bc_type='natural' 意味着在起点和终点的导数为0
        self.cs_x = CubicSpline(times, x, bc_type='natural')
        self.cs_y = CubicSpline(times, y, bc_type='natural')
        self.cs_z = CubicSpline(times, z, bc_type='natural')
        
        self.original_path = path
        self.keyframe_times = times
        
        return times, self.total_time

    def get_state(self, t):
        return np.array([self.cs_x(t), self.cs_y(t), self.cs_z(t)])

    def visualize(self):
        """
        Visualize the x, y, z trajectories over time.
        """
        t_sample = np.linspace(0, self.total_time, 200)
        x_smooth = self.cs_x(t_sample)
        y_smooth = self.cs_y(t_sample)
        z_smooth = self.cs_z(t_sample)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        # Plot X
        axs[0].plot(t_sample, x_smooth, label='Trajectory X(t)', color='r')
        axs[0].scatter(self.keyframe_times, self.original_path[:, 0], color='k', marker='o', label='Path Points')
        axs[0].set_ylabel('X Position (m)')
        axs[0].set_title('Trajectory Time History')
        axs[0].grid(True)
        axs[0].legend()

        # Plot Y
        axs[1].plot(t_sample, y_smooth, label='Trajectory Y(t)', color='g')
        axs[1].scatter(self.keyframe_times, self.original_path[:, 1], color='k', marker='o', label='Path Points')
        axs[1].set_ylabel('Y Position (m)')
        axs[1].grid(True)

        # Plot Z
        axs[2].plot(t_sample, z_smooth, label='Trajectory Z(t)', color='b')
        axs[2].scatter(self.keyframe_times, self.original_path[:, 2], color='k', marker='o', label='Path Points')
        axs[2].set_ylabel('Z Position (m)')
        axs[2].set_xlabel('Time (s)')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()