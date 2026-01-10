from flight_environment import FlightEnvironment

env = FlightEnvironment(50)
start = (1,2,0)
goal = (18,18,3)


import numpy as np
import matplotlib.pyplot as plt
from flight_environment import FlightEnvironment
from path_planner import PathPlanner
from trajectory_generator import TrajectoryGenerator

# 初始化环境
env = FlightEnvironment(50)
start = (1, 2, 0)
goal = (18, 18, 3)


# 1. Path Planning (路径规划)
print("1. 正在规划路径...")

# 实例化规划器
planner = PathPlanner(env)

# 调用 A* 算法计算路径
# path 是一个 N x 3 的 numpy 数组或列表
path = planner.plan(start, goal)

print(f"   路径规划完成，共生成 {len(path)} 个路径点。")

# Visualization 1: 3D Path (路径可视化)
print("2. 显示路径可视化...")
print("   >>> 请手动关闭弹出的 3D 路径窗口，以显示轨迹 <<<")

# 调用环境自带的绘图函数
env.plot_cylinders(path)

# 2. Trajectory Generation (轨迹生成)
print("3. 正在生成平滑轨迹 (Generating Trajectory)...")

# 实例化轨迹生成器
traj_gen = TrajectoryGenerator()

# 生成轨迹
# avg_speed 设置为 2.0 m/s 用于估算时间戳
times, total_time = traj_gen.generate_trajectory(path, avg_speed=2.0)

print(f"   轨迹生成完成，总飞行时间估算为: {total_time:.2f} 秒")

# Visualization 2: Trajectory Curves (轨迹曲线图)
print("4. 显示轨迹时间历程图 (x-t, y-t, z-t)...")

# 调用 visualize 
traj_gen.visualize()

# 显示最后一张图表
plt.show()

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.



# --------------------------------------------------------------------------------------------------- #




# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.




# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
