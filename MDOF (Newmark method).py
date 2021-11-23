"""
这是一个演示关于如何使用newmark模块
"""

from newmark import Newmark
import numpy as np


mdof1 = Newmark()  # initial a object
mdof1.input_degree(2)  # set degree of freedom

mass = 200
k = 5e5
c = 150
# four degree of freedom
m_matrix = [[mass, 0], [0, mass]]
k_matrix = [[k, -k], [-k, 2 * k]]
c_matrix = [[c, -c], [-c, 2 * c]]
mdof1.input_m_matrix(m_matrix)  # input mass matrix
mdof1.input_k_matrix(k_matrix)
mdof1.input_c_matrix(c_matrix)

mdof1.input_initial_u([0.0, 0.0])
mdof1.input_initial_v([0.0, 0.0])

t_total = 10.0
dt = 0.01
mdof1.set_calculate_time(t_total, dt)   # total time and time step
# external force
t_list = np.arange(0.0, t_total+dt, dt).reshape(-1, 1)
f1 = 500.0 * np.sin(2 * np.pi * 3 * t_list + 0.0)  # + 1.0 * np.sin(2*np.pi/3.0 * t_list + 0.0)
f2 = 0.0 * np.sin(2 * np.pi / 1.0 * t_list + 0.0)
f_all = np.concatenate((f1, f2), axis=1)
mdof1.input_force(f_all)  # force shape [n_time_step, n_degree_of_freedom]

mdof1.set_newmark_parameters(1/2, 1/4)  # gamma, delta (1/2,1/4-梯形法); (1/2, 1/6- 线性加速度法); (1/2, 0 - 中心差分法)

mdof1.calculate()
mdof1.plot_result()
