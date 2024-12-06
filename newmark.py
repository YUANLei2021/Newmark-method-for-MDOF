"""
Author: YUAN Lei (lei2021.yuan@connect.polyu.hk)
Date： 23 Nov 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
import datetime
import pandas as pd


class Newmark:
    """
    Newmark method
    """
    def __init__(self, task_name=None):
        # task number
        if task_name is None:
            self.number_task = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.number_task = task_name
        # default degree
        self.degree = 4

        # four degree of freedom  default matrix
        mass = 1.0
        k = 1.0
        c = 0.0
        self.m_matrix = np.array([[mass, 0, 0, 0], [0, mass, 0, 0], [0, 0, mass, 0], [0, 0, 0, mass]])
        self.k_matrix = np.array([[k, -k, 0, 0], [-k, 2 * k, -k, 0], [0, -k, 2 * k, -k], [0, 0, -k, 2 * k]])
        self.c_matrix = np.array([[c, -c, 0, 0], [-c, 2 * c, -c, 0], [0, -c, 2 * c, -c], [0, 0, -c, 2 * c]])

        # initial condition
        self.u0 = np.zeros([self.degree, 1])
        self.v0 = np.zeros([self.degree, 1])
        # time step
        self.t_total = 10.0
        self.dt = 0.001
        self.t_list = np.arange(0.0, self.t_total + self.dt, self.dt).reshape(-1, 1)
        # parameters in new-mark method
        self.gamma = 1 / 2
        self.delta = 1 / 4

        # external force
        f1 = 1.0 * np.sin(2 * np.pi * 1.0 * self.t_list + 0.0)
        f2 = 0.0 * np.sin(2 * np.pi * 1.0 * self.t_list + 0.0)
        f3 = 0.0 * self.t_list
        f4 = 0.0 * self.t_list
        self.f_all = np.concatenate((f1, f2, f3, f4), axis=1)

        # result
        self.u_all = np.array([])
        self.v_all = np.array([])
        self.a_all = np.array([])
        self.f_all_pred = np.array([])

    def input_degree(self, degree):
        """
        define degree
        :param degree: 自由度
        :return:
        """
        self.degree = degree

    def input_m_matrix(self, m_matrix):
        """
        define mass matrix
        :param m_matrix:
        :return:
        """
        if not isinstance(m_matrix, np.ndarray):
            m_matrix = np.array(m_matrix)
        if m_matrix.shape == (self.degree, self.degree):
            self.m_matrix = m_matrix
        else:
            print("The shape of the mass matrix is inconsistent with the degrees of freedom！")

    def input_k_matrix(self, k_matrix):
        """
        define stiffness matrix
        :param k_matrix:
        :return:
        """
        if not isinstance(k_matrix, np.ndarray):
            k_matrix = np.array(k_matrix)
        if k_matrix.shape == (self.degree, self.degree):
            self.k_matrix = k_matrix
        else:
            print("The shape of the stiffness matrix is inconsistent with the degrees of freedom！")

    def input_c_matrix(self, c_matrix):
        """
        define damping matrix
        :param c_matrix:
        :return:
        """
        if not isinstance(c_matrix, np.ndarray):
            c_matrix = np.array(c_matrix)
        if c_matrix.shape == (self.degree, self.degree):
            self.c_matrix = c_matrix
        else:
            print("The shape of the damping matrix is inconsistent with the degrees of freedom！")

    def input_initial_u(self, u0):
        """
        define initial displacement
        :param u0:
        :return:
        """
        if not isinstance(u0, np.ndarray):
            u0 = np.array(u0).reshape(-1, 1)
        else:
            u0 = u0.reshape(-1, 1)
        if u0.shape[0] == self.degree:
            self.u0 = u0
        else:
            print("The shape of the u0 is inconsistent with the degrees of freedom！")

    def input_initial_v(self, v0):
        """
        define initial velocity
        :param v0:
        :return:
        """
        if not isinstance(v0, np.ndarray):
            v0 = np.array(v0).reshape(-1, 1)
        else:
            v0 = v0.reshape(-1, 1)
        if v0.shape[0] == self.degree:
            self.v0 = v0
        else:
            print("The shape of the v0 is inconsistent with the degrees of freedom！")

    def input_force(self, f_all):
        """
        define external force at every time step
        :param f_all:
        :return:
        """
        if not isinstance(f_all, np.ndarray):
            f_all = np.array(f_all)
        if f_all.shape[1] == self.degree:
            self.f_all = f_all
        else:
            print("The shape of the force is inconsistent with the degrees of freedom！")

    def set_calculate_time(self, t_total, dt):
        """
        define the total time to calculate and delta time of each step
        :param t_total:
        :param dt:
        :return:
        """
        self.t_total = t_total
        self.dt = dt
        n_steps = int(np.round(self.t_total / self.dt))+1
        self.t_list = np.linspace(0.0, self.t_total, n_steps).reshape(-1, 1)

    def set_newmark_parameters(self, gamma, delta):
        """
        parameters in the newmark method
        (1/2,1/4-Trapezoidal method); (1/2, 1/6- Linear acceleration method); (1/2, 0 - Central difference method)
        :param gamma:
        :param delta:
        :return:
        """
        self.gamma = gamma
        self.delta = delta

    def calculate(self):
        print(f'Task:{self.number_task} start!')
        a0 = (np.linalg.inv(self.m_matrix).dot(self.f_all[0, :].reshape(-1, 1) - self.k_matrix.dot(self.u0)
                                               - self.c_matrix.dot(self.v0))).T

        z0 = 1 / self.dt ** 2 / self.delta
        z1 = self.gamma / self.dt / self.delta
        z2 = 1 / self.dt / self.delta
        z3 = 1 / 2 / self.delta - 1
        z4 = self.gamma / self.delta - 1
        z5 = self.dt / 2 * (self.gamma / self.delta - 2)
        z6 = self.dt * (1 - self.gamma)
        z7 = self.gamma * self.dt

        k_equal = self.k_matrix + z0 * self.m_matrix + z1 * self.c_matrix
        l_matrix = linalg.cholesky(k_equal, lower=True)
        j_matrix = np.linalg.inv(l_matrix.T).dot(np.linalg.inv(l_matrix))

        u_all = [self.u0.T]
        v_all = [self.v0.T]
        a_all = [a0]

        for i in range(self.t_list.shape[0]-1):
            f_t = self.f_all[i].reshape(-1, 1)
            x_t = u_all[i].reshape(-1, 1)
            v_t = v_all[i].reshape(-1, 1)
            a_t = a_all[i].reshape(-1, 1)
            q_t = f_t + self.m_matrix.dot(z0 * x_t + z2 * v_t + z3 * a_t) + self.c_matrix.dot(z1 * x_t + z4 * v_t
                                                                                              + z5 * a_t)
            x_t2 = j_matrix.dot(q_t)
            a_t2 = z0 * (x_t2 - x_t) - z2 * v_t - z3 * a_t
            v_t2 = v_t + z6 * a_t + z7 * a_t2
            u_all.append(x_t2.T)  # np.append(u_all, x_t2.T, axis=0)
            v_all.append(v_t2.T)  # np.append(v_all, v_t2.T, axis=0)
            a_all.append(a_t2.T)  # np.append(a_all, a_t2.T, axis=0)

        u_all = np.array(u_all).squeeze()
        v_all = np.array(v_all).squeeze()
        a_all = np.array(a_all).squeeze()

        f_all_pred = (self.m_matrix.dot(a_all.T) + self.k_matrix.dot(u_all.T) + self.c_matrix.dot(v_all.T)).T
        self.u_all = u_all
        self.v_all = v_all
        self.a_all = a_all
        self.f_all_pred = f_all_pred
        print('Calculate finish! ')

        data = pd.DataFrame(self.t_list, columns=['time'])
        u_name, v_name, a_name, f_name, f_name2 = [], [], [], [], []
        for i in range(self.degree):
            u_name.append(f'u{i + 1}')
            v_name.append(f'v{i + 1}')
            a_name.append(f'a{i + 1}')
            f_name.append(f'f_{i + 1}_pred')
            f_name2.append(f'f{i + 1}')

        dirs = ['./data', './png']
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        u_data = pd.DataFrame(self.u_all, columns=u_name)
        v_data = pd.DataFrame(self.v_all, columns=v_name)
        a_data = pd.DataFrame(self.a_all, columns=a_name)
        f_pred_data = pd.DataFrame(self.f_all_pred, columns=f_name)
        f_data = pd.DataFrame(self.f_all, columns=f_name2)
        data = pd.concat([data, u_data, v_data, a_data, f_pred_data, f_data], axis=1)
        data.to_csv(f'./data/{self.number_task}-data.csv', index=False, header=True)
        print(f'All data is saved in "data/{self.number_task}-data.csv"!')
        return u_all, v_all, a_all, f_all_pred

    def plot_result(self):
        """
        plot the result- u; v; a; f
        :return:
        """
        dirs = ['./data', './png']
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        u_all = self.u_all
        v_all = self.v_all
        a_all = self.a_all
        f_all_pred = self.f_all_pred

        l_data = u_all.shape[0]
        if l_data > 1000:
            n_step = l_data // 1000
        else:
            n_step = 1
        t_list = self.t_list[::n_step]
        plt.figure()
        for i in range(u_all.shape[1]):
            plt.plot(t_list, u_all[::n_step, i], label=f'u{i+1}')
        plt.legend()
        plt.savefig(f'./png/{self.number_task}-displacement.png')
        plt.show()

        plt.figure()
        for i in range(v_all.shape[1]):
            plt.plot(t_list, v_all[::n_step, i], label=f'v{i + 1}')
        plt.legend()
        plt.savefig(f'./png/{self.number_task}-velocity.png')
        plt.show()

        plt.figure()
        for i in range(a_all.shape[1]):
            plt.plot(t_list, a_all[::n_step, i], label=f'a{i + 1}')
        plt.legend()
        plt.savefig(f'./png/{self.number_task}-acceleration.png')
        plt.show()

        plt.figure()
        for i in range(self.f_all.shape[1]):
            plt.plot(t_list, f_all_pred[::n_step, i], label=f'f{i + 1}_pred')
            plt.scatter(t_list, self.f_all[::n_step, i], label=f'f{i + 1}_exact', s=3, c='k', marker='x')
        plt.legend()
        plt.savefig(f'./png/{self.number_task}-force.png')
        plt.show()

        print('Plot finish! \nAll figures are saved!')
