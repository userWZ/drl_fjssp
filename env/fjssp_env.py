import copy
import random
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
from gym.core import ObsType

from env.utils import gen_instance_uniformly, gen_instance_triangle
from env.base_env import BaseEnv


class TFN:
    def __init__(self, left, peak, right):
        self.left = left
        self.peak = peak
        self.right = right

class FjsspEnv(BaseEnv):
    def __init__(self, n_j, n_m, dur_low, dur_high, device):
        """
        模糊作业车间调度环境
        每一个工序task的操作时间为模糊数，因此工序的due_time和completion_time也为模糊数。
        本环境采用三角模糊数TFN作为模糊数。TFN = (Left, peak, right)
        将三角模糊数拆分为三个同步运行的线程，每个线程中工序的操作时间分别为三角模糊数的三个参数。
        三个线程中机器调度的工序顺序要求一致。
        Args:
            n_j:  job数量
            n_m: machine数量
            dur_low: task最短处理时间
            dur_high: task最长处理时间
            device: 环境运行的机器
        """
        super().__init__(n_j, n_m, device)
        # task的操作时间范围
        self.dur_low = dur_low
        self.dur_high = dur_high

        # 给每个task起个id
        self.task_ids = None

        # 标记哪些task已调度
        self.scheduled_marks = None

        # 邻接矩阵作为状态
        self.adj_matrix = None

        # 最大结束时间
        self.estimated_max_end_time = {
            "left": 0,
            "peak": 0,
            "right": 0,
        }

        # 工序task操作时间
        self.task_durations = {
            "left": None,
            "peak": None,
            "right": None,
        }

        # job的逐个累加task操作时间结果
        self.low_bounds = {
            "left": None,
            "peak": None,
            "right": None,
        }

        # 当前可供选择的task id，最多n_j个
        self.candidates = None

        # 标志哪些job是否结束
        self.mask = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        # episode的累计奖励
        self.episode_reward = 0
        self.cur_make_span = 0
        self.step_count = 0
        # episode实例读取
        
        if "data" in kwargs:
            (self.task_durations['left'], self.task_durations['peak'], self.task_durations['right'],
             self.task_machines) = kwargs.get("data")
            self.n_j, self.n_m = self.task_durations["left"].shape
        else:
            self.n_j = kwargs.get("n_j")
            self.n_m = kwargs.get("n_m")
            # 生成一个新的调度案例（每一个工序的操作时间 和 工序对应的机器编号）
            # task_durations: shape (n_j, n_m, 3)
            (self.task_durations['left'], self.task_durations['peak'], self.task_durations['right'],
             self.task_machines) = gen_instance_triangle(
                self.n_j, self.n_m, self.dur_low, self.dur_high
            )

        self.task_size = self.n_j * self.n_m
        # machine被占用的时间片段
        self.machine_occupied_times['left'] = [[] for _ in range(self.n_m)]
        self.machine_occupied_times['peak'] = [[] for _ in range(self.n_m)]
        self.machine_occupied_times['right'] = [[] for _ in range(self.n_m)]
        
        self.task_ids = np.array([i for i in range(self.n_j * self.n_m)], dtype=np.int64).reshape((self.n_j, self.n_m))
        # 已经调度的task_id列表
        self.scheduled_task_ids = []
        # job已经调度过的task的mask矩阵
        self.scheduled_marks = np.zeros_like(self.task_machines, dtype=np.int32)

        # job的逐个累加task操作时间结果 三角模糊数 (low, peak, high)
        # low_bounds[i, j]表示第i个job的前j个task的模糊数操作时间的累加结果
        self.low_bounds['left'] = np.cumsum(self.task_durations['left'], axis=1, dtype=np.single)
        self.low_bounds['peak'] = np.cumsum(self.task_durations['peak'], axis=1, dtype=np.single)
        self.low_bounds['right'] = np.cumsum(self.task_durations['right'], axis=1, dtype=np.single)

        # 最小化最大完成时间
        self.estimated_max_end_time['left'] = np.max(self.low_bounds['left'])
        self.estimated_max_end_time['peak'] = np.max(self.low_bounds['peak'])
        self.estimated_max_end_time['right'] = np.max(self.low_bounds['right'])

        # 构建邻接矩阵
        self.adj_matrix = self.build_adjacency_matrix()

        # task完成时间
        self.task_finish_times['left'] = np.zeros((self.n_j, self.n_m), dtype=np.single)
        self.task_finish_times['peak'] = np.zeros((self.n_j, self.n_m), dtype=np.single)
        self.task_finish_times['right'] = np.zeros((self.n_j, self.n_m), dtype=np.single)

        # 特征矩阵
        feature = np.concatenate(
            [
                self.low_bounds['left'].reshape(-1, 1) / 1000,
                self.low_bounds['peak'].reshape(-1, 1) / 1000,
                self.low_bounds['right'].reshape(-1, 1) / 1000,
                self.scheduled_marks.reshape(-1, 1)], axis=1, dtype=np.float32
        )
        # 初始可执行工序
        self.candidates = copy.deepcopy(self.task_ids[:, 0])
        self.mask = np.array([False for _ in range(self.n_j)])
        obs = copy.deepcopy((self.adj_matrix, feature, self.candidates, self.mask))
        obs = self._to_tensor(obs)
        info = {}

        return obs, info

    def step(self, task_id: int) -> Tuple[ObsType, float, bool, bool, dict]:
        # task_id = self.candidates[action]
        # task_id=action
    
        if task_id not in self.scheduled_task_ids:
            # self.scheduled_task_ids.append(task_id)
            self.step_count += 1
            # 根据task_id确定调度的是那个job的哪一个task
            row = task_id // self.n_m  # job
            col = task_id % self.n_m  # task
            # 把选择调度的动作mask掉
            self.scheduled_marks[row, col] = 1

            # 计算执行当前task结束时间
            self.compute_task_schedule_time(task_id, row, col)
            # 更新low_bounds, 根据已调度的task更新其后续的task的预期完成时间
            self.update_low_bounds(row, col)
            # 根据调度结果，重新生成邻接矩阵
            self.adj_matrix = self.build_adjacency_matrix()

        # 如果不是某个job的最后一个task，更新下一次的candidate
        if task_id not in self.task_ids[:, -1]:
            self.candidates[task_id // self.n_m] += 1
        else:
            # 标志某个job调度所有工序都调度结束
            self.mask[task_id // self.n_m] = True
        # 寻找
        feature = np.concatenate(
            [
                self.low_bounds['left'].reshape(-1, 1) / 1000,
                self.low_bounds['peak'].reshape(-1, 1) / 1000,
                self.low_bounds['right'].reshape(-1, 1) / 1000,
                self.scheduled_marks.reshape(-1, 1)], axis=1, dtype=np.float32
        )
        obs = copy.deepcopy((self.adj_matrix, feature, self.candidates, self.mask))
    
        obs = self._to_tensor(obs)
        # 三组奖励一起算            
        reward = {
            'left': self.estimated_max_end_time["left"] - np.max(self.low_bounds["left"]),
            'peak': self.estimated_max_end_time["peak"] - np.max(self.low_bounds["peak"]),
            'right': self.estimated_max_end_time["right"] - np.max(self.low_bounds["right"]),
        }
        # TODO  改进奖励函数
        total_reward = (reward["left"] + 2 * reward["peak"] + reward["right"]) / 4
                    
        # 最小化最大完成时间
        self.estimated_max_end_time["left"] = np.max(self.low_bounds["left"])
        self.estimated_max_end_time["peak"] = np.max(self.low_bounds["peak"])
        self.estimated_max_end_time["right"] = np.max(self.low_bounds["right"])
        terminated = False
        if len(self.scheduled_task_ids) == self.task_size:
            terminated = True
            # calculate makespan
            cur_make_span = np.max(self.compute_expect_fuzzy(self.low_bounds['left'][:, -1], 
                                                             self.low_bounds['peak'][:, -1],
                                                             self.low_bounds['right'][:, -1]))
            self.cur_make_span = cur_make_span
        
        info = {}
        
        self.episode_reward += total_reward
        
        return obs, total_reward, terminated, False, info

    def render(self):
        """
        可视化
        Returns
        -------

        """
        plt.figure(figsize=(12, 8))
        colors = list(mc.TABLEAU_COLORS.keys())
        for machine_idx in range(1, self.n_m+1):
            y = machine_idx
            plt.axhline(y, color='black')  # 绘制水平线，表示机器
        plt.yticks(list(range(self.n_m)))  # 设置y轴刻度
        for task in self.scheduled_task_ids:
            job_idx = task // self.n_m
            task_idx = task % self.n_m
            start = [self.task_finish_times['left'][job_idx, task_idx], self.task_finish_times['peak'][job_idx, task_idx], self.task_finish_times['right'][job_idx, task_idx]]
            end = [self.task_finish_times['left'][job_idx, task_idx], self.task_finish_times['peak'][job_idx, task_idx], self.task_finish_times['right'][job_idx, task_idx]]
            y = self.task_machines[job_idx, task_idx] + 1
            if start == [0, 0, 0]:
                plt.scatter(0, y, color=colors[job_idx])  # 绘制起始点
                plt.text(start[1], y-0.2, str(job_idx)+','+str(task_idx), verticalalignment='center', horizontalalignment='center',
                        fontsize=6)  # 在起始点添加文本标签
            else:    
                triangleX = start
                triangleY = [y, y-0.2, y]
                plt.fill(triangleX, triangleY, colors[job_idx])  # 绘制起始时间的三角形
                plt.text(start[1], y-0.3, str(job_idx)+','+str(task_idx) , verticalalignment='center', horizontalalignment='center',
                        fontsize=6)  # 在起始时间的三角形上添加文本标签
            triangleX = end
            triangleY = [y, y+0.2, y]
            plt.fill(triangleX, triangleY, colors[job_idx])  # 绘制结束时间的三角形
            plt.text(end[1], y+0.3, str(job_idx)+','+str(task_idx), verticalalignment='center', horizontalalignment='center', fontsize=6)  # 在结束时间的三角形上添加文本标签
        plt.show()
    
    def compute_expect_fuzzy(self, left, peak, right):
        """
        计算三角模糊数的期望
        Args:
            left:
            peak:
            right:

        Returns:

        """
        return (left + 2*peak + right) / 4

    def update_low_bounds(self, row, col):
        for key in self.low_bounds.keys():
            # 更新当前task的结束时间
            self.low_bounds[key][row, col] = self.task_finish_times[key][row, col]
            # 对于其他未调度的task，采用加上前面完成时间作为low bounds
            for i in range(self.n_j):
                for j in range(self.n_m):
                    if self.task_finish_times[key][i, j] == 0:
                        if j == 0:
                            self.low_bounds[key][i, j] = self.task_durations[key][i, j]
                        else:
                            self.low_bounds[key][i, j] = self.task_durations[key][i, j] + self.low_bounds[key][i, j - 1]

    def build_adjacency_matrix(self):
        """
        构建邻接矩阵( 有向图 )
        """
        adj_matrix = np.eye(self.task_size, dtype=np.single)
        # for i in range(1, 1 + self.task_size):
        #     if i == 0 or i % self.n_m != 0:
        #         for j in range(1, self.task_size):
        #             if i == j:
        #                 adj_matrix[i - 1, j] = 1

        # 根据scheduled_task_ids更新邻接矩阵
        for tasks in self.machine_occupied_times['left']:
            for i in range(0, len(tasks) - 1):
                adj_matrix[tasks[i][0], tasks[i + 1][0]] = 1
                # adj_matrix[tasks[i + 1][0], tasks[i][0]] = 1

        return adj_matrix

    def _to_tensor(self, obs):
        adj, feature, candidate, finish_mark = obs
        # adj = torch.tensor(adj.T, dtype=torch.float32).to(self.device).to_sparse()
        # feature = torch.tensor(feature, dtype=torch.float32).to(self.device)
        # candidate = torch.tensor(candidate, dtype=torch.int64).to(self.device).unsqueeze(0)
        # finish_mark = torch.tensor(finish_mark, dtype=torch.bool).to(self.device).unsqueeze(0)
        return adj.T, feature, candidate, finish_mark

    def max_TFN(self, A, B):
        """
        计算两个TFN的最大值
        A=(aL,a,aR), B=(bL,b,bR)
        max{A, B} =(max(aL,bL), max(a, b), max(aR,bR)) .
        """
        return (max(A[0], B[0]), max(A[1], B[1]), max(A[2], B[2]))


if __name__ == '__main__':
    env = FjsspEnv(6, 6, 1, 99, 'cpu')
    import os 
    from utils import get_project_root
    data = np.load(
        os.path.join(get_project_root(), "data", "generatedData{}_{}_BatchSize10_Seed{}.npy").format(
            6, 6, 200
        )
    )
    env.reset(data=data[0])
    for i in range(36):
        next_obs, reward, done, _, _ = env.step(i)
        env.render()
