import copy
from abc import ABC

import gym
import numpy as np


class BaseEnv(gym.Env, ABC):
    def __init__(self, n_j, n_m, device):
        self.n_j = n_j
        self.n_m = n_m
        self.device = device
        self.task_size = n_j * n_m

        # 每个task处理时长
        self.task_durations = None
        # 每个task对应的机器编号
        self.task_machines = None
        # 已经调度了的task的id列表
        self.scheduled_task_ids = None

        # ------- 用于辅助计算task的结束时间 ---------
        # 已调度任务的结束时间
        self.task_finish_times = {
            "left": None,
            "peak": None,
            "right": None,
        }
        

        # self.machine_start_times = None
        # self.machine_scheduled_tasks = None
        # 记录三种操作时间下机器被task占用的时间段, list of tuples, [(task,起始时间,结束时间),...]
        self.machine_occupied_times = {
            "left": None,
            "peak": None,
            "right": None,
        }

        # ------- 用于记录 ---------
        self.episode_reward = 0
        # 当前所有已调度的任务的结束时间，即t时刻的C_max
        self.cur_make_span = 0
        self.step_count = 0
        # ------- 用于绘图 ---------
        self.history_make_span = []

    def compute_task_schedule_time(self, task_id, row, col):
        """
        对于新来的task，检测在三种操作时间的情况下
        是否都可以插入当前已调度机器时间中的空闲片段:
            如果三个操作时间任何一个的插入会导致覆盖已有调度时间片段，则调度顺序都要则放到最后;
            否则可以插入空闲片段
        Args:
            task_id:
            row:
            col:
        Returns:
        """
        job_pre_task_col = col - 1 if col > 0 else 0
        job_task_ready_time = {
            "left": self.task_finish_times['left'][row, job_pre_task_col],
            "peak": self.task_finish_times['peak'][row, job_pre_task_col],
            "right": self.task_finish_times['right'][row, job_pre_task_col],
        }
        
        task_duration = {
            "left": self.task_durations['left'][row, col],
            "peak": self.task_durations['peak'][row, col],
            "right": self.task_durations['right'][row, col],
        }
        
        machine_id = int(self.task_machines[row, col])
        
        # 三种操作时间下的是否都可以插入当前已调度机器时间中的空闲片段
        inserted_left, start_time_left, insert_pos_left = self.insert_task(
            self.machine_occupied_times['left'][machine_id], task_id, job_task_ready_time['left'], task_duration['left'])
        inserted_peak, start_time_peak, insert_pos_peak = self.insert_task(
            self.machine_occupied_times['peak'][machine_id], task_id, job_task_ready_time['peak'], task_duration['peak'])
        inserted_right, start_time_right, insert_pos_right = self.insert_task(
            self.machine_occupied_times['right'][machine_id], task_id, job_task_ready_time['right'], task_duration['right'])
        
        start_time = {
            "left": start_time_left,
            "peak": start_time_peak,
            "right": start_time_right,
        }
        if inserted_left and inserted_peak and inserted_right and insert_pos_left == insert_pos_peak == insert_pos_right:
            # 三种操作时间都可以插入当前已调度机器时间中的空闲片段
            self.put_between(task_id, machine_id, start_time, insert_pos_left, task_duration)
        else:
            # 三种操作时间任何一个的插入会导致覆盖已有调度时间片段，则调度顺序都要则放到最后, 即机器的最后
            # 重新计算start_time,保证其必定是machine_ready_time, job_task_ready_time[key]的最大值
            for key in start_time.keys():
                machine_ready_time = 0
                if len(self.machine_occupied_times[key][machine_id]) > 0:
                    machine_ready_time = self.machine_occupied_times[key][machine_id][-1][2]
                start_time[key] = max(machine_ready_time, job_task_ready_time[key])
            self.put_end(task_id, machine_id, start_time, task_duration)
        # 更新机器完成周期
        self.cur_make_span = np.max(self.task_finish_times["right"])
    
    def insert_task(self, machine_occupied_times_true, task_id, job_task_ready_time, task_duration):
        """
        根据机器被占用的时间，考虑将task插入到什么位置

        Args:
            machine_occupied_times (List): task所对应机器被占用的时间段
            task_id (int): task的id
            job_task_ready_time (int): task前序task的结束时间
            task_duration (_type_):  task持续时间
        Returns:
            inserted: 是否被插入
            start_time: 计算得到的task的开始时间
            insert_pos: 插入位置（如果为-1则直接安排到末尾）
        """
        # 寻找可用的插入空隙
        # 记录机器被task占用的时间段, list of tuples,  [[(task,起始时间,结束时间),...],[],[],...]
        machine_occupied_times = copy.deepcopy(machine_occupied_times_true)
        insert_pos = -1
        inserted = False
        for i, (id, ostart, oend) in enumerate(machine_occupied_times):
            if ostart > job_task_ready_time:
                insert_pos = i
                break
            
        # 机器中开始时间没有晚于当前task开始时间，直接放到最后情况下，task的start_time
        machine_ready_time = 0
        if len(machine_occupied_times) > 0:
            machine_ready_time = machine_occupied_times[-1][2]
        start_time = max(machine_ready_time, job_task_ready_time)
        
        if insert_pos == -1:
            # 机器中开始时间没有晚于当前task开始时间的，直接放到最后
            return False, start_time, -1
        
        # 添加虚拟的时间片段,方便间隔计算
        machine_occupied_times.insert(insert_pos, (task_id, job_task_ready_time, job_task_ready_time + task_duration))
        
        # 计算可用的空闲间隔
        for i in range(insert_pos, len(machine_occupied_times) - 1):
            # 对于后续位置，计算每个位置之间的时间间隔
            if i == insert_pos:
                start_time = max(machine_occupied_times[i][1], machine_occupied_times[i - 1][2] if i > 0 else 0)
            else:
                start_time = machine_occupied_times[i][2]
            gap = machine_occupied_times[i + 1][1] - start_time
            # 判断空闲间隔是否足够当前task执行
            if gap >= task_duration:
                inserted = True
                return inserted, start_time, i
            
        # inserted为False,机器中没有足够的空闲时间片段,放到最后
        return inserted, start_time, -1
        

    def put_between(self, task_id, machine_id, start_time, insert_pos, task_duration):
        """
        在已调度序列中间插入调度任务,
        :param task_id: 插入的任务ID
        :param machine_id: 机器id
        :param start_time: 开始时间
        :param insert_pos: 插入位置
        :param task_duration:
        :return:
        """
        
        item = {
            "left": self.machine_occupied_times['left'][machine_id][insert_pos],
            "peak": self.machine_occupied_times['peak'][machine_id][insert_pos],
            "right": self.machine_occupied_times['right'][machine_id][insert_pos],
        } 
        
        # task_id调度顺序中插入task_scheduled_id
        ind = np.argwhere(np.array(self.scheduled_task_ids) == item["left"][0])
        self.scheduled_task_ids.insert(ind[0][0], task_id)
        
        row, col = task_id // self.n_m, task_id % self.n_m
        
        # 更新task完成时间
        self.task_finish_times['left'][row, col] = start_time["left"] + task_duration["left"]
        self.task_finish_times['peak'][row, col] = start_time["peak"] + task_duration["peak"]
        self.task_finish_times['right'][row, col] = start_time["right"] + task_duration["right"]
        
        # 更新机器被占用时间
        self.machine_occupied_times['left'][machine_id].insert(insert_pos, (task_id, start_time["left"], start_time["left"] + task_duration["left"]))
        self.machine_occupied_times['peak'][machine_id].insert(insert_pos, (task_id, start_time["peak"], start_time["peak"] + task_duration["peak"]))
        self.machine_occupied_times['right'][machine_id].insert(insert_pos, (task_id, start_time["right"], start_time["right"] + task_duration["right"]))

    def put_end(self, task_id, machine_id, start_time, task_duration):
        self.scheduled_task_ids.append(task_id)
        
        row, col = task_id // self.n_m, task_id % self.n_m
        
         # 更新task完成时间
        self.task_finish_times['left'][row, col] = start_time["left"] + task_duration["left"]
        self.task_finish_times['peak'][row, col] = start_time["peak"] + task_duration["peak"]
        self.task_finish_times['right'][row, col] = start_time["right"] + task_duration["right"]
        
        # 更新机器被占用时间
        self.machine_occupied_times['left'][machine_id].append((task_id, start_time["left"], start_time["left"] + task_duration["left"]))
        self.machine_occupied_times['peak'][machine_id].append((task_id, start_time["peak"], start_time["peak"] + task_duration["peak"]))
        self.machine_occupied_times['right'][machine_id].append((task_id, start_time["right"], start_time["right"] + task_duration["right"]))
