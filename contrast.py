import os
import pickle
import random

import torch
import time
import numpy as np
import global_util
from env.fjssp_env import FjsspEnv
from Params import setting_params
from visualization.visual import *
from visualization.utils import read_dataset
from result_generator import to_dataframe
from models.actor_critic import ActorCritic
from jssp_tool.rl.agent.ppo.ppo_discrete import PPODiscrete
import pandas as pd


class TriangleFuzzyNumber:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def expected_value(self):
        """计算并返回三角模糊数的期望值"""
        return (self.a + 2 * self.b + self.c) / 4

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c})"
    


class Job:
    def __init__(self, job_id, machine_sequence, durations):
        self.job_id = job_id
        self.machine_sequence = machine_sequence
        self.durations = durations

    def __repr__(self):
        return f"Job {self.job_id}: Machines {self.machine_sequence}, Durations {self.durations}"


def parse_triangle_fuzzy_numbers(line):
    parts = line.strip().split(') ')
    durations = []
    for part in parts:
        numbers = part.strip('()').split(',')
        durations.append(TriangleFuzzyNumber(int(numbers[0]), int(numbers[1]), int(numbers[2])))
    return durations


def read_job_shop_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    num_jobs = int(lines[1].strip().split()[-1])
    num_machines = int(lines[3].strip().split()[-1])

    machine_sequences = []
    for i in range(5, 5 + num_jobs):
        sequence = list(map(int, lines[i].strip().split()))
        machine_sequences.append(sequence)

    durations = []
    for i in range(5 + num_jobs, 3 + 2 * num_jobs):
        duration_line = parse_triangle_fuzzy_numbers(lines[i])
        durations.append(duration_line)

    jobs = []
    for i in range(num_jobs):
        job = Job(i, machine_sequences[i], durations[i])
        jobs.append(job)

    return jobs, num_machines


if __name__ == '__main__':
    # Specify the path to your file
    file_path = "/disk1/all_users_sed/yangwj/drl_fjssp/data/instances/ABZ5_Z.txt"
    jobs, machines = read_job_shop_data(file_path)
    # Print out the parsed data
    for job in jobs:
        print(job)