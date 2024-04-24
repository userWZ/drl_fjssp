import numpy as np
import re

def read_dataset(dataset_url):
    if "instance" in dataset_url:
        with open(dataset_url, 'r') as file:
            lines = file.readlines()
    
        # 获取作业数和机器数
        jobs = int(lines[1].strip())
        machines = int(lines[3].strip())
            # 初始化矩阵
        sequence_matrix = []
        duration_matrix = []
        # 读取序列矩阵
        for i in range(5, 5 + jobs):
            sequence = list(map(int, lines[i].strip().split()))
            sequence_matrix.append(sequence)
        for i in range(6 + jobs, 6 + 2 * jobs):
            # 移除行中所有的括号
            matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', lines[i])
            durations = [list(map(int, match)) for match in matches]
            duration_matrix.append(durations)
        
        # 将方针数组插入到三维数组中
        for row in range(len(duration_matrix)):
            for col in range(len(duration_matrix[0])):
                duration_matrix[row][col].insert(0, sequence_matrix[row][col])
        return duration_matrix
        
    with open(dataset_url, 'r') as file:
        processing_time = []
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾的换行符，并按空格分割成数字列表
            numbers = line.strip().split()

            # 每四个数字为一组，将每行的数字列表分割成若干个四个数字的小组
            groups = [list(map(int, numbers[i:i + 4])) for i in range(0, len(numbers), 4)]
            processing_time.append(groups)
    return processing_time

def read_solution(solution_url):
    with open(solution_url, 'r') as file:
        # 读取第一行，并将其转换为整数列表
        first_line = file.readline().strip().split()
        solution_list = [int(num) for num in first_line]

        # 初始化一个空的二维列表，用于存储整理后的数据
        com_time = []

        # 逐行读取文件内容，每三个数字为一个数组，每行的数组组成一个大的二维数组
        for line in file:
            numbers_str = line.strip().split()
            line_data = []
            # 将每三个数字转换为一个数组，添加到该行的数组中
            for i in range(0, len(numbers_str), 3):
                numbers = [int(num) for num in numbers_str[i:i + 3]]
                line_data.append(numbers)
            # 将该行的数组添加到二维列表中
            com_time.append(line_data)
    return solution_list, com_time

def get_schedule(dataset_url, solution_url):
    processing_time = read_dataset(dataset_url)
    solution_list, completion_time = read_solution(solution_url)
    machine_dict = {}
    for job_idx in range(len(completion_time)):
        for task_idx in range(len(completion_time[0])):
            due_time = decade(processing_time[job_idx][task_idx][1:], completion_time[job_idx][task_idx])
            task_info = {
                'machine': processing_time[job_idx][task_idx][0],
                'job_idx': job_idx,
                'task_idx': task_idx,
                'op_start_time': due_time,
                'op_end_time': completion_time[job_idx][task_idx]
            }
            machine = processing_time[job_idx][task_idx][0]
            if machine in machine_dict:
                machine_dict[machine].append(task_info)
            else:
                machine_dict[machine] = [task_info]
    return machine_dict

def decade(a, b):
    return [b[0]-a[0], b[1]-a[1], b[2]-a[2]]


if __name__ == '__main__':
    machine_schedule = get_schedule("visualization/data/instances/S10.2.txt","visualization\solutions\sol10_10_1.txt")
