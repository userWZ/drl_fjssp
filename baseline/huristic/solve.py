import re
from collections import defaultdict
from scheduler import Scheduler, FuzzyNumber
import time
import os
import pandas as pd
import argparse
import numpy as np

def parse_file(dataset_url: str):
    with open(dataset_url, 'r') as file:
            lines = file.readlines()
            
    # 获取作业数和机器数        
    num_jobs = int(lines[1])
    num_machine = int(lines[3])

    jobs = {}
    task_id = 1
    line_index = 5

    for job_id in range(1, num_jobs + 1):
        machines = list(map(int, lines[line_index].strip().split()))
        jobs[job_id] = {"tasks": []}
        for machine in machines:
            jobs[job_id]["tasks"].append({"id": task_id, "machine": machine})
            task_id += 1
        line_index += 1
    

    fuzzy_durations = []
    for i in range(6 + num_jobs, 6 + 2 * num_jobs):
        # 移除行中所有的括号
        matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', lines[i])
        durations = [list(map(int, match)) for match in matches]
        fuzzy_durations.append(durations)

    task_id = 1
    for job_id in range(1, num_jobs + 1):
        for task_index, duration in enumerate(fuzzy_durations[job_id - 1]):
            jobs[job_id]["tasks"][task_index]["processing_time"] = FuzzyNumber(*duration)
            task_id += 1

    dependencies = job2dependency(jobs)

    return jobs, dependencies


def job2dependency(jobs):
    dependencies = defaultdict(list)
    for job_id, job in jobs.items():
        tasks = job["tasks"]
        for i in range(len(tasks) - 1):
            dependencies[tasks[i]["id"]].append(tasks[i + 1]["id"])
    return dependencies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_type', type=str, default='synthetic')
    parser.add_argument('--instance', type=str, default='instances')
    parser.add_argument('--instance_nums', type=int, default=50)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--n_j', type=int, default=10)
    parser.add_argument('--n_m', type=int, default=5)
    parser.add_argument('--pdr', type=str, default='mwkr')
    args = parser.parse_args()
    instance_type = args.instance_type
    instance_url = os.path.join(args.instance, instance_type)

    result = []
    
    pdrs = {
        'spt': 'shortest_processing_time',
        'mwkr': 'most_work_remaining',
        'mopr': 'most_operations_remaining',
        'lpt': 'longest_processing_time',
        'lifo': 'last_in_first_out',
        'lor': 'least_operations_remaining',
        'fifo': 'first_come_first_served',
    }
    
    if instance_type == 'synthetic':
        instance_file = os.path.join(args.instance, 'synthetic', f"synthetic_{args.n_j}_{args.n_m}_instanceNums{args.instance_nums}_Seed{args.seed}.npy")   
        instances = np.load(instance_file)
        for i, data in enumerate(instances):
            start = time.time()
            left, peak, right, machine = data
            jobs = {}
            for job_id in range(1, len(machine) + 1):
                jobs[job_id] = {"tasks": []}
                for task_id in range(1, len(machine[job_id - 1]) + 1):
                    jobs[job_id]["tasks"].append({
                        "id": task_id,
                        "machine": int(machine[job_id - 1][task_id - 1]),
                        "processing_time": FuzzyNumber(left[job_id - 1][task_id - 1], peak[job_id - 1][task_id - 1], right[job_id - 1][task_id - 1])
                    })
            dependencies = job2dependency(jobs)
            scheduler = Scheduler(jobs, dependencies, pdrs['mwkr'])
            makespan = scheduler.schedule()
            end = time.time()
            print(f"Instance {i} with PDR {args.pdr} has makespan {makespan}, ", end - start)
            result.append([i, makespan, end - start])
    else:
        instances_path = os.path.join(args.instance, 'benchmarks')
        instances = os.listdir(instances_path)
        for instance in instances:
            start = time.time()
            print(f"Processing {instance}")
            instance_url = os.path.join(instances_path, instance)
            jobs, dependencies = parse_file(instance_url)
            
            scheduler = Scheduler(jobs, dependencies, pdr=pdrs['mwkr'])
            makespan = scheduler.schedule()
            end = time.time()
            print(f"Instance {instance} with PDR {args.pdr} has makespan {makespan}, ", end - start)
            result.append([instance, makespan, end - start])
            
        df_results = pd.DataFrame(result, columns=['Instance', 'Makespan', 'solve_time'])
        df_results.to_csv('baseline/huristic/{}.csv'.format(pdr), index=False)