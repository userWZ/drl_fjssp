import re
from collections import defaultdict
from scheduler import Scheduler, FuzzyNumber
import time
import os
import pandas as pd

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

    dependencies = defaultdict(list)
    for job_id in range(1, num_jobs + 1):
        tasks = jobs[job_id]["tasks"]
        for i in range(len(tasks) - 1):
            dependencies[tasks[i]["id"]].append(tasks[i + 1]["id"])

    return jobs, dependencies


if __name__ == '__main__':
    
    instances = os.listdir("instances")
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
    pdr = 'mopr'
    for instance in instances:
        start = time.time()
        print(f"Processing {instance}")
        instance_url = os.path.join("instances", instance)
        jobs, dependencies = parse_file(instance_url)
        
        scheduler = Scheduler(jobs, dependencies, pdr=pdrs['mwkr'])
        makespan = scheduler.schedule()
        end = time.time()
        print(f"Instance {instance} with PDR {pdr} has makespan {makespan}, ", end - start)
        result.append([instance, makespan, end - start])
        
    df_results = pd.DataFrame(result, columns=['Instance', 'Makespan', 'solve_time'])
    df_results.to_csv('baseline/huristic/{}.csv'.format(pdr), index=False)