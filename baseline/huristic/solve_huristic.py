import matplotlib.pyplot as plt
from job_shop_lib.visualization import plot_gantt_chart
from job_shop_lib import JobShopInstance, Operation
from job_shop_lib.dispatching import (
    DispatchingRuleSolver,
    DispatchingRule,
    Dispatcher
)
import time 
import re
import pandas as pd
import os
from job_shop_lib.benchmarking import load_benchmark_instance
import argparse
import numpy as np
plt.style.use("ggplot")


class fuzzyOperation(Operation):
    def __init__(self, machine, duration, fuzzy_duration: tuple[int]):
        super().__init__(machine, duration)
        self.fuzzy_duration = fuzzy_duration

def parse_file(dataset_url: str) -> JobShopInstance:
    with open(dataset_url, 'r') as file:
            lines = file.readlines()
            
    # 获取作业数和机器数        
    num_jobs = int(lines[1])
    num_machine = int(lines[3])

    machines = []
    for i in range(5, 5 + num_jobs):
        machines.append(list(map(int, lines[i].strip().split())))

    fuzzy_durations = []
    for i in range(6 + num_jobs, 6 + 2 * num_jobs):
            # 移除行中所有的括号
            matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', lines[i])
            durations = [list(map(int, match)) for match in matches]
            fuzzy_durations.append(durations)

    jobs = []
    for i in range(num_jobs):
        fuzzyOperations = []
        for j in range(num_machine):
            expect_duration = (sum(fuzzy_durations[i][j]) + fuzzy_durations[i][j][1] ) / 4
            operation = fuzzyOperation(machine=machines[i][j], duration=expect_duration, fuzzy_duration=fuzzy_durations[i][j])
            fuzzyOperations.append(operation)
        jobs.append(fuzzyOperations)

    return jobs


def longest_processing_time_rule(dispatcher: Dispatcher) -> Operation:
    """Dispatches the operation with the longest duration."""
    return max(
        dispatcher.available_operations(),
        key=lambda operation: operation.duration,
    )

def last_in_first_out_rule(dispatcher: Dispatcher) -> Operation:
    """Dispatches the last arrived operation."""
    return max(
        dispatcher.available_operations(),
        key=lambda operation: operation.position_in_job,
    )


def least_operations_remaining_rule(dispatcher: Dispatcher) -> Operation:
    """Dispatches the operation which job has the least remaining operations."""
    job_remaining_operations = [0] * dispatcher.instance.num_jobs
    for operation in dispatcher.uncompleted_operations():
        job_remaining_operations[operation.job_id] += 1

    return min(
        dispatcher.available_operations(),
        key=lambda operation: job_remaining_operations[operation.job_id],
    )
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_type', type=str, default='synthetic')
    parser.add_argument('--instance', type=str, default='instances')
    parser.add_argument('--instance_nums', type=int, default=50)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--n_j', type=int, default=6)
    parser.add_argument('--n_m', type=int, default=6)
    # parser.add_argument('--pdr', type=str, default='mwkr')
    args = parser.parse_args()
    instance_type = args.instance_type
    instance_path = os.path.join(args.instance, instance_type)
   
    pdrs = {
        'spt': 'shortest_processing_time',
        # 'mwkr': 'most_work_remaining',
        # 'mopr': 'most_operations_remaining',
        # 'lpt': 'longest_processing_time',
        # 'lifo': 'last_in_first_out',
        # 'lor': 'least_operations_remaining',
        # 'fifo': 'first_come_first_served',
    }
   
    if instance_type == 'synthetic':
        instance_file = os.path.join(instance_path, f"synthetic_{args.n_j}_{args.n_m}_instanceNums{args.instance_nums}_Seed{args.seed}.npy")
        instances = np.load(instance_file)
        df_results = pd.DataFrame()
        df_results['Instance'] = list(range(len(instances))) + ['Average']
        for key, value in pdrs.items():
            sovle_time = []
            makespan = []
            for i, data in enumerate(instances):
                start = time.time()
                data = data.astype(int)
                left, peak, right, machine = data
                jobs = []
                task_id = 1
                
                for i in range(len(machine)):
                    fuzzyOperations = []
                    for j in range(len(machine[i])):
                        expect_duration = (left[i][j] + 2 * peak[i][j] + right[i][j]) / 4
                        operation = fuzzyOperation(machine=int(machine[i][j]), duration=expect_duration, fuzzy_duration=[left[i][j], peak[i][j], right[i][j]])
                        fuzzyOperations.append(operation)
                    jobs.append(fuzzyOperations)
                
                jsp_instance = JobShopInstance(jobs, name=i)
                solver = DispatchingRuleSolver(dispatching_rule=value)
                schedule = solver(jsp_instance)
                end = time.time()
                makespan.append(schedule.makespan())
                sovle_time.append(end - start)
                
            df_results[key] = makespan + [np.mean(makespan)]
            df_results[key + '_time'] = sovle_time + [np.mean(sovle_time)]
        df_results.to_csv('baseline/huristic/pdr_synthetic_j{}_n{}.csv'.format(args.n_j, args.n_m), index=False)    
        print("synthetic instance n_j:{} n_m:{} has been solved".format(args.n_j, args.n_m))
    else:

        instances = os.listdir(instance_path)
        df_results = pd.DataFrame()
        df_results['Instance'] = instances
        for key, value in pdrs.items():
            sovle_time = []
            makespan = []
            
            for instance in instances:
                
                start_time = time.time()
                instance_url = os.path.join(instance_path, instance)
                jobs = parse_file(instance_url)
                jsp_instnce = JobShopInstance(jobs, name=instance)
        
                if value == 'least_operations_remaining':
                    solver = DispatchingRuleSolver(dispatching_rule=least_operations_remaining_rule)
                elif value == 'longest_processing_time':
                    solver = DispatchingRuleSolver(dispatching_rule=longest_processing_time_rule)
                elif value == 'last_in_first_out':
                    solver = DispatchingRuleSolver(dispatching_rule=last_in_first_out_rule)
                else:
                    solver = DispatchingRuleSolver(dispatching_rule=value)
                    
                schedule = solver(jsp_instnce)
                end_time = time.time()
                mk = schedule.makespan()
                makespan.append(mk)
                sovle_time.append(end_time - start_time)   
                 
                print(f"Instance {instance.name} has makespan {mk}, ", end_time - start_time)
            df_results[f'{key}_makespan'] = makespan
            df_results[f'{key}_solve_time'] = sovle_time
        df_results.to_csv('baseline/huristic/pdr_benchmarks.csv', index=False)
       