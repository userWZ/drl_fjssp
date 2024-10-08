from ortools.sat.python import cp_model
import pandas as pd
import sys
import os
# sys.path.append('DRL_FJSSP')
from utils import read_dataset
# from visualization.visual import draw_fuzzy_gantt_from_df
import time
import numpy as np
import argparse
time_limit_seconds = 600

def expected_makespan(left, peak, right):
    return (2 * peak + left + right) / 4 

def create_cp_model(num_machines, jobs_data):
    model = cp_model.CpModel()

    # 定义问题参数
    horizon = sum([sum([pt[2] for _, pt in job]) for job in jobs_data])

    # 定义变量
    start_times = {}
    end_times = {}
    intervals = {}
    makespan = model.NewIntVar(0, horizon, 'makespan')

    for job_id, job in enumerate(jobs_data):
        for task_id, (machine, (p1, p2, p3)) in enumerate(job):
            for l in range(3):
                start_times[(job_id, task_id, l)] = model.NewIntVar(0, horizon, f'start_{job_id}_{task_id}_{l}')
                end_times[(job_id, task_id, l)] = model.NewIntVar(0, horizon, f'end_{job_id}_{task_id}_{l}')
                intervals[(job_id, task_id, l)] = model.NewIntervalVar(
                    start_times[(job_id, task_id, l)],
                    [p1, p2, p3][l],
                    end_times[(job_id, task_id, l)],
                    f'interval_{job_id}_{task_id}_{l}'
                )

    # 添加约束
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            for l in range(3):
                model.Add(start_times[(job_id, task_id + 1, l)] >= end_times[(job_id, task_id, l)])

    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job)):
            for l in range(2):  # 确保 l+1 不超出范围
                model.Add(start_times[(job_id, task_id, l)] <= start_times[(job_id, task_id, l + 1)])

    for machine in range(num_machines):
        machine_tasks = {l: [] for l in range(3)}
        for job_id, job in enumerate(jobs_data):
            for task_id, (m, _) in enumerate(job):
                if m == machine:
                    for l in range(3):
                        machine_tasks[l].append((start_times[(job_id, task_id, l)], intervals[(job_id, task_id, l)]))

        for l in range(3):
            model.AddNoOverlap([interval for _, interval in machine_tasks[l]])

        for i in range(len(machine_tasks[1])):
            model.Add(machine_tasks[0][i][0] <= machine_tasks[1][i][0])
            model.Add(machine_tasks[1][i][0] <= machine_tasks[2][i][0])

    for job_id, job in enumerate(jobs_data):
        for l in range(3):
            model.Add(makespan >= end_times[(job_id, len(job) - 1, l)])

    model.Minimize(makespan)

    return model, makespan, start_times, end_times, jobs_data

def solve_cp_model(num_machines, jobs_data):
    model, makespan, start_times, end_times, jobs_data = create_cp_model(num_machines, jobs_data)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    
    status = solver.Solve(model)
    print(f'Status: {solver.StatusName(status)}')
    min_makespan = -1
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        
        # 创建DataFrame
        df_gantt = pd.DataFrame(columns=["Job", "Operation", "Machine",
                                         "start_left", "start_peak", "start_right",
                                         "end_left", "end_peak", "end_right"])

        for job_id, job in enumerate(jobs_data):
            for task_id, (machine, (p1, p2, p3)) in enumerate(job):
                start_left = solver.Value(start_times[(job_id, task_id, 0)])
                start_peak = solver.Value(start_times[(job_id, task_id, 1)])
                start_right = solver.Value(start_times[(job_id, task_id, 2)])
                end_left = solver.Value(end_times[(job_id, task_id, 0)])
                end_peak = solver.Value(end_times[(job_id, task_id, 1)])
                end_right = solver.Value(end_times[(job_id, task_id, 2)])
                new_row = pd.DataFrame({
                    "Job": [job_id],
                    "Operation": [task_id],
                    "Machine": [machine],
                    "start_left": [start_left],
                    "start_peak": [start_peak],
                    "start_right": [start_right],
                    "end_left": [end_left],
                    "end_peak": [end_peak],
                    "end_right": [end_right]
                })
                df_gantt = pd.concat([df_gantt, new_row], ignore_index=True)
        df_gantt['expected_start'] = df_gantt.apply(lambda row: expected_makespan(row['start_left'], row['start_peak'], row['start_right']), axis=1)
        df_gantt['expected_end'] = df_gantt.apply(lambda row: expected_makespan(row['end_left'], row['end_peak'], row['end_right']), axis=1)
        min_makespan = df_gantt['expected_end'].max()
        print(f'Minimal makespan: {min_makespan}')
        # print(max_expected_end)
    else:
        print('No solution found.')
        
        
    return min_makespan, df_gantt, solver.StatusName(status)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, default='instances/benchmarks')
    parser.add_argument('--instance', type=str, default='instances')
    parser.add_argument('--instance_nums', type=int, default=50)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--n_j', type=int, default=50)
    parser.add_argument('--n_m', type=int, default=20)
    parser.add_argument('--single_instance', type=int, default=140)
    args = parser.parse_args()
    instance_path = args.instance_path
    
    instance_type = 'synthetic' if 'synthetic' in instance_path else 'benchmark'
    
    if instance_type == 'synthetic':
        save_path = f'baseline/or-tools/or-tools-synthetic_j{args.n_j}_m{args.n_m}.csv'
    else:
        save_path = 'baseline/or-tools/or-tools-benchmarks.csv'
    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path)
    else:
        df_results = pd.DataFrame(columns=['Instance', 'Makespan', 'all_time', 'solve_time', 'solve_status'])
        
    if instance_type == 'synthetic':
        instance_file = os.path.join(args.instance_path, f"synthetic_{args.n_j}_{args.n_m}_instanceNums{args.instance_nums}_Seed{args.seed}.npy")
        instances = np.load(instance_file)
        for i, data in enumerate(instances):
            # print(save_path)
            if df_results['Instance'].isin([i]).any():
                print(f"Instance {i} already solved")
                continue
            print("solve synthetic instance {}".format(i))
            start_time = time.time()
            data = data.astype(int)
            left, peak, right, machine = data
            num_machines = args.n_m
            num_jobs = args.n_j
            jobs_data = [[(machine[i][j], (left[i][j], peak[i][j], right[i][j])) for j in range(len(machine[i]))] for i in range(len(machine))]
            read_time = time.time()
            
            makespan, resolution, solve_status = solve_cp_model(num_machines, jobs_data)
            end_time = time.time()
            result = [i, makespan, end_time - start_time, end_time - read_time,solve_status]
            print(f"Instance {i} has makespan {makespan}")
            temp_df = pd.DataFrame([result], columns=['Instance', 'Makespan', 'all_time', 'solve_time', 'solve_status'])
            df_results = pd.concat([df_results, temp_df], ignore_index=True)
            df_results.to_csv(f'baseline/or-tools/or-tools-synthetic_j{args.n_j}_m{args.n_m}.csv', index=False)  
    else: # benchmark
        print(instance_path)
        instances = os.listdir(instance_path)
        if args.single_instance:
            instance = instances[args.single_instance]
            start_time = time.time()
            instance_path = os.path.join(instance_path, instance)
            print(f"Processing instance {instance}")
            # 读取数据集
            num_jobs, num_machines, processing_time = read_dataset(instance_path)
            
            # 处理数据集
            machine = [[item[0] for item in sublist] for sublist in processing_time]
            op = [[item[1:] for item in sublist] for sublist in processing_time]
            jobs_data = [[(machine[i][j], op[i][j]) for j in range(len(machine[i]))] for i in range(len(machine))]
            read_time = time.time()
            # 求解
            makespan, resolution, solve_status = solve_cp_model(num_machines, jobs_data)
            end_time = time.time()
            result = [instance, makespan, end_time - start_time, end_time - read_time, solve_status]
            print(result)
            print(f"Instance {instance} has makespan {makespan} and solve status {solve_status} in {end_time - start_time} seconds with {end_time - read_time} seconds for solving")
        else:
            for instance in instances:
                if df_results['Instance'].isin([instance]).any():
                    print(f"Instance {instance} already solved")
                    continue
                start_time = time.time()
                instance_path = os.path.join(instance_path, instance)
                print(f"Processing instance {instance}")
                # 读取数据集
                num_jobs, num_machines, processing_time = read_dataset(instance_path)
                
                # 处理数据集
                machine = [[item[0] for item in sublist] for sublist in processing_time]
                op = [[item[1:] for item in sublist] for sublist in processing_time]
                jobs_data = [[(machine[i][j], op[i][j]) for j in range(len(machine[i]))] for i in range(len(machine))]
                read_time = time.time()
                # 求解
                makespan, resolution, solve_status = solve_cp_model(num_machines, jobs_data)
                end_time = time.time()
                result = [instance, makespan, end_time - start_time, end_time - read_time, solve_status]
                # print(f"Instance {instance} has makespan {makespan}")
                # print(makespan)
                # if num_machines > 10:
                #     print('[Render faild]: OUT of color bound, instance have too many jobs.')
                #     # sys.exit(0)
                # # 绘制甘特图
                # draw_fuzzy_gantt_from_df(resolution, num_machines)

                
                temp_df = pd.DataFrame([result], columns=['Instance', 'Makespan', 'all_time', 'solve_time', 'solve_status'])
                df_results = pd.concat([df_results, temp_df], ignore_index=True)
                df_results.to_csv('baseline/or-tools/or-tools-benchmarks.csv', index=False)
