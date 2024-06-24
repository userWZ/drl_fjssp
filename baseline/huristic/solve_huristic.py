import matplotlib.pyplot as plt
from job_shop_lib.visualization import plot_gantt_chart
from job_shop_lib import JobShopInstance, Operation
from job_shop_lib.dispatching import (
    DispatchingRuleSolver,
    DispatchingRule,
)
import re
import pandas as pd
import os
from job_shop_lib.benchmarking import load_benchmark_instance

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
            expect_duration = (sum(fuzzy_durations[i][j]) + fuzzy_durations[i][j][1] )// 4
            operation = fuzzyOperation(machine=machines[i][j], duration=expect_duration, fuzzy_duration=fuzzy_durations[i][j])
            fuzzyOperations.append(operation)
        jobs.append(fuzzyOperations)

    return jobs


if __name__ == "__main__":

    instances = os.listdir("instances")
    result = []
    # SHORTEST_PROCESSING_TIME = "shortest_processing_time"
    # FIRST_COME_FIRST_SERVED = "first_come_first_served"
    # MOST_WORK_REMAINING = "most_work_remaining"
    # MOST_OPERATIONS_REMAINING = "most_operations_remaining"
    # RANDOM = "random"
    
    DispatchingRule = 'MOST_OPERATIONS_REMAINING'
    for instance in instances:
        instance_url = os.path.join("instances", instance)
        jobs = parse_file(instance_url)
        instance = JobShopInstance(jobs, name=instance)
        solver = DispatchingRuleSolver(dispatching_rule=DispatchingRule)
        schedule = solver(instance)
        makespan = schedule.makespan()
        result.append([instance, makespan])
        print(f"Instance {instance.name} has makespan {makespan}")

    df_results = pd.DataFrame(result, columns=['Instance', 'Makespan'])
    df_results.to_csv('baseline/huristic/{}.csv'.format(DispatchingRule), index=False)