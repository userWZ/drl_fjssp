import heapq
from collections import deque

class FuzzyNumber:
    def __init__(self, low, mid, high):
        self.low = int(low)
        self.mid = int(mid)
        self.high = int(high)

    def max(self, other):
        return FuzzyNumber(
            max(self.low, other.low),
            max(self.mid, other.mid),
            max(self.high, other.high)
        )

    def __add__(self, other):
        return FuzzyNumber(
            self.low + other.low,
            self.mid + other.mid,
            self.high + other.high
        )

class Scheduler:
    def __init__(self, jobs, dependencies, pdr):
        self.jobs = jobs
        self.dependencies = dependencies
        self.pdr = pdr
        self.in_degree = self.calculate_in_degree()
        self.task_start_times = {task: FuzzyNumber(0, 0, 0) for task in self.get_all_tasks()}
        self.task_end_times = {}  # 用于记录任务的结束时间
        self.task_map = self.create_task_map()
        self.machine_count = self.get_machine_count()
        self.arrival_order = deque(self.get_all_tasks())  # FIFO and LIFO support

    def get_all_tasks(self):
        all_tasks = set()
        for job in self.jobs.values():
            for task in job["tasks"]:
                all_tasks.add(task["id"])
        return all_tasks

    def create_task_map(self):
        task_map = {}
        for job_id, job in self.jobs.items():
            for task in job["tasks"]:
                task_map[task["id"]] = (job_id, task)
        return task_map

    def get_task_details(self, task_id):
        return self.task_map.get(task_id, (None, None))

    def calculate_in_degree(self):
        in_degree = {task: 0 for task in self.get_all_tasks()}
        for task, dependents in self.dependencies.items():
            if task not in in_degree:
                in_degree[task] = 0
            for dependent in dependents:
                if dependent not in in_degree:
                    in_degree[dependent] = 0
                in_degree[dependent] += 1
        return in_degree

    def get_machine_count(self):
        max_machine_index = 0
        for job in self.jobs.values():
            for task in job["tasks"]:
                if task["machine"] > max_machine_index:
                    max_machine_index = task["machine"]
        return max_machine_index + 1

    def weighted_average(self, fuzzy_number):
        return (fuzzy_number.low + 2 * fuzzy_number.mid + fuzzy_number.high) / 4

    def schedule(self):
        queue = [task for task in self.dependencies if self.in_degree[task] == 0]
        heapq.heapify(queue)
        topo_order = []

        while queue:
            current_tasks = list(queue)
            queue.clear()

            sorted_tasks = self.sort_tasks(current_tasks)

            for current in sorted_tasks:
                topo_order.append(current)
                for neighbor in self.dependencies[current]:
                    self.in_degree[neighbor] -= 1
                    if self.in_degree[neighbor] == 0:
                        heapq.heappush(queue, neighbor)

        # 初始化机器状态
        machines = [FuzzyNumber(0, 0, 0) for _ in range(self.machine_count)]

        # 调度
        for task in topo_order:
            job_id, task_detail = self.get_task_details(task)
            machine_index = task_detail["machine"]
            processing_time = task_detail["processing_time"]

            # 验证机器索引
            if machine_index >= self.machine_count:
                raise ValueError(f"Machine index {machine_index} out of range for task {task}")

            # 计算任务的开始时间
            start_time = machines[machine_index].max(self.task_start_times[task])
            end_time = start_time + processing_time

            # 更新机器状态
            machines[machine_index] = end_time

            # 记录任务的结束时间
            self.task_end_times[task] = end_time

            # 更新后续任务的开始时间
            for neighbor in self.dependencies[task]:
                self.task_start_times[neighbor] = self.task_start_times[neighbor].max(end_time)

            # print(f"Task {task} (Job {job_id}) assigned to machine {machine_index} from {start_time} to {end_time}")

        # 计算 makespan
        makespan = self.calculate_makespan()
        # print(f"Final machine states: {machines}")
        print(f"Makespan: {makespan}")
        return makespan

    def calculate_makespan(self):
        # 找到所有任务的结束时间中的最大值
        final_end_times = [self.task_end_times[task] for task in self.task_end_times]
        max_end_time = max(final_end_times, key=self.weighted_average)
        return self.weighted_average(max_end_time)

    def sort_tasks(self, tasks):
        if self.pdr == 'shortest_processing_time':
            return sorted(tasks, key=lambda task: self.weighted_average(self.get_task_details(task)[1]["processing_time"]))
        elif self.pdr == 'most_work_remaining':
            return sorted(tasks, key=lambda task: -sum(self.weighted_average(t["processing_time"]) for t in self.jobs[self.get_task_details(task)[0]]["tasks"]))
        elif self.pdr == 'most_operations_remaining':
            return sorted(tasks, key=lambda task: -len(self.jobs[self.get_task_details(task)[0]]["tasks"]))
        elif self.pdr == 'longest_processing_time':
            return sorted(tasks, key=lambda task: -self.weighted_average(self.get_task_details(task)[1]["processing_time"]))
        elif self.pdr == 'last_in_first_out':
            return list(reversed(self.arrival_order))
        elif self.pdr == 'least_operations_remaining':
            return sorted(tasks, key=lambda task: len(self.jobs[self.get_task_details(task)[0]]["tasks"]))
        elif self.pdr == 'first_come_first_served':
            return list(self.arrival_order)
        else:
            return tasks  # 默认不排序

# # 示例用法
# # Parse input
# jobs, dependencies = parse_input(input_str)

# # Create and run scheduler
# scheduler = Scheduler(jobs, dependencies, "shortest_processing_time")
# scheduler.schedule()
