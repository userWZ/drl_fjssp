import numpy as np

def gen_instance_uniformly(n_j, n_m, low, high):
    # 每个task的处理时长
    durations = np.random.randint(low=low, high=high, size=(n_j, n_m))
    # 机器编号，从0开始
    machines = np.expand_dims(np.arange(0, n_m), axis=0).repeat(repeats=n_j, axis=0)
    machines = _permute_rows(machines)
    return durations, machines


def gen_instance_triangle(n_j, n_m, low, high):
    """
    根据job数和machine数以及job的时间范围生成实例数据。
    Args:
        n_j：job数量
        n_m：machine数量
        low：随机处理时间下界
        high: 随机处理时间上界
    """
    # 每个task的操作时长的三角模糊数
    # processing_time = np.zeros((n_j, n_m, 3))
    processing_time = {
        'left': np.zeros((n_j, n_m)),
        'peak': np.zeros((n_j, n_m)),
        'right': np.zeros((n_j, n_m)),
    }

    for job in range(n_j):
        for machine in range(n_m):
            left = np.random.randint(low, high)
            peak = np.random.randint(left, high)
            right = np.random.randint(peak, high)
            # processing_time[job, machine] = [left, peak, right]
            processing_time['left'][job, machine] = left
            processing_time['peak'][job, machine] = peak
            processing_time['right'][job, machine] = right

    # 机器编号，从0开始, shape (n_j,n_m), 每一行代表一个job的机器处理顺序
    task_machines = np.expand_dims(np.arange(0, n_m), axis=0).repeat(repeats=n_j, axis=0)
    task_machines = _permute_rows(task_machines)
    return processing_time, task_machines


def _permute_rows(x: np.ndarray):
    """
    打乱每个job的machine处理顺序
    Args:
        x (np.ndarray): shape (n_j,n_m)
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


if __name__ == '__main__':
    durations, machines = gen_instance_uniformly(10, 10, 10, 20)
    print(durations)
    print(machines)