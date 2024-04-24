import numpy as np
import os

def get_project_root() -> str:
    """
    Get the absolute path of the project root

    Returns
    -------

    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.exists(os.path.join(current_dir, "README.md")):
        # 如果当前目录下不存在 README.md 文件，则继续向上一级目录查找
        current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        # 到达文件系统的根目录时停止搜索
        if current_dir == os.path.abspath(os.sep):
            raise FileNotFoundError("Project root not found.")
    return current_dir
    # return os.path.join(os.path.dirname(__file__), ".")


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
            left = np.random.randint(0, 10)
            peak = np.random.randint(left, high)
            right = np.random.randint(0, 10)
            # processing_time[job, machine] = [left, peak, right]
            processing_time['left'][job, machine] = peak - left if peak - left >= 1 else peak
            processing_time['peak'][job, machine] = peak
            processing_time['right'][job, machine] = peak + right if peak + right <= 99 else peak

    # 机器编号，从0开始, shape (n_j,n_m), 每一行代表一个job的机器处理顺序
    task_machines = np.expand_dims(np.arange(0, n_m), axis=0).repeat(repeats=n_j, axis=0)
    task_machines = _permute_rows(task_machines)
    return processing_time['left'], processing_time['peak'], processing_time['right'], task_machines


def _permute_rows(x: np.ndarray):
    """
    打乱每个job的machine处理顺序
    Args:
        x (np.ndarray): shape (n_j,n_m)
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]



def gen_and_save(n_j=6, n_m=6, low=1, high=99, batch_size=100, seed=200):
    np.random.seed(seed)
    data = np.array([gen_instance_triangle(n_j=n_j, n_m=n_m, low=low, high=high) for _ in range(batch_size)])
    print(data.shape)
    folder = os.path.join(get_project_root(), "data")
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "generatedData{}_{}_instanceNums{}_Seed{}.npy".format(n_j, n_m, batch_size, seed)), data)


if __name__ == '__main__':
    gen_and_save(n_j=15,
                 n_m=10,
                 low=1,
                 high=99,
                 batch_size=100,
                 seed=200)