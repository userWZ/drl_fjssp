import matplotlib.pyplot as plt
import matplotlib.colors as mc
from utils import get_schedule
        
def draw_fuzzy_gantt(machine_list, machine_nb):
        plt.figure(figsize=(12, 8))
        colors = list(mc.TABLEAU_COLORS.keys())
        for machine_idx, machine_schedule in machine_list.items():
            y = machine_idx
            plt.axhline(y, color='black')  # 绘制水平线，表示机器
            plt.yticks(list(range(machine_nb)))  # 设置y轴刻度
            for task in machine_schedule:
                k, start, end, o = task['job_idx'], task['op_start_time'], task['op_end_time'], task['task_idx']
                if start == [0, 0, 0]:
                    plt.scatter(0, y, color=colors[k])  # 绘制起始点
                    plt.text(start[1], y-0.2, str(k)+','+str(o), verticalalignment='center', horizontalalignment='center',
                             fontsize=6)  # 在起始点添加文本标签
                else:
                    triangleX = start
                    triangleY = [y, y-0.2, y]
                    plt.fill(triangleX, triangleY, colors[k])  # 绘制起始时间的三角形
                    plt.text(start[1], y-0.3, str(k)+','+str(o), verticalalignment='center', horizontalalignment='center',
                             fontsize=6)  # 在起始时间的三角形上添加文本标签
                triangleX = end
                triangleY = [y, y+0.2, y]
                plt.fill(triangleX, triangleY, colors[k])  # 绘制结束时间的三角形
                plt.text(end[1], y+0.3, str(k)+','+str(o), verticalalignment='center', horizontalalignment='center', fontsize=6)  # 在结束时间的三角形上添加文本标签
        plt.show()


if __name__ == '__main__':
    machine_schedule = get_schedule("data/gao10_10_1.txt","solutions/sol10_10_1.txt")
    draw_fuzzy_gantt(machine_schedule, len(machine_schedule))