import pandas as pd


def to_dataframe(task_durations, task_machines, task_finish_times):
    df_gantt = pd.DataFrame(columns=["Job", "Operation", "Machine",
                                     "start_time_left", "start_time_peak", "start_time_right"
                                     "end_time_left", "end_time_peak", "end_time_right"])
    
    for i in range(len(task_durations["left"])):
        for j in range(len(task_durations["left"][i])):
            df = pd.DataFrame(
                {
                    "Job": [i],
                    "Operation": [j],
                    "Machine": [task_machines[i][j]],
                    "start_left": [task_finish_times["left"][i][j] - task_durations["left"][i][j]],
                    "start_peak": [task_finish_times["peak"][i][j] - task_durations["peak"][i][j]],
                    "start_right": [task_finish_times["right"][i][j] - task_durations["right"][i][j]],
                    "end_left": [task_finish_times["left"][i][j]],
                    "end_peak": [task_finish_times["peak"][i][j]],
                    "end_right": [task_finish_times["right"][i][j]],
                }
            )
            
            df_gantt = pd.concat([df_gantt if not df_gantt.empty else None, df])
    df_gantt = df_gantt.reset_index(drop=True)
    return df_gantt
