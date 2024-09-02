import os
import pickle
import random

import torch
import time
import numpy as np
import global_util
from env.fjssp_env import FjsspEnv
from Params import setting_params
from visualization.visual import *
from visualization.utils import read_dataset
from result_generator import to_dataframe
from models.actor_critic import ActorCritic
from jssp_tool.rl.agent.ppo.ppo_discrete import PPODiscrete
import pandas as pd
import datetime
configs = setting_params()
configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_ppo(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    return PPODiscrete(
        model,
        configs.eps_clip,
        configs.k_epochs,
        optimizer,
        configs.ploss_coef,
        configs.vloss_coef,
        configs.entloss_coef,
        device=configs.device,
    )

def to_tensor(adj, fea, candidate, mask):
        adj_tensor = torch.from_numpy(np.copy(adj)).to(configs.device).to_sparse()
        fea_tensor = torch.from_numpy(np.copy(fea)).to(configs.device)
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(configs.device).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(configs.device).unsqueeze(0)
        return adj_tensor, fea_tensor, candidate_tensor, mask_tensor
    
def evaluation(data, ppo=None, render=True, save=True, sample_strategy='sample'):
    env = FjsspEnv(configs.n_j, configs.n_m, configs.low, configs.high, configs.device)
    
    obs, _ = env.reset(data=data)
    reward = 0
    index = 0
    start_time = time.time()
    while True:
        index = index + 1
        obs = to_tensor(*obs)
        pi, value = ppo.policy_old(obs)
        if sample_strategy == 'greedy':
            action = ppo.greedy_select_action(pi, obs[2])
        else:
            action, a_idx, logprob = ppo.sample_action(pi, obs[2])
        # print(index, action.item())
        next_obs, reward, done, _, _ = env.step(action.item())
        obs = next_obs
        if done:
            break
    end_time = time.time()
    makespan = env.cur_make_span
    # print(makespan)
    if render:
        if env.n_j > 10:
            print('[Render faild]: OUT of color bound, instance have too many jobs.')
            return makespan, end_time - start_time
        df_schedule = to_dataframe(env.task_durations, env.task_machines, env.low_bounds)
        if save: 
            save_url = os.path.join(configs.eval_save_path, instance + '.png')
            # save_url = instance
            draw_fuzzy_gantt_from_df(df_schedule, env.n_m, save_url)
        else:
            draw_fuzzy_gantt_from_df(df_schedule, env.n_m)
    return makespan, end_time - start_time


if __name__ == '__main__':
    sample_times = 10
    model = ActorCritic(
        n_j=configs.n_j,
        n_m=configs.n_m,
        num_layers=configs.num_layers,
        learn_eps=False,
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
        device=configs.device,
    )
    ppo = build_ppo(model)
    ppo.policy.load_state_dict(torch.load(configs.eval_model_path, configs.device), False)
    
    print('evaluation begin >>> model:{}, n_j: {}, n_m: {}, instance_type: {}'.format(configs.eval_model_path, 
                                                                                      configs.n_j, 
                                                                                      configs.n_m, 
                                                                                      configs.instance_type))
    
    results = []
    # 合成实例
    if configs.instance_type == 'synthetic':
        instance_file = os.path.join(configs.instance, 'synthetic', f"synthetic_{configs.n_j}_{configs.n_m}_instanceNums{configs.instance_nums}_Seed{configs.np_seed_validation}.npy")
        eval_instances = np.load(instance_file)
        for i, data in enumerate(eval_instances):
            makespan, solve_time = evaluation(data=data, ppo=ppo, render=configs.render, sample_strategy=configs.sample_strategy)
            results.append([i, makespan, solve_time])
            print("instances", i, makespan, solve_time)
        df_results = pd.DataFrame(results, columns=['Instance', 'Makespan', 'Time'])
        save_path = os.path.join(configs.eval_save_path, 'synthetic_model_j{j}_m{m}_{s}_results_{t}.csv'.format(j=configs.n_j, m=configs.n_m, s=configs.sample_strategy, t=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        df_results.to_csv(save_path, index=False)
            
    else:
        instances_path = os.path.join(configs.instance, 'benchmarks')
        instances = os.listdir(instances_path)
        for instance in instances:
            print('====%s eval begin===='%instance)
            # 读取实例
            instance_path = os.path.join(instances_path, instance)
            jobs, machines, processing_time = read_dataset(instance_path)
            # 假设你的三维列表名为three_dimensional_list
            # 创建四个空的二维列表，分别用于存放包含a、b、c、d的元素
            machine = np.array([[item[0] for item in sublist] for sublist in processing_time])
            op_left = np.array([[item[1] for item in sublist] for sublist in processing_time])
            op_peak = np.array([[item[2] for item in sublist] for sublist in processing_time])
            op_right = np.array([[item[3] for item in sublist] for sublist in processing_time])
            
            data = (op_left, op_peak, op_right, machine)
            
            if configs.sample_strategy == 'sample':
                sample_result = []
                for i in range(sample_times):
                    makespan, solve_time = evaluation(data=data, ppo=ppo, render=configs.render, sample_strategy=configs.sample_strategy)
                    sample_result.append([makespan, solve_time])
                best_result_index = np.argmin([item[0] for item in sample_result])
                makespan, solve_time = sample_result[best_result_index]
    
            else:
                makespan, solve_time = evaluation(data=data, ppo=ppo, render=configs.render, sample_strategy=configs.sample_strategy)

            results.append([instance, makespan, solve_time])
            print(instance, makespan, solve_time)
            
        df_results = pd.DataFrame(results, columns=['Instance', 'Makespan', 'Time'])
        df_results = df_results.sort_values(by='Instance', ascending=True)
        
        save_path = os.path.join(configs.eval_save_path, 'benchmarks_model_j{j}_m{m}_{s}_results_{t}.csv'.format(j=configs.n_j, m=configs.n_m, s=configs.sample_strategy, t=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        df_results.to_csv(save_path, index=False)






