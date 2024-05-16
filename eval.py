import os
import pickle
import random

import torch
import numpy as np
import global_util
from env.fjssp_env import FjsspEnv
from Params import configs
from visualization.visual import *
from visualization.utils import read_dataset
from result_generator import to_dataframe
from models.actor_critic import ActorCritic
from jssp_tool.rl.agent.ppo.ppo_discrete import PPODiscrete
import pandas as pd
import argparse

configs.device = torch.device(configs.device if torch.cuda.is_available() else "cpu")


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
    
def evaluation(dataset=r'visualization/data/instances/Ta61_F.txt', ppo=None, render=False):
    processing_time = read_dataset(dataset)
    # 假设你的三维列表名为three_dimensional_list
    # 创建四个空的二维列表，分别用于存放包含a、b、c、d的元素
    machine = np.array([[item[0] - 1 for item in sublist] for sublist in processing_time])
    op_left = np.array([[item[1] for item in sublist] for sublist in processing_time])
    op_peak = np.array([[item[2] for item in sublist] for sublist in processing_time])
    op_right = np.array([[item[3] for item in sublist] for sublist in processing_time])
    
    data = (op_left, op_peak, op_right, machine)
    env = FjsspEnv(configs.n_j, configs.n_m, configs.low, configs.high, configs.device)

    obs, _ = env.reset(data=data)
    reward = 0
    index = 0
    while True:
        index = index + 1
        obs = to_tensor(*obs)
        pi, value = ppo.policy_old(obs)
        action, a_idx, logprob = ppo.sample_action(pi, obs[2])
        # print(index, action.item())
        next_obs, reward, done, _, _ = env.step(action.item())
        obs = next_obs
        if done:
            break
    makespan = env.cur_make_span
    # print(makespan)
    if render:
        df_schedule = to_dataframe(env.task_durations, env.task_machines, env.low_bounds)
        draw_fuzzy_gantt_from_df(df_schedule, env.n_m)
    return makespan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default='visualization/data/instances')
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    
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
    
    ppo.policy.load_state_dict(torch.load(os.path.join(args.model_path, "best.pth"), configs.device), False)

    instances = os.listdir(args.instance)
    results = []
    for instance in instances:
        instance_path = os.path.join(args.instance, instance)
        makespan = evaluation(dataset=instance_path, ppo=ppo)
        results.append([instance, makespan])
        print(instance, makespan)

    df_results = pd.DataFrame(results, columns=['Instance', 'Makespan'])
    df_results.to_csv(os.path.join(args.model_path, 'results.csv'), index=False)







