import os.path
import torch
import numpy as np
import sys

sys.path.append(".")
from env.fjssp_env import FjsspEnv
from Params import configs
import global_util
from models.actor_critic import ActorCritic
from runner import Runner
from jssp_tool.env.util import set_random_seed
from jssp_tool.rl.agent.ppo.ppo_discrete import PPODiscrete
from env.utils import gen_and_save
import json

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


def main():
    set_random_seed(configs.torch_seed)

    env = FjsspEnv(configs.n_j, configs.n_m, configs.low, configs.high, configs.device)
    
    
    vali_data_dir =  os.path.join(global_util.get_project_root(), "data", "generatedData{}_{}_instanceNums{}_Seed{}.npy").format(
            configs.n_j, configs.n_m, configs.instance_nums,configs.np_seed_validation
        )
    if not os.path.exists(vali_data_dir):
        gen_and_save(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high,
                batch_size=configs.instance_nums, seed=configs.np_seed_train)
        vali_data = np.load(vali_data_dir)
    else:
        vali_data = np.load(vali_data_dir)

    model = ActorCritic(
        # job number
        n_j=configs.n_j,
        # machine number
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

    # configs.output = os.path.join(
    #     configs.output, "j{}_m{}_l{}_h{}".format(configs.n_j, configs.n_m, configs.low, configs.high)
    # )
    runner = Runner(configs, env, vali_data)
    if configs.test:
        # 加载模型
        ppo.policy.load_state_dict(torch.load(os.path.join(configs.output, "best.pth"), configs.device), False)
        # 测试
        runner.test(vali_data, ppo, float("inf"), 0, phase="test")
    else:
        configs.device = str(configs.device)
        with open(os.path.join(configs.output, "config.json"), 'w') as f:
            json.dump(vars(configs), f, indent=4)
        runner.train(ppo)


if __name__ == "__main__":
    main()
