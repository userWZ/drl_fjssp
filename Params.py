import argparse
import datetime
import os
import json
    
parser = argparse.ArgumentParser(description="Arguments for ppo_fjssp")
# args for device
parser.add_argument("--device", type=str, default="cuda:0,1,2,3", help="Number of jobs of instances")
# args for env
parser.add_argument("--n_j", type=int, default=30, help="Number of jobs of instance")
parser.add_argument("--n_m", type=int, default=20, help="Number of machines instance")
parser.add_argument("--rewardscale", type=float, default=0.0, help="Reward scale for positive rewards")
parser.add_argument(
    "--init_quality_flag", type=bool, default=False, help="Flag of whether init state quality is 0, True for 0"
)
parser.add_argument("--low", type=int, default=1, help="LB of duration")
parser.add_argument("--high", type=int, default=99, help="UB of duration")
parser.add_argument("--np_seed_train", type=int, default=200, help="Seed for numpy for training")
parser.add_argument("--np_seed_validation", type=int, default=200, help="Seed for numpy for validation")
parser.add_argument("--torch_seed", type=int, default=600, help="Seed for torch")
parser.add_argument(
    "--et_normalize_coef",
    type=int,
    default=1000,
    help="Normalizing constant for feature LBs (end time), normalization way: fea/constant",
)
parser.add_argument(
    "--wkr_normalize_coef", type=int, default=100, help="Normalizing constant for wkr, normalization way: fea/constant"
)
# args for network
parser.add_argument(
    "--num_layers", type=int, default=3, help="No. of layers of feature extraction GNN including input layer"
)
parser.add_argument("--neighbor_pooling_type", type=str, default="sum", help="neighbour pooling type")
parser.add_argument("--graph_pool_type", type=str, default="average", help="graph pooling type")
parser.add_argument("--input_dim", type=int, default=4, help="number of dimension of raw node features")
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dim of MLP in fea extract GNN")
parser.add_argument(
    "--num_mlp_layers_feature_extract", type=int, default=2, help="No. of layers of MLP in fea extract GNN"
)
parser.add_argument("--num_mlp_layers_actor", type=int, default=2, help="No. of layers in actor MLP")
parser.add_argument("--hidden_dim_actor", type=int, default=32, help="hidden dim of MLP in actor")
parser.add_argument("--num_mlp_layers_critic", type=int, default=2, help="No. of layers in critic MLP")
parser.add_argument("--hidden_dim_critic", type=int, default=32, help="hidden dim of MLP in critic")
# args for PPO
parser.add_argument("--num_envs", type=int, default=1, help="No. of envs for training")
parser.add_argument("--max_updates", type=int, default=100000, help="No. of episodes of each env for training")
parser.add_argument("--lr", type=float, default=2e-5, help="lr")
parser.add_argument("--decayflag", type=bool, default=False, help="lr decayflag")
parser.add_argument("--decay_step_size", type=int, default=2000, help="decay_step_size")
parser.add_argument("--decay_ratio", type=float, default=0.9, help="decay_ratio, e.g. 0.9, 0.95")
parser.add_argument("--gamma", type=float, default=1, help="discount factor")
parser.add_argument("--k_epochs", type=int, default=1, help="update policy for K epochs")
parser.add_argument("--eps_clip", type=float, default=0.2, help="clip parameter for PPO")
parser.add_argument("--vloss_coef", type=float, default=1, help="critic loss coefficient")
parser.add_argument("--ploss_coef", type=float, default=2, help="policy loss coefficient")
parser.add_argument("--entloss_coef", type=float, default=0.01, help="entropy loss coefficient")

# args for training
parser.add_argument("--test", action="store_true", default=False, help="是否执行测试，否-训练")
parser.add_argument("--output", type=str, default="output/", help="root path of output dir")
parser.add_argument("--model_dir", type=str, default="model", help="folder path to save/load neural network models")
parser.add_argument("--continue_model_path", type=str, default=None, help="path of model to continue training")
# parser.add_argument("--continue_model_path", type=str, default=".\\output\\j30_m20_seed600\\2024-05-25-19-43-08\\model\\episode_0.pth", help="path of model to continue training")
parser.add_argument("--val_frequency", type=int, default=100, help="frequency for validation")
parser.add_argument("--save_frequency", type=int, default=1, help="frequency for save")
parser.add_argument("--log_dir", type=str, default="runs/", help="root path of log dir")
parser.add_argument("--instance_nums", type=int, default=50, help="number of instances for validation")
parser.add_argument("--output_prefix", type=str, default='', help="prefix of output dir")
parser.add_argument('--instance', type=str, default='visualization/data/instances')
parser.add_argument('--eval_model_path', type=str, default=None)
configs = parser.parse_args()

run_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


if configs.continue_model_path is not None:
    if not os.path.exists(configs.continue_model_path):
        raise ValueError("continue_model_path is not valid")
    
    model_dir = os.path.dirname(configs.continue_model_path)
    base_dir = os.path.dirname(model_dir)
    config_path = os.path.join(base_dir, "config.json")
    configs.continued = True
    # 从 config.json 文件中读取超参数
    with open(config_path, 'r') as f:
        config = json.load(f)
    run_time = os.path.basename(base_dir) + os.path.basename(configs.continue_model_path).replace(".pth", "") +  "_continued"
    

output_prefix = "j{}_m{}_seed{}".format(configs.n_j, configs.n_m, configs.torch_seed)
configs.output = os.path.join(configs.output, output_prefix, run_time)
configs.output_prefix = output_prefix
configs.log_dir = os.path.join(configs.log_dir, output_prefix, run_time)