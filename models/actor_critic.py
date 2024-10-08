import torch.nn as nn

from models.graph_util import g_pool_cal
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn import GraphCNN
import torch


class ActorCritic(nn.Module):
    def __init__(
        self,
        n_j,
        n_m,
        # feature extraction net unique attributes:
        num_layers,
        learn_eps,
        input_dim,
        hidden_dim,
        # feature extraction net MLP attributes:
        num_mlp_layers_feature_extract,
        # actor net MLP attributes:
        num_mlp_layers_actor,
        hidden_dim_actor,
        # actor net MLP attributes:
        num_mlp_layers_critic,
        hidden_dim_critic,
        # actor/critic/feature_extraction shared attribute
        device,
        batch_size=None,
    ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        # self.n_j = n_j
        # machine size for problems, no business with network
        # self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device
        # if batch_size is None:
        #     batch_size = self.n_j * self.n_m

        self.feature_extract = GraphCNN(
            num_layers, num_mlp_layers_feature_extract, input_dim, hidden_dim, learn_eps, device
        ).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim * 2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

        # n_nodes = self.n_j * self.n_m
        # self.graph_pool_batch = g_pool_cal("average", (batch_size, n_nodes, n_nodes), n_nodes, device)
        # self.graph_pool_step = g_pool_cal("average", (1, n_nodes, n_nodes), n_nodes, device)

    def forward(self, obs):
        adj, x, candidate, mask = obs
        n_j = candidate.shape[-1]
        h_pooled, h_nodes = self.compute_feature(adj, x)
        # prepare policy feature: concat omega feature with global feature
        dummy = candidate.unsqueeze(-1).expand(-1, n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        # concatenate feature
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float("-inf")

        pi = F.softmax(candidate_scores, dim=1).squeeze(-1)
        v = self.critic(h_pooled)
        return pi, v

    def compute_feature(self, adj, x):
        n_nodes = x.shape[0]
        if adj.shape[0] == n_nodes:
            graph_pool_step = g_pool_cal("average", (1, n_nodes, n_nodes), n_nodes, self.device)
            graph_pool = graph_pool_step
        else:
            graph_pool_batch = g_pool_cal("average", (n_nodes, n_nodes, n_nodes), n_nodes, self.device)
            graph_pool = graph_pool_batch
        h_pooled, h_nodes = self.feature_extract(x=x, graph_pool=graph_pool, adj=adj)
        return h_pooled, h_nodes

    # def policy(self, obs):
    #     adj, x, candidate, mask = obs
    #     h_pooled, h_nodes = self.compute_feature(adj, x)
    #     # prepare policy feature: concat omega feature with global feature
    #     dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
    #     candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
    #     h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
    #
    #     # concatenate feature
    #     concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
    #     candidate_scores = self.actor(concateFea)
    #
    #     # perform mask
    #     mask_reshape = mask.reshape(candidate_scores.size())
    #     candidate_scores[mask_reshape] = float("-inf")
    #
    #     pi = F.softmax(candidate_scores, dim=1).squeeze(-1)
    #     return pi
    #
    # def value(self, obs):
    #     adj, x, candidate, mask = obs
    #     h_pooled, h_nodes = self.compute_feature(adj, x)
    #     v = self.critic(h_pooled)
    #     return v
