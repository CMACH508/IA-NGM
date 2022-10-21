import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.evaluation_metric import objective_score
from src.lap_solvers.hungarian import hungarian
import math
from src.utils.gpu_memory import gpu_free_memory

# from torch_geometric.data import Data, Batch
# from torch_geometric.utils import dense_to_sparse, to_dense_batch

from src.utils.config import cfg

from src.backbone import *

import numpy as np

CNN = eval(cfg.BACKBONE)


def pad_edges(feat_list):
    max_num = 0
    new_list = []
    d = feat_list[0].shape[1]
    for i in range(len(feat_list)):
        max_num = max(max_num, feat_list[i].shape[0])
    for i in range(len(feat_list)):
        feat = torch.zeros((max_num, d))
        for j in range(feat_list[i].shape[0]):
            feat[j, :] = feat_list[i][j, :]
        new_list.append(feat)
    return torch.stack(new_list, dim=0)


class Net(CNN):
    def __init__(self, dA=22, dB=22):
        super(Net, self).__init__()
        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))
        self.tau = cfg.NGM.SK_TAU
        self.rescale = cfg.PROBLEM.RESCALE
        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.gumbel_sinkhorn = GumbelSinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau * 10,
                                              epsilon=cfg.NGM.SK_EPSILON, batched_operation=True)
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5,
                                           k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        self.n_func = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1), nn.ReLU())
        self.self_func = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1), nn.ReLU())
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                # gnn_layer = Gconv(1, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                # gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                # gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                # gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)
        # self.dAlign = nn.Linear(dB, dA)

    def forward(self, data_dict, **kwargs):
        batch_size = data_dict['batch_size']
        ns_src = torch.tensor(data_dict['ns'], dtype=torch.long).cuda()
        ns_tgt = torch.tensor(data_dict['ns'], dtype=torch.long).cuda()
        U_src = data_dict['atom_feats']
        U_tgt = data_dict['residue_feats']

        A_src = data_dict['adj_mat0']
        A_tgt = data_dict['adj_mat1']

        F_src = []
        F_tgt = []
        for b in range(batch_size):
            F = []
            for row in range(A_src.shape[1]):
                for col in range(A_src.shape[2]):
                    if A_src[b, row, col] == 1:
                        F.append(torch.cat((U_src[b, row, :], U_src[b, col, :]), dim=0))
            F_src.append(torch.stack(F, dim=0))
        for b in range(batch_size):
            F = []
            for row in range(A_tgt.shape[1]):
                for col in range(A_tgt.shape[2]):
                    if A_tgt[b, row, col] == 1:
                        F.append(torch.cat((U_tgt[b, row, :], U_tgt[b, col, :]), dim=0))
            F_tgt.append(torch.stack(F, dim=0))
        F_src = pad_edges(F_src)
        F_tgt = pad_edges(F_tgt)

        if F_src.shape[1] > F_tgt.shape[1]:
            F_tgt_new = torch.zeros(F_src.shape)
            F_tgt_new[:, :F_tgt.shape[1], :] = F_tgt[:, :, :]
            F_tgt = F_tgt_new
        else:
            F_src_new = torch.zeros(F_tgt.shape)
            F_src_new[:, :F_src.shape[1], :] = F_src[:, :, :]
            F_src = F_src_new
        # print(U_src.shape, U_tgt.shape, F_src.shape, F_tgt.shape)
        G_src = data_dict['G0']
        G_tgt = data_dict['G1']
        H_src = data_dict['H0']
        H_tgt = data_dict['H1']
        K_G, K_H = data_dict['KGHs']

        tgt_len = U_tgt.shape[1]
        X = F_src.transpose(1, 2).to(device=U_src.device)
        Y = F_tgt.transpose(1, 2).to(device=U_src.device)
        U_src = U_src.transpose(1, 2)
        U_tgt = U_tgt.transpose(1, 2)

        # affinity layer
        Ke, Kp = self.affinity_layer(X, Y, U_src, U_tgt)

        K = construct_aff_mat(Ke, torch.zeros_like(Kp), K_G, K_H)

        A = (K > 0).to(K.dtype)

        if cfg.NGM.FIRST_ORDER:
            emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
        else:
            emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        emb_K = K.unsqueeze(-1)

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt)  # , norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], tgt_len, -1).transpose(1, 2)

        if self.training or cfg.NGM.GUMBEL_SK <= 0:
            # if cfg.NGM.GUMBEL_SK <= 0:
            ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
            x = hungarian(ss, ns_src, ns_tgt)
        else:
            gumbel_sample_num = cfg.NGM.GUMBEL_SK
            if self.training:
                gumbel_sample_num //= 10
            ss_gumbel = self.gumbel_sinkhorn(s, ns_src, ns_tgt, sample_num=gumbel_sample_num, dummy_row=True)

            repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
            if not self.training:
                ss_gumbel = hungarian(ss_gumbel, repeat(ns_src), repeat(ns_tgt))
            ss_gumbel = ss_gumbel.reshape(batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])

            if ss_gumbel.device.type == 'cuda':
                dev_idx = ss_gumbel.device.index
                free_mem = gpu_free_memory(dev_idx) - 100 * 1024 ** 2  # 100MB as buffer for other computations
                K_mem_size = K.element_size() * K.nelement()
                max_repeats = free_mem // K_mem_size
                if max_repeats <= 0:
                    print('Warning: GPU may not have enough memory')
                    max_repeats = 1
            else:
                max_repeats = gumbel_sample_num

            obj_score = []
            for idx in range(0, gumbel_sample_num, max_repeats):
                if idx + max_repeats > gumbel_sample_num:
                    rep_num = gumbel_sample_num - idx
                else:
                    rep_num = max_repeats
                obj_score.append(
                    objective_score(
                        ss_gumbel[:, idx:(idx + rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                        repeat(K, rep_num)
                    ).reshape(batch_size, -1)
                )
            obj_score = torch.cat(obj_score, dim=1)
            min_obj_score = obj_score.min(dim=1)
            ss = ss_gumbel[torch.arange(batch_size), min_obj_score.indices.cpu(), :, :]
            x = hungarian(ss, repeat(ns_src), repeat(ns_tgt))

        data_dict.update({
            'ds_mat': ss,
            'perm_mat': x,
            'aff_mat': K
        })
        return data_dict
