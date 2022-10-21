import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn
from src.feature_align import feature_align
from src.gconv import Siamese_Gconv
from models.PCA.affinity_layer import Affinity
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg
from models.PCA.model_config import model_cfg

from src.backbone import *

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
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter=cfg.PCA.SK_ITER_NUM, epsilon=cfg.PCA.SK_EPSILON, tau=cfg.PCA.SK_TAU)
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5,
                                           k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        # self.pointer_net = PointerNet(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT // 2, alpha=cfg.PCA.VOTING_ALPHA)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))
        self.cross_iter = cfg.PCA.CROSS_ITER
        self.cross_iter_num = cfg.PCA.CROSS_ITER_NUM
        self.rescale = cfg.PROBLEM.RESCALE

    def reload_backbone(self):
        self.node_layers, self.edge_layers = self.get_backbone(True)

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
                        F.append(U_src[b, row, :] - U_src[b, col, :])
            F_src.append(torch.stack(F, dim=0))
        for b in range(batch_size):
            F = []
            for row in range(A_tgt.shape[1]):
                for col in range(A_tgt.shape[2]):
                    if A_tgt[b, row, col] == 1:
                        F.append(U_tgt[b, row, :] - U_tgt[b, col, :])
            F_tgt.append(torch.stack(F, dim=0))
        F_src = pad_edges(F_src).to(device=U_src.device)
        F_tgt = pad_edges(F_tgt).to(device=U_src.device)

        if F_src.shape[1] > F_tgt.shape[1]:
            F_tgt_new = torch.zeros(F_src.shape)
            F_tgt_new[:, :F_tgt.shape[1], :] = F_tgt[:, :, :]
            F_tgt = F_tgt_new.to(device=U_src.device)
        else:
            F_src_new = torch.zeros(F_tgt.shape)
            F_src_new[:, :F_src.shape[1], :] = F_src[:, :, :]
            F_src = F_src_new.to(device=U_src.device)

        emb1, emb2 = U_src, U_tgt
        ss = []

        if not self.cross_iter:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

                ss.append(s)

                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                    new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2
        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1], device=emb1.device)

            for x in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                ss.append(s)

        data_dict.update({
            'ds_mat': ss[-1],
            'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)
        })
        return data_dict
