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
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
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
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']

            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
        elif 'IA' in data_dict:
            U_src, U_tgt, F_src, F_tgt = data_dict['UFs']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            ns_src, ns_tgt = data_dict['ns']
            A_src, A_tgt = data_dict['As']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('Unknown data type for this model.')

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
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
        if 'IA' not in data_dict:
            data_dict['IA'] = True
            del data_dict['images']
            data_dict['UFs'] = (U_src, U_tgt, F_src, F_tgt)
            data_dict['Ps'] = (P_src, P_tgt)
            data_dict['ns'] = (ns_src, ns_tgt)
            data_dict['As'] = (A_src, A_tgt)
        return data_dict


class IANet(CNN):
    def __init__(self):
        super(IANet, self).__init__()
        self.net = Net()

    def forward(self, data_dict, alpha=0.9):
        data_dict1 = self.net(data_dict)
        s1 = data_dict['perm_mat']
        U_src0, U_tgt0, F_src0, F_tgt0 = data_dict1['UFs']
        U_src1 = torch.bmm(s1, U_tgt0.transpose(2, 1)).transpose(2, 1)
        U_tgt1 = torch.bmm(s1.transpose(2, 1), U_src0.transpose(2, 1)).transpose(2, 1)
        F_src1 = torch.bmm(s1, F_tgt0.transpose(2, 1)).transpose(2, 1)
        F_tgt1 = torch.bmm(s1.transpose(2, 1), F_src0.transpose(2, 1)).transpose(2, 1)
        U_srcT1 = alpha * U_src0 + (1 - alpha) * U_src1
        U_tgtT1 = alpha * U_tgt0 + (1 - alpha) * U_tgt1
        F_srcT1 = alpha * F_src0 + (1 - alpha) * F_src1
        F_tgtT1 = alpha * F_tgt0 + (1 - alpha) * F_tgt1
        data_dict1['UFs'] = (U_srcT1, U_tgtT1, F_srcT1, F_tgtT1)

        data_dict2 = self.net(data_dict1)
        return data_dict2
