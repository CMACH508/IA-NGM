import torch
import torch.nn as nn

from utils.sinkhorn import Sinkhorn
from utils.hungarian import hungarian
from utils.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.feature_align import feature_align
from PCA.gconv import Siamese_Gconv
from PCA.gcns import Siamese_Net
from PCA.merge_layers import MergeLayers
from PCA.affinity_layer import Affinity
from PCA.cosine_affinity import Cosine_Affinity
import torch.nn.functional as F
from PCA.probability_layer import SequencePro, FullPro

from utils.config import cfg

import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()

        self.gnn_layers = 1
        self.iteration_ = 4
        self.a1 = 0.75
        self.a2 = 1.25

        # resnet
        self.R = nn.Linear(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.FEATURE_CHANNEL * 2, bias=False)
        self.relu = nn.ReLU()

        # self.affinity = Affinity(cfg.PCA.FEATURE_CHANNEL * 2)
        # self.cosine_affinity = Cosine_Affinity(cfg.PCA.FEATURE_CHANNEL * 2)
        self.merge_layer = MergeLayers(cfg.PCA.FEATURE_CHANNEL * 2).cuda()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON)
        self.voting_layer_0 = FullPro(alpha=cfg.PCA.VOTING_ALPHA)
        self.voting_layer_1 = FullPro(alpha=(cfg.PCA.VOTING_ALPHA/2))
        self.displacement_layer = Displacement()
        self.hung = hungarian
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        for i in range(self.gnn_layers):
            ggnn = Siamese_Net(cfg.PCA.FEATURE_CHANNEL * 2)
            self.add_module('ggnn_{}'.format(i), ggnn)

        for k in range(self.iteration_):
            self.add_module('affinity_{}'.format(k), Affinity(cfg.PCA.FEATURE_CHANNEL * 2))

            # if k > 1:
            #     self.add_module('voting_{}'.format(k), Voting(alpha=cfg.PCA.VOTING_ALPHA/2))
            # else:
            #     self.add_module('voting_{}'.format(k), Voting(alpha=cfg.PCA.VOTING_ALPHA))

    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
        if type == 'img' or type == 'image':
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
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        # adjacency matrices
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        A_src = A_src.repeat(1, 1, 2)
        A_tgt = A_tgt.repeat(1, 1, 2)

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)

        emb1_init, emb2_init = emb1, emb2

        for i in range(self.gnn_layers):
            ggnn_layer = getattr(self, 'ggnn_{}'.format(i))
            emb1, emb2 = ggnn_layer([emb1, A_src], [emb2, A_tgt])
            # emb1 = emb1_new
            # emb2 = emb1_new
        emb1 = self.relu(emb1 + self.R(emb1_init))
        emb2 = self.relu(emb2 + self.R(emb2_init))

        for k in range(self.iteration_):
            affinity = getattr(self, 'affinity_{}'.format(k))
            # voting = getattr(self, 'voting_{}'.format(k))
            if k > 0:
                s = self.a1*affinity(emb1, emb2) + self.a2*affinity(torch.bmm(p, emb2), torch.bmm(p.transpose(1, 2), emb1))
                s = self.voting_layer_1(s, ns_src, ns_tgt)
            else:
                s = affinity(emb1, emb2)
                s = self.voting_layer_0(s, ns_src, ns_tgt)

            # s = self.affinity(emb1, emb2)
            # s = self.relu(s)
            # s = self.merge_layer(s, ns_src, ns_tgt)
            # s = self.voting_layer(s, ns_src, ns_tgt)
            s = self.bi_stochastic(s, ns_src, ns_tgt)
            p = self.hung(s, ns_src, ns_tgt)
            # self.a1 -= 0.25
            # self.a2 += 0.25

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d
