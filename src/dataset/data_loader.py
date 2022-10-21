import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch_geometric as pyg
import numpy as np
import random
from src.build_graphs import build_graphs
from src.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch import CSRMatrix3d
from src.dataset import *
import os
import pickle
import Bio
from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem
import scipy.optimize as opt
from sklearn import preprocessing
from lxml import etree
import deepchem as dc

from src.utils.config import cfg

from itertools import combinations, product
from src.gconv import Gconv

PATH_REFINED = './refined-set/'
PATH_OTHER = './v2020-other-PL/'
PATH_ALL = './refined-other/'

ELEM_REFINED = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
DEGREE_REFINED = [1, 2, 3, 4, 5]
CHARGE_REFINED = [0, 1, -1]
RESIDUE_TYPE = ['HIS', 'ARG', 'LYS',
                'PHE', 'ALA', 'LEU', 'MET', 'ILE', 'TRP', 'PRO', 'VAL',
                'CYS', 'GLY', 'GLN', 'ASN', 'SER', 'TYR', 'THR',
                'ASP', 'GLU',
                ]
DIST_THRESHOLD = 6
X = 379.627
Y = 409.308
Z = 537.002
# BATCH_SIZE = cfg.BATCH_SIZE

random.seed(123)


def normalize(inp):
    scaler = preprocessing.MinMaxScaler()
    return torch.Tensor(scaler.fit_transform(inp))


def onek_encoding_unk(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))


def node_features(node, inp_type):
    assert inp_type in ['ligand', 'residue']
    if inp_type == 'ligand':
        return torch.Tensor(
            onek_encoding_unk(node.GetSymbol(), ELEM_REFINED)
            + onek_encoding_unk(node.GetDegree(), DEGREE_REFINED)
            + onek_encoding_unk(node.GetFormalCharge(), CHARGE_REFINED)
            + [node.GetIsAromatic()])
    else:
        return torch.Tensor(onek_encoding_unk(node.get_resname(), RESIDUE_TYPE))  # + [len(list(node))])
        # emb = nn.Embedding(20, 22)
        # return emb(torch.LongTensor([RESIDUE_TYPE.index(node.get_resname())]))


def edge_features(coord0, coord1):
    return torch.Tensor(coord0) - torch.Tensor(coord1)


def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two):
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float64)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


def calc_residues_dist(residues):
    mat = np.zeros((len(residues), len(residues)), np.float64)
    for row, residue_1 in enumerate(residues):
        for col, residue_2 in enumerate(residues):
            mat[row, col] = calc_residue_dist(residue_1, residue_2)
    return mat


def getGH(adj):
    edge_idx = 0
    edge_num = int(torch.sum(adj))
    node_num = adj.shape[0]
    G = torch.zeros((node_num, edge_num))
    H = torch.zeros((node_num, edge_num))
    for i in range(node_num):
        for j in range(node_num):
            if adj[i, j] == 1:
                G[i, edge_idx] = 1
                H[j, edge_idx] = 1
                edge_idx += 1
    return G, H


class GMDataset(Dataset):
    def __init__(self, name, length, cls=None, problem='2GM', **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        self.length = length  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
        # length here represents the iterations between two checkpoints
        self.obj_size = self.ds.obj_resize
        self.cls = None if cls in ['none', 'all'] else cls

        if self.cls is None:
            if problem == 'MGMC':
                self.classes = list(combinations(self.ds.classes, cfg.PROBLEM.NUM_CLUSTERS))
            else:
                self.classes = self.ds.classes
        else:
            self.classes = [self.cls]

        self.problem_type = problem

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.problem_type == '2GM':
            return self.get_pair(idx, self.cls)
        elif self.problem_type == 'MGM':
            return self.get_multi(idx, self.cls)
        elif self.problem_type == 'MGMC':
            return self.get_multi_cluster(idx)
        else:
            raise NameError("Unknown problem type: {}".format(self.problem_type))

    @staticmethod
    def to_pyg_graph(A, P):
        rescale = max(cfg.PROBLEM.RESCALE)

        edge_feat = 0.5 * (np.expand_dims(P, axis=1) - np.expand_dims(P, axis=0)) / rescale + 0.5  # from Rolink's paper
        edge_index = np.nonzero(A)
        edge_attr = edge_feat[edge_index]

        edge_attr = np.clip(edge_attr, 0, 1)
        assert (edge_attr > -1e-5).all(), P

        o3_A = np.expand_dims(A, axis=0) * np.expand_dims(A, axis=1) * np.expand_dims(A, axis=2)
        hyperedge_index = np.nonzero(o3_A)

        pyg_graph = pyg.data.Data(
            x=torch.tensor(P / rescale).to(torch.float32),
            edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr).to(torch.float32),
            hyperedge_index=torch.tensor(np.array(hyperedge_index), dtype=torch.long),
        )
        return pyg_graph

    def get_pair(self, idx, cls):
        # anno_pair, perm_mat = self.ds.get_pair(self.cls if self.cls is not None else
        #                                       (idx % (cfg.BATCH_SIZE * len(self.classes))) // cfg.BATCH_SIZE)
        try:
            anno_pair, perm_mat = self.ds.get_pair(cls, tgt_outlier=cfg.PROBLEM.TGT_OUTLIER,
                                                   src_outlier=cfg.PROBLEM.SRC_OUTLIER)
        except TypeError:
            anno_pair, perm_mat = self.ds.get_pair(cls)
        if min(perm_mat.shape[0], perm_mat.shape[1]) <= 2 or perm_mat.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
            return self.get_pair(idx, cls)

        cls = [anno['cls'] for anno in anno_pair]
        P1 = [(kp['x'], kp['y']) for kp in anno_pair[0]['keypoints']]
        P2 = [(kp['x'], kp['y']) for kp in anno_pair[1]['keypoints']]

        n1, n2 = len(P1), len(P2)
        univ_size = [anno['univ_size'] for anno in anno_pair]

        P1 = np.array(P1)
        P2 = np.array(P2)

        A1, G1, H1, e1 = build_graphs(P1, n1, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)
        if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same':
            G2 = perm_mat.transpose().dot(G1)
            H2 = perm_mat.transpose().dot(H1)
            A2 = G2.dot(H2.transpose())
            e2 = e1
        else:
            A2, G2, H2, e2 = build_graphs(P2, n2, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)

        pyg_graph1 = self.to_pyg_graph(A1, P1)
        pyg_graph2 = self.to_pyg_graph(A2, P2)

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1, P2]],
                    'ns': [torch.tensor(x) for x in [n1, n2]],
                    'es': [torch.tensor(x) for x in [e1, e2]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G1, G2]],
                    'Hs': [torch.Tensor(x) for x in [H1, H2]],
                    'As': [torch.Tensor(x) for x in [A1, A2]],
                    'pyg_graphs': [pyg_graph1, pyg_graph2],
                    'cls': [str(x) for x in cls],
                    'univ_size': [torch.tensor(int(x)) for x in univ_size],
                    }

        imgs = [anno['image'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_pair[0]['keypoints'][0]:
            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['keypoints']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['keypoints']], axis=-1)
            ret_dict['features'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict

    def get_multi(self, idx, cls):
        if (self.ds.sets == 'test' and cfg.PROBLEM.TEST_ALL_GRAPHS) or (
                self.ds.sets == 'train' and cfg.PROBLEM.TRAIN_ALL_GRAPHS):
            num_graphs = self.ds.len(cls)
        else:
            num_graphs = cfg.PROBLEM.NUM_GRAPHS
        anno_list, perm_mat_list = self.ds.get_multi(cls, num=num_graphs)

        assert isinstance(perm_mat_list, list)
        refetch = False
        for pm in perm_mat_list:
            if pm.shape[0] <= 2 or pm.shape[1] <= 2 or pm.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
                refetch = True
                break
        if refetch:
            return self.get_multi(idx, cls)

        cls = [anno['cls'] for anno in anno_list]
        Ps = [[(kp['x'], kp['y']) for kp in anno_dict['keypoints']] for anno_dict in anno_list]

        ns = [len(P) for P in Ps]
        univ_size = [anno['univ_size'] for anno in anno_list]

        Ps = [np.array(P) for P in Ps]

        As = []
        Gs = []
        Hs = []
        As_tgt = []
        Gs_tgt = []
        Hs_tgt = []
        for P, n, perm_mat in zip(Ps, ns, perm_mat_list):
            # In multi-graph matching (MGM), when a graph is regarded as target graph, its topology may be different
            # from when it is regarded as source graph. These are represented by suffix "tgt".
            if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same' and len(Gs) > 0:
                G = perm_mat.dot(Gs[0])
                H = perm_mat.dot(Hs[0])
                A = G.dot(H.transpose())
                G_tgt = G
                H_tgt = H
                A_tgt = G_tgt.dot(H_tgt.transpose())
            else:
                A, G, H, _ = build_graphs(P, n, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT)
                A_tgt, G_tgt, H_tgt, _ = build_graphs(P, n, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT)
            As.append(A)
            Gs.append(G)
            Hs.append(H)
            As_tgt.append(A_tgt)
            Gs_tgt.append(G_tgt)
            Hs_tgt.append(H_tgt)

        pyg_graphs = [self.to_pyg_graph(A, P) for A, P in zip(As, Ps)]
        pyg_graphs_tgt = [self.to_pyg_graph(A, P) for A, P in zip(As_tgt, Ps)]

        ret_dict = {
            'Ps': [torch.Tensor(x) for x in Ps],
            'ns': [torch.tensor(x) for x in ns],
            'gt_perm_mat': perm_mat_list,
            'Gs': [torch.Tensor(x) for x in Gs],
            'Hs': [torch.Tensor(x) for x in Hs],
            'As': [torch.Tensor(x) for x in As],
            'Gs_tgt': [torch.Tensor(x) for x in Gs_tgt],
            'Hs_tgt': [torch.Tensor(x) for x in Hs_tgt],
            'As_tgt': [torch.Tensor(x) for x in As_tgt],
            'pyg_graphs': pyg_graphs,
            'pyg_graphs_tgt': pyg_graphs_tgt,
            'cls': [str(x) for x in cls],
            'univ_size': [torch.tensor(int(x)) for x in univ_size],
        }

        imgs = [anno['image'] for anno in anno_list]
        if imgs[0] is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_list[0]['keypoints'][0]:
            feats = [np.stack([kp['feat'] for kp in anno_dict['keypoints']], axis=-1) for anno_dict in anno_list]
            ret_dict['features'] = [torch.Tensor(x) for x in feats]

        return ret_dict

    def get_multi_cluster(self, idx):
        dicts = []
        if self.cls is None or self.cls == 'none':
            cls_iterator = random.choice(self.classes)
        else:
            cls_iterator = self.cls
        for cls in cls_iterator:
            dicts.append(self.get_multi(idx, cls))
        ret_dict = {}
        for key in dicts[0]:
            ret_dict[key] = []
            for dic in dicts:
                ret_dict[key] += dic[key]
        return ret_dict


class DrugDataset(Dataset):
    def __init__(self, train=True, refined=True, **args):
        self.refined = refined
        self.train = train
        if refined:
            self.sdf_dir = PATH_ALL
        else:
            self.sdf_dir = PATH_OTHER
        self.sdf_path = os.listdir(self.sdf_dir)

        self.gc = Gconv(3, 3)
        self.dAlign = nn.Sequential(nn.Linear(300, 3), nn.Sigmoid(), nn.Linear(3, 3), nn.Sigmoid())

        # if refined:
        EXC_FILE = open('EXC_ALL_40.pkl', 'rb')
        exc_set = pickle.load(EXC_FILE)
        EXC_FILE.close()
        for name in exc_set:
            self.sdf_path.remove(name)
        # else:
        #     EXC_FILE = open('EXC_OTHER.pkl', 'rb')
        #     exc_set = pickle.load(EXC_FILE)
        #     EXC_FILE.close()
        #     for name in exc_set:
        #         self.sdf_path.remove(name)
        random.shuffle(self.sdf_path)
        split_idx = int(len(self.sdf_path) * 0.8)
        if train:
            self.sdf_path = self.sdf_path[:split_idx]
        else:
            self.sdf_path = self.sdf_path[split_idx:]

    def __len__(self):
        return len(self.sdf_path)

    def __getitem__(self, idx):
        try:
            pdb = self.sdf_dir + self.sdf_path[idx] + '/' + self.sdf_path[idx] + '_pocket.pdb'
            # pdb = self.sdf_dir + self.sdf_path[idx] + '/' + self.sdf_path[idx] + '_protein.pdb'
            # mol=Chem.rdmolfiles.MolFromPDBFile(pdb)
            # mol=Chem.MolToSmiles(mol)
            # featurizer = dc.feat.CircularFingerprint(size=300)
            # residue_feats = featurizer.featurize(mol)
            parser = PDBParser(PERMISSIVE=False)
            model = list(parser.get_structure(self.sdf_path[idx], pdb).get_models())[0]
            residues_raw = list(model.get_residues())
            residues = [r for r in residues_raw if r.get_resname() in RESIDUE_TYPE]
            # residues = []
            residue_feats = []
            bridge_feats = []
            # for ix in range(len(residues_raw)):
            #     if residues_raw[ix].get_resname() in RESIDUE_TYPE:
            #         residues.append(residues_raw[ix])
            # residue_feats.append(node_features(residues_raw[ix], 'residue'))

            sdf = self.sdf_dir + self.sdf_path[idx] + '/' + self.sdf_path[idx] + '_ligand.sdf'
            ligand = Chem.SDMolSupplier(sdf, removeHs=False)[0]
            atoms = ligand.GetAtoms()
            bonds = ligand.GetBonds()
            atom_feats = []
            bond_feats = []
            adj_mat0 = torch.zeros((len(atoms), len(atoms)))
            for atom in atoms:
                atom_feats.append(dc.feat.graph_features.atom_features(atom))

            neighbors = []
            for idx, atom in enumerate(atoms):
                neighbor = []
                x0, y0, z0 = ligand.GetConformer().GetAtomPosition(idx)
                for bond in bonds:
                    if bond.GetBeginAtomIdx() == idx:
                        x1, y1, z1 = ligand.GetConformer().GetAtomPosition(bond.GetEndAtomIdx())
                        neighbor.append((bond.GetEndAtomIdx(),
                                         np.sqrt(
                                             (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))))
                    elif bond.GetEndAtomIdx() == idx:
                        x1, y1, z1 = ligand.GetConformer().GetAtomPosition(bond.GetBeginAtomIdx())
                        neighbor.append((bond.GetBeginAtomIdx(),
                                         np.sqrt(
                                             (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))))
                neighbor.sort(key=lambda x: x[1])
                neighbors.append(neighbor[:3])
            for idx, atom in enumerate(atoms):
                for neighbor in neighbors[idx]:
                    np.concatenate((atom_feats[idx], atom_feats[neighbor[0]]))
                while len(atom_feats[idx]) < 300:
                    atom_feats[idx] = np.concatenate((atom_feats[idx], np.zeros((75))))
                atom_feats[idx] = torch.tensor(atom_feats[idx])

            for bond in bonds:
                atomIdx0 = bond.GetBeginAtomIdx()
                atomIdx1 = bond.GetEndAtomIdx()
                bond_feats.append(atom_feats[atomIdx0] - atom_feats[atomIdx1])

            # for i, atom in enumerate(atoms):
            #     x, y, z = ligand.GetConformer().GetAtomPosition(i)
            #     x /= X
            #     y /= Y
            #     z /= Z
            #     atom_feats.append(
            #         torch.cat((node_features(atom, 'ligand'), torch.Tensor([x, y, z]))))

            dist_mat = torch.zeros((len(atoms), len(residues)))
            for row, atom in enumerate(atoms):
                for col, residue in enumerate(residues):
                    x0, y0, z0 = ligand.GetConformer().GetAtomPosition(row)
                    x1, y1, z1 = residue['CA'].coord
                    dist_mat[row, col] = np.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))

            row, col = opt.linear_sum_assignment(dist_mat, maximize=False)
            gt_perm_mat = np.zeros(dist_mat.shape)
            gt_perm_mat[row, col] = 1
            gt_perm_mat = gt_perm_mat[~(gt_perm_mat == 0).all(1)]
            gt_perm_mat = gt_perm_mat[:, ~(gt_perm_mat == 0).all(0)]

            for bond in bonds:
                atomIdx0 = bond.GetBeginAtomIdx()
                atomIdx1 = bond.GetEndAtomIdx()
                #     bond_feats.append(edge_features(list(ligand.GetConformer().GetAtomPosition(atomIdx0)),
                #                                     list(ligand.GetConformer().GetAtomPosition(atomIdx1))))
                adj_mat0[atomIdx0][atomIdx1] = 1
            num = min(len(atoms), len(residues))
            if num < len(atoms):
                adj_mat0 = adj_mat0[sorted(row), :][:, sorted(row)]
                gt_idx = sorted(row)
                atoms_new = []
                bonds_new = []
                atom_feats_new = []
                bond_feats_new = []
                for i, atom in enumerate(atoms):
                    if i in gt_idx:
                        atoms_new.append(atom)
                        atom_feats_new.append(atom_feats[i])
                for i, bond in enumerate(bonds):
                    if bond.GetBeginAtomIdx() in gt_idx and bond.GetEndAtomIdx() in gt_idx:
                        bonds_new.append(bond)
                        bond_feats_new.append(bond_feats[i])
                atoms = atoms_new
                atom_feats = atom_feats_new
                bonds = bonds_new
                bond_feats = bond_feats_new

            elif num < len(residues):
                gt_idx = sorted(col)
                residues_new = []
                for i, residue in enumerate(residues):
                    if i in gt_idx:
                        residues_new.append(residue)
                residues = residues_new

            for residue in residues:
                raf = []
                for i, atom in enumerate(residue.child_list):
                    if atom.name == 'CA':
                        idx = i
                    raf.append([atom.bfactor, atom.mass, atom.occupancy])
                raf = torch.unsqueeze(torch.tensor(raf), 0)
                A = torch.ones((1, raf.shape[1], raf.shape[1]))
                raf = self.gc(A, raf)
                residue_feats.append(raf[0, idx])

            # for residue in residues:
            #     residue_feats.append(node_features(residue, 'residue'))
            residue_feats = torch.stack(residue_feats, dim=0)
            # print(residue_feats.shape)
            # input()

            atom_feats = torch.stack(atom_feats, dim=0)
            bond_feats = normalize(torch.stack(bond_feats, dim=0))
            G0, H0 = getGH(adj_mat0)

            dist_res = calc_residues_dist(residues)
            adj_mat1 = dist_res.copy()
            adj_mat1[adj_mat1 < DIST_THRESHOLD] = 1
            adj_mat1[adj_mat1 >= DIST_THRESHOLD] = 0
            adj_mat1 -= np.eye(adj_mat1.shape[0])
            adj_mat1 = torch.Tensor(adj_mat1)
            G1, H1 = getGH(adj_mat1)

            for row in range(adj_mat1.shape[0]):
                for col in range(adj_mat1.shape[1]):
                    if adj_mat1[row, col] == 1:
                        bridge_feats.append(residue_feats[row] - residue_feats[col])
            bridge_feats = torch.stack(bridge_feats, dim=0)
            # print(len(atoms), len(residues))

            row, col = opt.linear_sum_assignment(dist_mat, maximize=False)
            gt_perm_mat = np.zeros(dist_mat.shape)
            gt_perm_mat[row, col] = 1
            row_ind = ~(gt_perm_mat == 0).all(1)
            col_ind = ~(gt_perm_mat == 0).all(0)
            gt_perm_mat = gt_perm_mat[row_ind, :]
            gt_perm_mat = gt_perm_mat[:, col_ind]
            gt_perm_mat = torch.eye(gt_perm_mat.shape[0])

            # print(len(atom_feats), len(residue_feats))

            # n = len(atoms)
            # print(atom_feats.shape,residue_feats.shape)
            # input()
            #
            # print(self.dAlign(atom_feats))
            # input()
            # atom_feats = self.dAlign(atom_feats)
            # print(atom_feats.shape,residue_feats.shape)
            # input()

            # K = torch.zeros((n * n, n * n, atom_feats.shape[1]))
            # for i in range(n * n):
            #     for j in range(n * n):
            #         print((atom_feats[i // n] - atom_feats[i % n]) - (
            #                 residue_feats[j // n] - residue_feats[j % n]))
            #         input()
            #         K[i][j][:] = (atom_feats[i // n] - atom_feats[i % n]) - (
            #                 residue_feats[j // n] - residue_feats[j % n])

            if len(atom_feats) <= 2 or len(residue_feats) <= 2:
                # return self.__getitem__(random.randint(0,len(self)-1))
                while 1:
                    idx = random.randint(0, len(self.sdf_path) - 1)
                    if self.sdf_path[idx] not in self.exc_set:
                        return self.__getitem__(idx)
            # pyg_ligand = self.to_pyg_graph(adj_mat0, atom_feats, bond_feats)
            # pyg_residue = self.to_pyg_graph(adj_mat1, residue_feats, bridge_feats)

            data_dict = {
                # ligand
                'atom_feats': atom_feats,
                'bond_feats': bond_feats,
                'adj_mat0': adj_mat0,
                'G0': G0,
                'H0': H0,
                # residue
                'residue_feats': residue_feats,
                'bridge_feats': bridge_feats,
                'adj_mat1': adj_mat1,
                'G1': G1,
                'H1': H1,
                # 'K':K,

                # 'ns': [],
                # 'es': [],
                'gt_perm_mat': torch.Tensor(gt_perm_mat),
                'Gs': [torch.Tensor(G0), torch.Tensor(G1)],
                'Hs': [torch.Tensor(H0), torch.Tensor(H1)],
                'As': [torch.Tensor(adj_mat0), torch.Tensor(adj_mat1)],
                # 'pyg_graphs': [pyg_ligand, pyg_residue]
            }
            # print(data_dict)
            return data_dict
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))

    @staticmethod
    def to_pyg_graph(adj_mat, node_feats, edge_feats):
        edge_index = np.nonzero(adj_mat)

        edge_attr = np.clip(edge_feats, 0, 1)
        assert (edge_attr > -1e-5).all(), node_feats

        o3_A = np.expand_dims(adj_mat, axis=0) * np.expand_dims(adj_mat, axis=1) * np.expand_dims(adj_mat, axis=2)
        hyperedge_index = np.nonzero(o3_A)

        pyg_graph = pyg.data.Data(
            x=node_feats,
            edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
            edge_attr=torch.tensor(edge_feats).to(torch.float32),
            hyperedge_index=torch.tensor(np.array(hyperedge_index), dtype=torch.long),
        )
        return pyg_graph


class KBDataset(Dataset):
    def __init__(self, train=True):
        self.sdf_dir = PATH_ALL
        EXC_FILE = open('EXC_ALL_40.pkl', 'rb')
        exc_set = pickle.load(EXC_FILE)
        EXC_FILE.close()
        PAIR_FILE = open('PAIR_ALL.pkl', 'rb')
        self.pair = pickle.load(PAIR_FILE)
        PAIR_FILE.close()
        self.sdf_path = os.listdir(self.sdf_dir)
        for name in exc_set:
            # print(name)
            if name in self.sdf_path:
                self.sdf_path.remove(name)
        random.shuffle(self.sdf_path)
        split_idx = int(len(self.sdf_path) * 0.8)
        if train:
            self.sdf_path = self.sdf_path[:split_idx]
        else:
            self.sdf_path = self.sdf_path[split_idx:]
        self.exc_set = exc_set

    def __len__(self):
        return len(self.sdf_path)

    def __getitem__(self, idx):
        sdf = self.sdf_dir + self.sdf_path[idx] + '/' + self.sdf_path[idx] + '_ligand.sdf'
        ligand = Chem.SDMolSupplier(sdf, removeHs=False)[0]
        atoms = ligand.GetAtoms()
        bonds = ligand.GetBonds()
        adj_mat0 = torch.zeros((len(atoms), len(atoms)))
        for bond in bonds:
            atomIdx0 = bond.GetBeginAtomIdx()
            atomIdx1 = bond.GetEndAtomIdx()
            adj_mat0[atomIdx0][atomIdx1] = 1

        pdb = self.sdf_dir + self.sdf_path[idx] + '/' + self.sdf_path[idx] + '_pocket.pdb'
        # pdb = self.sdf_dir + self.sdf_path[idx] + '/' + self.sdf_path[idx] + '_protein.pdb'
        parser = PDBParser(PERMISSIVE=False)
        model = list(parser.get_structure(self.sdf_path[idx], pdb).get_models())[0]
        residues_raw = list(model.get_residues())
        residues = [r for r in residues_raw if r.get_resname() in RESIDUE_TYPE]

        # cmd = "plip -i {} -xv".format(self.sdf_path[idx])
        # os.system(cmd)
        # for file in os.listdir('./'):
        #     if self.sdf_path[idx] in file:
        #         cmd = "rm {} -rf".format(file)
        #         os.system(cmd)
        tree = etree.parse(self.sdf_dir + self.sdf_path[idx] + '/report.xml')
        xs = [float(x.text) for x in tree.xpath("//x")][::2]
        ys = [float(x.text) for x in tree.xpath("//y")][::2]
        zs = [float(x.text) for x in tree.xpath("//z")][::2]
        coords = list(zip(xs, ys, zs))
        rid = [x.text[:-1] for x in tree.xpath("//bs_residue")]
        select_idx_residue = [True if str(r.get_full_id()[-1][1]) in rid else False for r in residues]
        select_idx_atom = [True if tuple(ligand.GetConformer().GetAtomPosition(i)) in coords else False for i in
                           range(len(atoms))]

        idx0 = [i for i in range(len(atoms)) if select_idx_atom[i]]
        adj_mat0 = adj_mat0[idx0, :][:, idx0]
        atoms = [atoms[i] for i in range(len(atoms)) if select_idx_atom[i]]
        residues = [residues[i] for i in range(len(residues)) if select_idx_residue[i]]

        dist_mat = torch.zeros((len(atoms), len(residues)))
        # print(dist_mat.shape)
        for row, atom in enumerate(atoms):
            if not select_idx_atom[row]:
                continue
            for col, residue in enumerate(residues):
                if not select_idx_residue[col]:
                    continue
                x0, y0, z0 = ligand.GetConformer().GetAtomPosition(row)
                x1, y1, z1 = residue['CA'].coord
                dist_mat[row, col] = np.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))

        row, col = opt.linear_sum_assignment(dist_mat, maximize=False)
        gt_perm_mat = np.zeros(dist_mat.shape)
        gt_perm_mat[row, col] = 1
        row_ind = ~(gt_perm_mat == 0).all(1)
        col_ind = ~(gt_perm_mat == 0).all(0)
        gt_perm_mat = gt_perm_mat[row_ind, :]
        gt_perm_mat = gt_perm_mat[:, col_ind]
        gt_perm_mat = torch.eye(gt_perm_mat.shape[0])

        atoms = [atoms[i] for i in range(len(atoms)) if row_ind[i] == 1]
        residues = [residues[i] for i in range(len(residues)) if col_ind[i] == 1]
        row_ind = [e[0] for e in list(enumerate(row_ind)) if e[1] == 1]

        if len(atoms) <= 2 or len(residues) <= 2:
            # return self.__getitem__(random.randint(0,len(self)-1))
            while 1:
                idx = random.randint(0, len(self.sdf_path) - 1)
                if self.sdf_path[idx] not in self.exc_set:
                    return self.__getitem__(idx)
        # print(len(atoms),len(residues),dist_mat.shape)
        # input()

        # F = adj_mat0[row_ind, :][:, row_ind]
        F = torch.ones((len(atoms)), len(atoms))
        D = torch.tensor(calc_residues_dist(residues)).to(torch.float32)
        B = torch.zeros((len(atoms), len(atoms)))
        for i, atom in enumerate(atoms):
            for j, residue in enumerate(residues):
                if (atom.GetSymbol(), residue.get_resname()) in self.pair:
                    B[i, j] = 1
        for i in range(len(residues)):
            for j in range(len(residues)):
                if i == j:
                    D[i, j] = 10
                else:
                    D[i, j] = 1 / D[i, j]

        ret_dict = {
            'Fi': F,
            'Fj': D,
            'B': B,
            'gt_perm_mat': torch.tensor(gt_perm_mat),
            'ns': torch.tensor([torch.tensor(x) for x in gt_perm_mat.shape])
        }

        return ret_dict


class QAPDataset(Dataset):
    def __init__(self, name, length, pad=16, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args, cls=cls)
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls
        self.length = length

    def __len__(self):
        # return len(self.ds.data_list)
        return self.length

    def __getitem__(self, idx):
        Fi, Fj, perm_mat, sol, name = self.ds.get_pair(idx % len(self.ds.data_list))
        if perm_mat.size <= 2 * 2 or perm_mat.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # if np.max(ori_aff_mat) > 0:
        #    norm_aff_mat = ori_aff_mat / np.mean(ori_aff_mat)
        # else:
        #    norm_aff_mat = ori_aff_mat
        ret_dict = {'Fi': Fi,
                    'Fj': Fj,
                    'gt_perm_mat': perm_mat,
                    'ns': [torch.tensor(x) for x in perm_mat.shape],
                    'solution': torch.tensor(sol),
                    'name': name,
                    'univ_size': [torch.tensor(x) for x in perm_mat.shape], }

        return ret_dict


def pad_tensor_drug(inp):
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break
    max_shape = np.array(max_shape)
    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))
    ret = torch.stack(padded_ts, 0)
    return ret


def pad_feature(data_dict):
    ret = {}
    for kvs in zip(*[x.items() for x in data_dict]):
        ks, vs = zip(*kvs)
        for k in ks:
            assert k == ks[0], "Keys mismatch."
        if type(vs[0]) == torch.Tensor:
            ret[k] = pad_tensor_drug(vs)
        # if type(vs[0]) == list:
        #     pass
        # if type(vs[0]) == pyg.data.Data:
        #     ret[k] = pyg.data.Batch.from_data_list(vs)
        # else:
        #     print(type(vs[0]))
    return ret


def collate_fn_drug(data_dict):
    ns = []
    BATCH_SIZE = len(data_dict)
    for b in range(BATCH_SIZE):
        max_n = max(data_dict[b]['atom_feats'].shape[0], data_dict[b]['residue_feats'].shape[0])
        ns.append(max_n)
        perm_mat = torch.zeros((max_n, max_n))
        perm_mat[:data_dict[b]['atom_feats'].shape[0], :data_dict[b]['residue_feats'].shape[0]] \
            = data_dict[b]['gt_perm_mat'][:, :]
        data_dict[b]['gt_perm_mat'] = perm_mat

    ret = pad_feature(data_dict)
    ret['ns'] = ns

    num_atom = ret['atom_feats'].shape[1]
    num_residue = ret['residue_feats'].shape[1]
    max_num_n = max(num_atom, num_residue)
    atom_feats = torch.zeros((BATCH_SIZE, max_num_n, ret['atom_feats'].shape[-1]))
    residue_feats = torch.zeros((BATCH_SIZE, max_num_n, ret['residue_feats'].shape[-1]))
    for b in range(BATCH_SIZE):
        for n in range(max_num_n):
            if n < ret['atom_feats'].shape[1]:
                atom_feats[b, n, :] = ret['atom_feats'][b, n, :]
            if n < ret['residue_feats'].shape[1]:
                residue_feats[b, n, :] = ret['residue_feats'][b, n, :]
    ret['atom_feats'] = atom_feats
    ret['residue_feats'] = residue_feats

    # print(ret['G0'].shape, ret['H0'].shape, ret['G1'].shape, ret['H1'].shape)
    num_edge0 = ret['G0'].shape[2]
    num_edge1 = ret['G1'].shape[2]
    max_num_e = max(num_edge0, num_edge1)
    G0_new = torch.zeros((BATCH_SIZE, max_num_n, max_num_e))
    G1_new = torch.zeros((BATCH_SIZE, max_num_n, max_num_e))
    H0_new = torch.zeros((BATCH_SIZE, max_num_n, max_num_e))
    H1_new = torch.zeros((BATCH_SIZE, max_num_n, max_num_e))
    for b in range(BATCH_SIZE):
        for n in range(max_num_e):
            if n < ret['G0'].shape[2]:
                G0_new[b, :ret['G0'].shape[1], n] = ret['G0'][b, :, n]
            if n < ret['H0'].shape[2]:
                H0_new[b, :ret['H0'].shape[1], n] = ret['H0'][b, :, n]
            if n < ret['G1'].shape[2]:
                G1_new[b, :ret['G1'].shape[1], n] = ret['G1'][b, :, n]
            if n < ret['H1'].shape[2]:
                H1_new[b, :ret['H1'].shape[1], n] = ret['H1'][b, :, n]
    ret['G0'] = G0_new
    ret['G1'] = G1_new
    ret['H0'] = H0_new
    ret['H1'] = H1_new
    ret['batch_size'] = BATCH_SIZE
    # K1G = [kronecker_sparse(x, y) for x, y in
    #        zip(G1_new, G0_new)]  # 1 as source graph, 2 as target graph
    # K1H = [kronecker_sparse(x, y) for x, y in zip(H1_new, H0_new)]
    # K1G = CSRMatrix3d(K1G)
    # K1H = CSRMatrix3d(K1H).transpose()
    #
    # ret['KGHs'] = K1G, K1H
    return ret


def collate_fn_KB(data_dict):
    BATCH_SIZE = len(data_dict)
    ret = pad_feature(data_dict)
    ret['ns'] = ret['ns'].transpose(0, 1)
    if 'Fi' in ret and 'Fj' in ret and 'B' in ret:
        Fi = ret['Fi']
        Fj = ret['Fj']
        n = Fi.shape[1]
        B = ret['B']
        aff_mat = kronecker_torch(Fj, Fi)+kronecker_torch(B.transpose(1,2),B)
        # print(B.shape,aff_mat.shape)/
        # for b in range(BATCH_SIZE):
        #     for i in range(n):
        #         for j in range(n):
        #             for k in range(n):
        #                 for l in range(n):
        #                     aff_mat[b, i * k, j * l] += (B[b, i, k] + B[b, j, l])
        ret['aff_mat'] = aff_mat

    ret['batch_size'] = BATCH_SIZE

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break
    # print(ret)
    return ret


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            # pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == pyg.data.Data:
            ret = pyg.data.Batch.from_data_list(inp)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive Kronecker product here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        if cfg.PROBLEM.TYPE == '2GM' and len(ret['Gs']) == 2 and len(ret['Hs']) == 2:
            G1, G2 = ret['Gs']
            H1, H2 = ret['Hs']
            if cfg.FP16:
                sparse_dtype = np.float16
            else:
                sparse_dtype = np.float32
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in
                   zip(G2, G1)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()

            ret['KGHs'] = K1G, K1H
        elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC'] and 'Gs_tgt' in ret and 'Hs_tgt' in ret:
            ret['KGHs'] = dict()
            for idx_1, idx_2 in product(range(len(ret['Gs'])), repeat=2):
                # 1 as source graph, 2 as target graph
                G1 = ret['Gs'][idx_1]
                H1 = ret['Hs'][idx_1]
                G2 = ret['Gs_tgt'][idx_2]
                H2 = ret['Hs_tgt'][idx_2]
                if cfg.FP16:
                    sparse_dtype = np.float16
                else:
                    sparse_dtype = np.float32
                KG = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]
                KH = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
                KG = CSRMatrix3d(KG)
                KH = CSRMatrix3d(KH).transpose()
                ret['KGHs']['{},{}'.format(idx_1, idx_2)] = KG, KH
        else:
            raise ValueError('Data type not understood.')

    if 'Fi' in ret and 'Fj' in ret:
        Fi = ret['Fi']
        Fj = ret['Fj']
        aff_mat = kronecker_torch(Fj, Fi)
        ret['aff_mat'] = aff_mat

    ret['batch_size'] = len(data)
    ret['univ_size'] = torch.tensor([max(*[item[b] for item in ret['univ_size']]) for b in range(ret['batch_size'])])

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break

    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        # _drug,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


def get_dataloader_drug(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn_drug,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


def get_dataloader_KB(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn_KB,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )
