import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random

from src.dataset.cmu_house_hotel import CmuObject
from src.build_graphs import build_graphs
from src.factorize_graph_matching import kronecker_sparse
from src.sparse_torch import CSRMatrix3d
from src.utils.config import cfg

class GMDataset(Dataset):
    def __init__(self, name, length, clss=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        self.length = length
        self.cls = None if clss in ['none', 'all'] else clss

        if self.cls is not None:
            self.classes = [self.cls]
        else:
            self.classes = self.ds.classes
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anno_pair, perm_mat = self.ds.get_pair(self.cls)

        P1 = [(attr['coor_x'], attr['coor_y']) for attr in anno_pair[0]['fea']]
        P2 = [(attr['coor_x'], attr['coor_y']) for attr in anno_pair[1]['fea']]
        
        n1, n2 = len(P1), len(P2)

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
        

        ret_dict = {
                    'As': [torch.Tensor(x) for x in [A1, A2]],
                    # 'As': [torch.Tensor(x) for x in [anno_pair[0]['adj'], anno_pair[1]['adj']]],
                    'ns': [torch.tensor(x) for x in [n1, n2]],
                    'Gs': [torch.Tensor(x) for x in [G1, G2]],
                    'Hs': [torch.Tensor(x) for x in [H1, H2]],
                    'gt_perm_mat': perm_mat,
                    'Ps':  [torch.Tensor(x) for x in [P1, P2]]}
        
        feat1 = np.stack([p['scf'] for p in anno_pair[0]['fea']], axis=0)
        feat2 = np.stack([p['scf'] for p in anno_pair[1]['fea']], axis=0)

        ret_dict['scfs'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict
    


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
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive Kronecker product here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        if len(ret['Gs']) == 2 and len(ret['Hs']) == 2:
            G1, G2 = ret['Gs']
            H1, H2 = ret['Hs']
            sparse_dtype = np.float32

            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2, G1)]
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2, H1)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()
            ret['KGHs'] = K1G, K1H
    ret['batch_size']=len(data)
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
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


if __name__ == '__main__':
    dataset_len = {'train': 70, 'test': 10}
    image_dataset = {
        x: GMDataset('CmuObject', sets = x, length = dataset_len[x]) for x in ('train', 'test')
    }
    
    data_loader = {
        x: get_dataloader(image_dataset[x], fix_seed = (x == 'test')) for x in ('train','test')
    }       
    for inputs in data_loader['train']:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        perm_mat = inputs['gt_perm_mat'].cuda()

        print(perm_mat)