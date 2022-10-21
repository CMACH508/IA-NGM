from math import e
from pathlib  import Path
import random
import numpy as np

from src.build_graphs import build_graphs, delaunay_triangulate
from src.utils.config import cfg

class CmuObject:
    def __init__(self, sets):
        """
        Parameter:  sets, 'train' or 'test'

        """
        super(CmuObject, self).__init__()
        self.sets = sets
        self.classes = cfg.CmuObject.CLASSES
        self.root_path = Path(cfg.CmuObject.ROOT_DIR)

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        self.train_len = cfg.CmuObject.TRAIN_NUM

        # 存放对应类的.scf文件名
        self.fea_list = [] 

        for cls_name in self.classes:
            assert type(cls_name) is str, 'the type of class name is not str'
            cls_fea_list = [p for p in (self.root_path /cls_name).glob('*.scf')]
            ori_len = len(cls_fea_list)

            assert ori_len > 0, 'No data found for CMU Object Class. Is the dataset installed correctly?'
            assert self.train_len <= ori_len, 'Train length is larger than dataset length.'

            if sets == 'train':
                self.fea_list.append(cls_fea_list[:self.train_len])
            else:
                self.fea_list.append(cls_fea_list[self.train_len:])

    def get_pair(self, cls=None):
        """
        Randomly get a pair of objects from CMU-House-Hotel dataset

        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        # list, 存放两个anno_dict, source和 target
        anno_pair = []
        # source and target 随机从self.fea_list[cls] 中选择
        for fea_name in random.sample(self.fea_list[cls], 2):
            anno_dict = self.__get_anno_dict(fea_name)
            anno_pair.append(anno_dict)
        
        # 生成ground truth, permutation matrix
        perm_mat = np.zeros((len(anno_pair[0]['fea']),len(anno_pair[1]['fea'])), dtype=np.float32)
        # perm_mat = np.zeros_like(anno_dict['adj'], dtype=np.float32)
        for i, keypoint in enumerate(anno_pair[0]['fea']):
            for j, _keypoint in enumerate(anno_pair[1]['fea']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    break
        
        return anno_pair, perm_mat

    
    # 从fea_name文件中得到每张图的信息，存放在字典 anno_dict中, 
    # anno_dict 包括: list 形式 'fea' = fea_list, [0] ~ [29], 代表图中30个keypoints
    # fea_list 包括: dict 形式 attr, 'name' -> 0~29
    #                          'scf' -> shape context 1X60 array
    #                          'x' -> cooridinate x
    #                          'y' -> cooridinate y
    def __get_anno_dict(self, fea_name, shuffle=True):
        assert fea_name.exists(), '{} does not exist.'.format(fea_name)

        fea_list = []
        # file cooridinate/hotelxxx
        coor_name = fea_name.stem
        coor_file = fea_name.parent / 'cooridinate' / coor_name
        
        cooridinate_array = np.loadtxt(coor_file, dtype= np.int16)

        # file .scf
        lines_fea = open(fea_name).readlines()
        for idx, line in enumerate(lines_fea):
            attr = {'name': idx}
            attr['scf'] = np.array([float(p) for p in line.split()])
            attr['coor_x'] = cooridinate_array[idx, 0]
            attr['coor_y'] = cooridinate_array[idx, 1]
            fea_list.append(attr)
        
        if shuffle:
            random.shuffle(fea_list)
        
        P = []
        for i in range(len(fea_list)):
            coor_tmp = []
            coor_tmp.append(fea_list[i]['coor_x'])
            coor_tmp.append(fea_list[i]['coor_y'])
            P.append(coor_tmp)
            
        A = delaunay_triangulate(np.array(P))

        anno_dict = dict()
        anno_dict['fea'] = fea_list
        anno_dict['adj'] = A

        return anno_dict



        

