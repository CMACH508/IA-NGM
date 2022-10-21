IA-NGM: A Bidirectional Learning Method For Neural Graph Matching with Feature Fusion
==========
Most existing deep learning methods for graph matching tasks tend to focus on affinity learning in a feedforward fashion to assist the neural network solver. However, the potential benefits of a direct feedback from the neural network solver to the affinity learning are usually underestimated and overlooked. %Meanwhile, the existing bidirectional learning method has its limitations on the exploration of bidirectional learning and its performance is now out of date. 
In this paper, we propose a bidirectional learning method to tackle the above issues. Our method leverages the output of a neural network solver to perform feature fusion on the input of affinity learning. Such direct feedback helps augment the input feature maps of the raw images according to the current solution. A feature fusion procedure is proposed to enhance the raw features with pseudo features that contain deviation information of the current solution from the ground-truth one. As a result, the bidrectional alternation enables the learning component to benefit from the feedback, while keeping the strengths of learning affinity models.
According to the results of experiments conducted on five benchmark datasets, our methods outperform the corresponding state-of-the-art feedforward methods.  


Environment
-------------
Get the recommended docker image by
```bash
docker pull runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3-pygmtools0.2.4
```

See details in [ThinkMatch-runtime](https://github.com/Thinklab-SJTU/ThinkMatch-runtime).

Dataset
-------------
Available datasets include PascalVOC,WILLOW-ObjectClass,IMC_PT_SparseGM,CUB_200_2011 and QAPLIB.

Note: All following datasets can be automatically downloaded and unzipped by `pygmtools`, but you can also download the dataset yourself if a download failure occurs. 

1. PascalVOC-Keypoint

    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/TrainVal/VOCdevkit/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``. **This file must be added manually.**

    Please cite the following papers if you use PascalVOC-Keypoint dataset:
    ```
    @article{EveringhamIJCV10,
      title={The pascal visual object classes (voc) challenge},
      author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
      journal={International Journal of Computer Vision},
      volume={88},
      pages={303–338},
      year={2010}
    }
    
    @inproceedings{BourdevICCV09,
      title={Poselets: Body part detectors trained using 3d human pose annotations},
      author={Bourdev, L. and Malik, J.},
      booktitle={International Conference on Computer Vision},
      pages={1365--1372},
      year={2009},
      organization={IEEE}
    }
    ```
1. Willow-Object-Class
    1. Download [Willow-ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)
    1. Unzip the dataset and make sure it looks like ``data/WillowObject/WILLOW-ObjectClass``

    Please cite the following paper if you use Willow-Object-Class dataset:
    ```
    @inproceedings{ChoICCV13,
      author={Cho, Minsu and Alahari, Karteek and Ponce, Jean},
      title = {Learning Graphs to Match},
      booktitle = {International Conference on Computer Vision},
      pages={25--32},
      year={2013}
    }
    ```

1. CUB2011
    1. Download [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz).
    1. Unzip the dataset and make sure it looks like ``data/CUB_200_2011/CUB_200_2011``

    Please cite the following report if you use CUB2011 dataset:
    ```
    @techreport{CUB2011,
      Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
      Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
      Year = {2011},
      Institution = {California Institute of Technology},
      Number = {CNS-TR-2011-001}
    }
    ```

1. IMC-PT-SparseGM
    1. Download the IMC-PT-SparseGM dataset from [google drive](https://drive.google.com/file/d/1Po9pRMWXTqKK2ABPpVmkcsOq-6K_2v-B/view?usp=sharing) or [baidu drive (code: 0576)](https://pan.baidu.com/s/1hlJdIFp4rkiz1Y-gztyHIw)
    1. Unzip the dataset and make sure it looks like ``data/IMC_PT_SparseGM/annotations``

    Please cite the following papers if you use IMC-PT-SparseGM dataset:
    ```
    @article{JinIJCV21,
      title={Image Matching across Wide Baselines: From Paper to Practice},
      author={Jin, Yuhe and Mishkin, Dmytro and Mishchuk, Anastasiia and Matas, Jiri and Fua, Pascal and Yi, Kwang Moo and Trulls, Eduard},
      journal={International Journal of Computer Vision},
      pages={517--547},
      year={2021}
    }
    
    @unpublished{WangPAMIsub21,
      title={Robust Self-supervised Learning of Deep Graph Matching with Mixture of Modes},
      author={Wang, Runzhong and Jiang, Shaofei and Yan, Junchi and Yang, Xiaokang},
      note={submitted to IEEE Transactions of Pattern Analysis and Machine Intelligence},
      year={2021}
    }
    ```
1.	QAPLIB
	1. Download [QAPLIB](https://www.opt.math.tugraz.at/qaplib/data.d/qapdata.tar.gz).
	1. Unzip the dataset and make sure it looks like ``data/qapdata``

Run the Experiment
-------------

Run training and evaluation
```bash
python train_eval.py --cfg path/to/your/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
```bash
python train_eval.py --cfg experiments/vgg16_pca_voc.yaml
```

Default configuration files are stored in``experiments/`` and you are welcomed to try your own configurations. 


Trained model path
-----------
As saved files are too big, we only provide trained models of IA-NGM on PascalVOC via [baidu drive (code:9xx2)]https://pan.baidu.com/s/14Fn976zXyxi_EGfRreddVA.

Trained models using different models and datasets were saved under:  
2227/data2/qintianxiang/IA-NGM/output/
2227/data2/qintianxiang/IA-NGM/old_output_back_up/

The two folders above contains results,optim.pt and params.pt files, the first one is mainly for experiments during revision, and the latter is the back up of results in the past.