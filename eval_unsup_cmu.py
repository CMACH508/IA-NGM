import torch
import time
from datetime import datetime
from pathlib import Path

from src.binary import Binary
from src.dataset.data_loader_cmu import GMDataset, get_dataloader
from src.evaluation_metric import *
from src.parallel import DataParallel
from src.model_sl import load_model

from src.utils.config import cfg


def eval_model(model, dataloader, eval_epoch=None, verbose=False, xls_sheet = False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    lap_solver = Binary()

    recalls, precisions, f1s = [], [], []

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        recall_list, precision_list, f1_list = [], [], []

        for inputs in dataloader:

            A1, A2 = [_.cuda() for _ in inputs['As']]
            feat1, feat2 = [_.cuda() for _ in inputs['scfs']]
            P1, P2 = [_.cuda() for _ in inputs['Ps']]
            n1, n2 = [_.cuda() for _ in inputs['ns']]
            G1, G2 = [_.cuda() for _ in inputs['Gs']]
            H1, H2 = [_.cuda() for _ in inputs['Hs']]
            K_G, K_H = [_.cuda() for _ in inputs['KGHs']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            iter_num = iter_num + 1

            batch_num = feat1.size(0)


            with torch.set_grad_enabled(False):
                if cfg.MODULE == 'BIIA.model_unsup_cmu': # zhao's paper.
                    v, s, Bv = model(feat1, feat2, A1, A2, n1, n2)
                elif cfg.MODULE == 'GMN_UNSUP.model_cmu_noDeep': # 12's paper.
                    v, s, Bv = model(feat1, feat2, A1, A2,  P1, P2,n1, n2)
                else:
                    v, s, Bv = model(feat1, feat2,  G1, G2, H1, H2, n1, n2, K_G, K_H, inp_type='feat')
                
                Bv = lap_solver(s)

            recall, precision, _  = matching_accuracy(Bv, perm_mat, n1)
            recall_list.append(recall)
            precision_list.append(precision)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1[torch.isnan(f1)] = 0
            f1_list.append(f1)

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()
        
        recalls.append(torch.cat(recall_list))
        precisions.append(torch.cat(precision_list))
        f1s.append(torch.cat(f1_list))
    
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode = was_training)
    
    #
    # 表头
    # 
    if xls_sheet:
        for idx, cls in enumerate(classes):
            xls_sheet.write(0, idx + 1, cls)    #(行，列，value)
        xls_sheet.write(0, idx + 2, 'mean') 

        xls_row = 1
        xls_sheet.write(xls_row, 0, 'precision')
        xls_sheet.write(xls_row + 1, 0, 'recall')
        xls_sheet.write(xls_row + 2, 0, 'f1')

    for idx, (cls, cls_p, cls_r, cls_f1) in enumerate(zip(classes, precisions, recalls, f1s)):
        print('{}: {}'.format(cls, format_accuracy_metric(cls_p, cls_r, cls_f1))) 
        if xls_sheet:
            xls_sheet.write(xls_row, idx + 1, torch.mean(cls_p).item())
            xls_sheet.write(xls_row + 1, idx + 1, torch.mean(cls_r).item())
            xls_sheet.write(xls_row + 2, idx + 1, torch.mean(cls_f1).item())
    print('mean accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions), torch.cat(recalls), torch.cat(f1s))))
    if xls_sheet:
        xls_sheet.write(xls_row, idx + 2, torch.mean(torch.cat(precisions)).item())
        xls_sheet.write(xls_row + 1, idx + 2, torch.mean(torch.cat(recalls)).item())
        xls_sheet.write(xls_row + 2, idx + 2, torch.mean(torch.cat(f1s)).item())
    
    return torch.Tensor(list(map(torch.mean, recalls)))


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.IANet

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              obj_resize=cfg.PROBLEM.RESCALE)
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
