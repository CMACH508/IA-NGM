import torch
import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader_cmu import GMDataset, get_dataloader
from src.loss_func import RobustLoss,CrossEntropyLoss
from src.evaluation_metric import matching_accuracy
from src.regularization import Regularization
from src.parallel import DataParallel
from src.model_sl import load_model, save_model
from eval_unsup_cmu import eval_model
from src.lap_solvers.hungarian import hungarian
from src.count_model_params import count_parameters

from src.utils.config import cfg


def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     num_epochs=25,
                     resume=False,
                     start_epoch=0,
                     weight_decay = 0.001):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    lap_solver = hungarian

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:
  
            A1, A2 = [_.cuda() for _ in inputs['As']]
            feat1, feat2 = [_.cuda() for _ in inputs['scfs']]
            P1, P2 = [_.cuda() for _ in inputs['Ps']]
            n1, n2 = [_.cuda() for _ in inputs['ns']]
            G1, G2 = [_.cuda() for _ in inputs['Gs']]
            H1, H2 = [_.cuda() for _ in inputs['Hs']]
            K_G, K_H = [_.cuda() for _ in inputs['KGHs']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                if cfg.MODULE == 'BIIA.model_unsup_cmu': # zhao's paper.
                    v, s, Bv = model(feat1, feat2, A1, A2, n1, n2)
                elif cfg.MODULE == 'GMN_UNSUP.model_cmu_noDeep': # 12's paper.
                    v, s, Bv = model(feat1, feat2, A1, A2,  P1, P2,n1, n2)
                else:
                    v, s, Bv = model(feat1, feat2,  G1, G2, H1, H2, n1, n2, K_G, K_H, inp_type='feat')
                
            
                if cfg.TRAIN.LOSS_FUNC == 'perm':
                    loss = criterion(v ,Bv, n1, n2)
                    # if loss.item() > 100:
                    #     print(loss)
                    loss1 = criterion(Bv, perm_mat, n1, n2)
                    # print(loss1)
                else:
                    raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))
                
                if weight_decay > 0:
                    reg_loss = Regularization(model, weight_decay, p=2).to(device)
                    loss = loss + reg_loss(model)
                else:
                    print("no regularization")

                # backward + optimize
                loss.backward()
                # print(model.afifnity_Layer.omega1.grad)

                optimizer.step()

                # training accuracy statistic
                acc, _, _  = matching_accuracy(Bv, perm_mat, n1)

                # tfboard writer
                loss_dict = dict()
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                
                loss1_dict = dict()
                loss1_dict['loss1'] = loss1.item()
                tfboard_writer.add_scalars('loss-Bv-Gt', loss1_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)

                accdict = dict()
                accdict['matching accuracy'] =  torch.mean(acc)
                tfboard_writer.add_scalars(
                    'training accuracy', accdict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num )

                # statistics
                running_loss += loss.item() * perm_mat.size(0)
                epoch_loss += loss.item() * perm_mat.size(0)

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * perm_mat.size(0) / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.size(0)))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        accs = eval_model(model, dataloader['test'])
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        tfboard_writer.add_scalars(
            'Eval acc',
            acc_dict,
            (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        )

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE) 

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {x: GMDataset(cfg.DATASET_FULL_NAME, sets=x, length=dataset_len[x], clss=cfg.TRAIN.CLASS if x == 'train' else None) for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test')) for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("test random seed: ", torch.randn(5))
    # n_nodes = image_dataset['train'][0]['gt_perm_mat'].shape[0]

    model = mod.IANet().to(device)

    if cfg.TRAIN.LOSS_FUNC == 'perm':
        criterion = CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    # model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}k'.format(count_parameters(model) / 1e3))
        model = train_eval_model(model, criterion, optimizer, dataloader, tfboardwriter, num_epochs=cfg.TRAIN.NUM_EPOCHS, resume=cfg.TRAIN.START_EPOCH != 0, start_epoch=cfg.TRAIN.START_EPOCH)
