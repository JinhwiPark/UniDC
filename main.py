import wandb
import csv
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm


from utils.util_func import *
from utils.metric_func import *

import json
from config import args as args_config
from model_list import import_model
import pdb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'


args = args_config
best_rmse = 10.0
fieldnames = ['epoch', 'loss', 'rmse', 'mae', 'delta_125_1']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) , sum(p.numel() for p in model.parameters() if not p.requires_grad)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.workers=4*len(convert_str_to_num(args.gpus,'int'))
    
    current_time = time.strftime('%y%m%d_%H%M%S')
    # args.save_dir = '/workspace/logs/train/model_comparison/{}_{}_{}'.format(current_time,args.model_name,args.loss)
    args.save_dir = '/workspace/logs/train/NIPS2024/{}_{}_{}'.format(current_time,args.model_name,args.loss)
    if args.multiple_times:
        args.save_dir_multiple_times = '/workspace/logs/PAPER_REPORT/{}_{}'.format(args.model_name,args.data_name)
        if args.withoutDenseGT_KITTI:
            args.save_dir_multiple_times = '/workspace/logs/PAPER_REPORT/[WithoutDenseGT_train-{}Line_test-{}Line{}'.format(args.lidar_lines, args.kitti_val_lidars, args.model_name)
        os.makedirs(args.save_dir_multiple_times, exist_ok=True)
        with open(args.save_dir_multiple_times + '/result_{}{}shot_{}.csv'.format(args.few_shot_way,args.minidataset_fewshot_number,args.num_multiple_times), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        try:
            with open(args.save_dir_multiple_times + '/TOTAL_result_{}{}shot.csv'.format(args.few_shot_way,args.minidataset_fewshot_number), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        except: pass
        
    os.makedirs(args.save_dir, exist_ok=True)
    

    with open(args.save_dir + '/train.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(args.save_dir + '/val.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    args.num_gpu = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)


def main_worker(args):

    global best_rmse
    
    best_rmse = 10.0
    args.num_images_seen = 0
    
    # Data loading code
    if args.data_name == 'NYU':
        if args.model_name=='depth_prompt_main_midas_ARKIT':
            args.patch_height, args.patch_width = 256,192    
        else:
            args.patch_height, args.patch_width = 240, 320
        args.max_depth = 10.0
        args.split_json = './data/nyu.json'
        args.dir_data = '/workspace/data/DepthCompletion/nyudepthv2/'
        from data.nyu import NYU as NYU_Dataset
        train_dataset = NYU_Dataset(args, 'train')
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [NYU_Dataset(args, 'test', num_sample_test=v) for v in target_vals]

    elif args.data_name == 'KITTIDC':
        args.patch_height, args.patch_width = 256, 1216
        args.max_depth = 80.0
        args.split_json = './data/kitti_dc.json'
        args.dir_data = '/workspace/data/DepthCompletion/kitti_DC/'
        target_vals = convert_str_to_num(args.kitti_val_lidars, 'int')
        args.top_crop = 96
        from data.kittidc import KITTIDC as KITTI_dataset
        train_dataset = KITTI_dataset(args, 'train')
        val_datasets = [KITTI_dataset(args, 'test', num_lidars_test=v) for v in target_vals] # TODO:
    else:
        print("Please Choice Dataset !!")
        raise NotImplementedError

    model = import_model(args)
    init_lr = args.lr 

    if args.loss == 'reltoabs2':
        from loss.l1l2loss_reltoabs2 import L1L2Loss
        criterion = L1L2Loss(args).cuda(args.gpus)
    else:
        raise NotImplementedError("Loss Check")


    trainable = filter(lambda x: x.requires_grad, model.parameters())
    learnable_params , non_learnable_params = count_parameters(model)

    print()
    print("*"*30)
    print("Model :", args.model_name)
    print("Dataset :", args.data_name)
    # print("Loss : ",args.loss)
    print("# of TOTAL LEARNABLE PARAMETER :", learnable_params )
    print("# of TOTAL Non-LEARNABLE PARAMETER :", non_learnable_params )
    # print('Save Directory {}'.format(args.save_dir))
    print("*"*30)
    print()

    optimizer = torch.optim.Adam(trainable, init_lr, betas=args.betas,
                                eps=args.epsilon, weight_decay=args.weight_decay)

    calculator = LRFactor(args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, calculator.get_factor, verbose=False)
    
    model = torch.nn.DataParallel(model)
    model.cuda() 

    cudnn.benchmark = True

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)


    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=False, drop_last=False) for val_dataset in val_datasets]
    

    
    for epoch in range(args.start_epoch, args.epochs):


        train_loss, train_rmse, train_mae = train(train_loader, model, criterion, optimizer, epoch, args)

        print('Validation Epoch {}/{} | {}{}shot | multipletime: {}'.format(str(epoch),str(args.epochs),args.few_shot_way,args.minidataset_fewshot_number,args.num_multiple_times))
        if str(epoch + 1) in args.validation_epochs.split(',') or epoch + 1 == args.epochs:
                
            avg_rmse = AverageMeter('avg_rmse', ':6.3f')
            avg_mae = AverageMeter('avg_mae', ':6.3f')
            avg_delta1 = AverageMeter('avg_mae', ':6.3f')

            for target_val, val_loader in zip(target_vals, val_loaders):
                val_loss, val_rmse, val_mae, val_irmse, val_imae, val_rel, val_delta_125, val_delta_125_2 , val_delta_125_3 = validate(val_loader, model, criterion,epoch, args)
                print("{:2.4f} {:2.4f} {:2.4f} ".format(val_rmse,val_mae,val_delta_125),end="")
                print()
                avg_rmse.update(val_rmse)
                avg_mae.update(val_mae)
                avg_delta1.update(val_delta_125)

            total_val_rmse = avg_rmse.avg
            best_rmse = min(total_val_rmse, best_rmse)
            scheduler.step()
 

def train(train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter('Loss', ':.4f')
    rmse = AverageMeter('RMSE', ':.4f')
    mae = AverageMeter('MAE', ':.4f')

    model.train()

    for i, sample in enumerate(train_loader):
        if i==5 and args.exp_name=='Debug':
            break



        sample = {key: val.cuda() for key, val in sample.items() if val is not None}
        args.num_images_seen += len(sample['rgb'])

        output = model(sample)

        loss = criterion(output, sample['gt'])


        losses.update(loss.item(), sample['gt'].size(0))

        rmse_result = rmse_eval(sample, output['pred'])
        rmse.update(rmse_result, sample['gt'].size(0))

        mae_result = mae_eval(sample, output['pred'])
        mae.update(mae_result, sample['gt'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.wandb:
            wandb.log({"num_images_seen": args.num_images_seen}, step=epoch)
            try:
                wandb.log({"curvature_0": output['curvatures'][0]}, step=epoch)
                wandb.log({"curvature_1": output['curvatures'][1]}, step=epoch)
                wandb.log({"curvature_2": output['curvatures'][2]}, step=epoch)
                wandb.log({"curvature_3": output['curvatures'][3]}, step=epoch)
            except: pass
    return losses.avg, rmse.avg, mae.avg

def validate(val_loader, model, criterion, epoch, args):
    
    losses = AverageMeter('Loss', ':.4e')

    rmse = AverageMeter('RMSE', ':.4f')
    mae = AverageMeter('MAE', ':.4f')
    irmse = AverageMeter('iRMSE', ':.4f')
    imae = AverageMeter('iMAE', ':.4f')
    rel = AverageMeter('REL', ':.4f')
    delta_125 = AverageMeter('DELTA_125', ':.4f')
    delta_125_2 = AverageMeter('DELTA_125_2', ':.4f')
    delta_125_3 = AverageMeter('DELTA_125_3', ':.4f')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            if i==100 and args.exp_name=='Debug':
                break


            sample = {key: val.cuda() for key, val in sample.items() if val is not None}

            output = model(sample)

            loss = criterion(output, sample['gt'])

            losses.update(loss.item(), sample['gt'].size(0))

            result_all = evaluate_all_metric(sample, output['pred'])

            rmse.update(result_all[0], sample['gt'].size(0))
            mae.update(result_all[1], sample['gt'].size(0))
            irmse.update(result_all[2], sample['gt'].size(0))
            imae.update(result_all[3], sample['gt'].size(0))
            rel.update(result_all[4], sample['gt'].size(0))
            delta_125.update(result_all[5], sample['gt'].size(0))
            delta_125_2.update(result_all[6], sample['gt'].size(0))
            delta_125_3.update(result_all[7], sample['gt'].size(0))

        os.system("chmod -R 777 {}".format(args.save_dir))
    return losses.avg, rmse.avg, mae.avg, irmse.avg, imae.avg, rel.avg, delta_125.avg, delta_125_2.avg, delta_125_3.avg

if __name__ == '__main__':
    main()
