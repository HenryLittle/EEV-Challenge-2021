import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn.functional as F

import time
import shutil
import numpy as np
import math
from datetime import datetime
from einops import rearrange
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sam import SAM
import torch.nn.functional as F

from args import parser
from model_zzd import Baseline, TCFPN
from eev_dataset import EEV_Dataset
from utils import AverageMeter, correlation, loss_function, loss_functionC, loss_function_zzd, interpolate_output

best_corr = 0.0

def main_train():
    global args, best_corr
    
    args.store_name = '{}'.format(args.model)
    args.store_name = 'zzd'+args.store_name + datetime.now().strftime('_%m-%d_%H-%M-%S')
    args.start_epoch = 0

    check_rootfolders(args)
    if args.model == 'Baseline':
        model = Baseline()
        model2 = Baseline()
    elif args.model == 'TCFPN':
        model = TCFPN(layers=[48, 64, 96], in_channels=(2048 + 128), num_classes=15, kernel_size=11)
    
    model = torch.nn.DataParallel(model).cuda()
    model2 = torch.nn.DataParallel(model2).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.learning_rate = 0.02
    print('init: %f'%args.learning_rate)
    optimizer = torch.optim.SGD([
{'params': model.parameters(), 'lr': args.learning_rate},
{'params': model2.parameters(), 'lr': args.learning_rate},
                    ], weight_decay=1e-4, momentum=0.9, nesterov=True )
    # custom optimizer
    if args.use_sam:
        base_optim = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optim, lr=args.learning_rate)
    # custom lr scheduler
    #print(args.use_cos_wr)
    #if args.use_cos_wr:
    #args.cos_wr_t0 = 10
    #print('using Restart: %d'%args.cos_wr_t0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.cos_wr_t0,T_mult=2)
    #elif args.use_cos:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_t_max)
    # SWA
    if args.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=args.learning_rate)

    # ckpt structure {epoch, state_dict, optimizer, best_corr}
    if args.resume and os.path.isfile(args.resume):
        print('Load checkpoint:', args.resume)
        ckpt = torch.load(args.resume)
        args.start_epoch = ckpt['epoch']
        best_corr = ckpt['best_corr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Loaded ckpt at epoch:', args.start_epoch)


    # initialize datasets
    train_loader = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=args.train_csv,
            vidmap_path=args.train_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='train', lpfilter=args.lp_filter,
            train_freq=args.train_freq,
            val_freq=args.val_freq
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True
    )
    train_loader2 = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=args.train_csv,
            vidmap_path=args.train_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='train', lpfilter=args.lp_filter,
            train_freq=args.train_freq,
            val_freq=args.val_freq
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True
    )
    train_loader3 = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=args.train_csv,
            vidmap_path=args.train_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='train', lpfilter=args.lp_filter,
            train_freq=args.train_freq,
            val_freq=args.val_freq
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=args.val_csv,
            vidmap_path=args.val_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='val',
            train_freq=args.train_freq,
            val_freq=args.val_freq
        ),
        batch_size=None, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    accuracy = correlation
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    tb_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, train_loader2, train_loader3, model, model2, optimizer, epoch, log_training, tb_writer)
        # do lr scheduling after epoch
        if args.use_swa and epoch >= args.swa_start:
            print('swa stepping...')
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif args.use_cos_wr:
            print('cos warm restart (T0:{} Tm:{}) stepping...'.format(args.cos_wr_t0, args.cos_wr_t_mult))
            scheduler.step()
        elif args.use_cos:
            print('cos (Tmax:{}) stepping...'.format(args.cos_t_max))
            scheduler.step()
        

        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            # validate
            if args.use_swa and epoch >= args.swa_start:
                # validate use swa model
                corr = validate(val_loader, swa_model, accuracy, epoch, log_training, tb_writer)
            else:
                corr = validate(val_loader, model, accuracy, epoch, log_training, tb_writer)
            is_best = corr > best_corr
            best_corr = max(corr, best_corr)
            tb_writer.add_scalar('acc/validate_corr_best', best_corr, epoch)
            output_best = 'Best corr: %.4f\n' % (best_corr)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_corr': best_corr,
            }, is_best)



def train(train_loader, train_loader2, train_loader3, model, model2, optimizer, epoch, log, tb_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train() # switch to train mode
    model2.train() # switch to train mode

    t_start = time.time()
    iter_count = -1
    for data,data2,data3 in zip(train_loader, train_loader2, train_loader3):
        # measure data load time
        img_feat, au_feat, labels = data
        img_feat2, au_feat2, labels2 = data2
        img_feat3, au_feat3, labels3 = data3
        data_time.update(time.time() - t_start)

        # print(img_feat.dtype, au_feat.dtype, labels.dtype)
        img_feat, au_feat, labels = img_feat.cuda(), au_feat.cuda(), labels.cuda()
        img_feat2, au_feat2, labels2 = img_feat2.cuda(), au_feat2.cuda(), labels2.cuda()
        img_feat3, au_feat3, labels3 = img_feat3.cuda(), au_feat3.cuda(), labels3.cuda()

        output, i1, a1 = model(img_feat, au_feat)
        output2, i2, a2 = model2(img_feat2, au_feat2)
       
        alpha = 0
        loss = loss_functionC(output, labels) + alpha*loss_functionC(i1, labels)+ alpha*loss_functionC(a1, labels)
        loss += loss_functionC(output2, labels2)+alpha*loss_functionC(i2, labels) + alpha*loss_functionC(a2, labels)
        if epoch >= 50:
            output_reg,_,_ = model(img_feat3, au_feat3)
            output_reg2,_,_ = model2(img_feat3, au_feat3)
            mean_reg = (output_reg+output_reg2)/2
            reg = F.l1_loss( output_reg, mean_reg.detach() )  + F.l1_loss( output_reg2, mean_reg.detach() )
            loss += 0.5*reg
        losses.update(loss.item(), img_feat.size()[0])

        # apply different optimizer
        optimizer.zero_grad()
        if args.use_sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)
            #need re-write
            loss_function_zzd(model(img_feat, au_feat), model2(img_feat, au_feat), labels).backward()
            # reduce_gradients_from_all_accelerators() # 
            optimizer.second_step(zero_grad=True)
        else:
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - t_start)
        t_start = time.time() # reset timer
        iter_count = iter_count+1
        if iter_count % args.print_freq == 0 or epoch <= 1:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, iter_count, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            # print(losses.val, losses.avg, losses.sum, losses.count)
            log.write(output + '\n')
            log.flush()
    
    tb_writer.add_scalar('loss/train', losses.avg, epoch)
    tb_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, accuracy, epoch, log, tb_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    correlations = AverageMeter()

    model.eval()
    t_start = time.time()

    # load 1 video at a time for now (loader is not batched when batch_size == None)
    # but since we split one vid into multiple small clips, the input shape is still batch-like
    with torch.no_grad():
        for i, (img_feat, au_feat, labels, frame_count) in enumerate(val_loader):
            data_time.update(time.time() - t_start)

            # print(type(img_feat), len(img_feat), img_feat[0].size())
            img_feat = torch.stack(img_feat).cuda()
            au_feat = torch.stack(au_feat).cuda()
            labels = torch.stack(labels).cuda()

            output, _, _ = model(img_feat, au_feat) # [Clip S 15]
            # rearrange and remove extra padding in the end
            output = rearrange(output, 'Clip S C -> (Clip S) C')
            output = torch.cat([output, output[-1:]]) # repeat the last frame to avoid missing 
            if args.train_freq < args.val_freq:
                output = interpolate_output(output, args.train_freq, args.val_freq)
            output = output[:frame_count]
            labels = rearrange(labels, 'Clip S C -> (Clip S) C')[:frame_count]
            
            loss = loss_function(output, labels, validate=True)

            mean_cor, cor = accuracy(output, labels) # mean and per-class correlation
            # update statistics
            losses.update(loss[0].item())
            assert not math.isnan(mean_cor.item()), 'at epoch %d' % (epoch)
            correlations.update(mean_cor.item())

            batch_time.update(time.time() - t_start)
            t_start = time.time()

            if i % args.print_freq == 0:
                output = ('Val: [{0}/{1}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Corr: {corr.val:.4f} ({corr.avg:.4f}, {corr.count})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    corr=correlations))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Validate Results: Corr:{corr.avg:.4f} Loss {loss.avg:.5f}'
              .format(corr=correlations, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tb_writer is not None:
        tb_writer.add_scalar('loss/validate', losses.avg, epoch)
        tb_writer.add_scalar('acc/validate_corr', correlations.avg, epoch)

    return correlations.avg


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_ckpt,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_ckpt, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_ckpt, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def main_merge():
    global args, best_corr
    
    args.store_name = '{}_merged'.format(args.model)
    args.store_name = args.store_name + datetime.now().strftime('_%m-%d_%H-%M')
    args.start_epoch = 0

    check_rootfolders(args)

    model = Baseline()

    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # ckpt structure {epoch, state_dict, optimizer, best_corr}
    if args.resume and os.path.isfile(args.resume):
        print('Load checkpoint:', args.resume)
        ckpt = torch.load(args.resume)
        args.start_epoch = ckpt['epoch']
        best_corr = ckpt['best_corr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Loaded ckpt at epoch:', args.start_epoch)

    # initialize datasets
    train_loader = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=[args.train_csv, args.val_csv],
            vidmap_path=[args.train_vidmap, args.val_vidmap],
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='merge'
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        drop_last=True
    )

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    tb_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch, log_training, tb_writer)


def main_test():
    print('Running test...')
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = Baseline(num_layers=2)
    model2 = Baseline(num_layers=3)
    if args.use_swa:
        model = torch.optim.swa_utils.AveragedModel(model)
    model = torch.nn.DataParallel(model).cuda()
    model2 = torch.nn.DataParallel(model2).cuda()
    # ckpt structure {epoch, state_dict, optimizer, best_corr}
    if args.resume and os.path.isfile(args.resume):
        print('Load checkpoint:', args.resume)
        ckpt = torch.load(args.resume)
        args.start_epoch = ckpt['epoch']
        best_corr = ckpt['best_corr']
        model.load_state_dict(ckpt['state_dict'])
        print('Loaded ckpt at epoch:', args.start_epoch)
    else:
        print('No model given. Abort!')
        exit(1)

    test_loader = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=None,
            vidmap_path=args.test_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='test',
            test_freq=args.test_freq
        ),
        batch_size=None, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    model.eval()
    model2.eval()
    batch_time = AverageMeter()

    t_start = time.time()

    outputs = []
    with torch.no_grad():
        for i, (img_feat, au_feat, frame_count, vid) in enumerate(test_loader):
            img_feat = torch.stack(img_feat).cuda()
            au_feat = torch.stack(au_feat).cuda()
            assert len(au_feat.size()) == 3, 'bad auf %s' % (vid)
            output = model(img_feat, au_feat) + model2(img_feat, au_feat)# [Clip S 15]
            # rearrange and remove extra padding in the end
            output = rearrange(output, 'Clip S C -> (Clip S) C')
            output = torch.cat([output, output[-1:]]) # repeat the last frame to avoid missing 
            if args.train_freq < args.test_freq:
                # print('interpolating:', output.size()[0], frame_count)
                output = interpolate_output(output, args.train_freq, 6)
            # print('Interpolated:', output.size()[0], frame_count)
            # truncate extra frames
            assert output.size(0) >= frame_count, '{}/{}'.format(output.size(0), frame_count)
            output = output[:frame_count]
            outputs.append((vid, frame_count, output.cpu().detach().numpy()))

            # update statistics
            batch_time.update(time.time() - t_start)
            t_start = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time))
                print(output)
    
    time_stamps = [0, 166666, 333333, 500000, 666666, 833333]
    time_step = 1000000 # time starts at 0
    header = 'Video ID,Timestamp (milliseconds),amusement,anger,awe,concentration,confusion,contempt,contentment,disappointment,doubt,elation,interest,pain,sadness,surprise,triumph\n'
   
    final_res = {}
    for vid, frame_count, out in outputs:# videos
        video_time = frame_count // 6 + 1
        # print('video', vid, video_time)
        entry_count = 0
        for t in range(video_time): # seconds
            for i in range(6): # frames
                timestamp = time_step * t + time_stamps[i]
                fcc = t * 6 + i
                if fcc >= frame_count:
                    continue
                # print('Frame count', frame_count)
                frame_output = out[fcc]
                frame_output = [str(x) for x in frame_output]
                temp = '{vid},{timestamp},'.format(vid=vid,timestamp=timestamp) + ','.join(frame_output) + '\n'
                # file.write(temp)
                if vid in final_res:
                    final_res[vid].append(temp)
                else:
                    final_res[vid] = [temp]
                entry_count += 1
        assert entry_count == frame_count
    # fixed for now
    missing = [('WKXrnB7alT8', 2919), ('o0ooW14pIa4', 3733), ('GufMoL_MuNE',2038), ('Uee0Tv1rTz8', 1316), ('ScvvOWtb04Q', 152), ('R9kJlLungmo', 3609),('QMW3GuohzzE', 822), ('fjJYTW2n6rk', 4108), ('rbTIMt0VcLw', 1084),('L9cdaj74kLo', 3678), ('l-ka23gU4NA', 1759)]
    for vid, length in missing:
        video_time = length // 6 + 1
        # print('video', vid, video_time)
        for t in range(video_time): # seconds
            for i in range(6): # frames
                timestamp = time_step * t + time_stamps[i]
                fcc = t * 6 + i
                if fcc >= length:
                    continue
                frame_output = ',0'*15
                temp = '{vid},{timestamp}'.format(vid=vid, timestamp=timestamp) + frame_output + '\n'
                # file.write(temp)
                if vid in final_res:
                    final_res[vid].append(temp)
                else:
                    final_res[vid] = [temp]
    print('Write test outputs...')
    with open('test_output.csv', 'w') as file:
        file.write(header)
        temp_vidmap = [x.strip().split(' ') for x in open(args.test_vidmap)]
        temp_vidmap = [x[0] for x in temp_vidmap]
        for vid in tqdm(temp_vidmap):
            for entry in final_res[vid]:
                file.write(entry)




                
if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if args.run_test:
        main_test() # test model on test
    elif args.run_merge:
        main_merge() # train model using merged train/val
    else:
        main_train() # train model using train only
