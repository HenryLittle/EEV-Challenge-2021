# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F

import time
import os
import shutil
import numpy as np
import math
from datetime import datetime
from einops import rearrange
from tensorboardX import SummaryWriter


from args import parser
from model import Baseline
from eev_dataset import EEV_Dataset
from utils import AverageMeter, correlation

best_corr = 0.0

def main():
    global args, best_corr
    args = parser.parse_args()

    args.store_name = 'Baseline_GRU'
    args.store_name = args.store_name + datetime.now().strftime('_%m-%d-%Y_%H-%M')
    args.start_epoch = 0

    check_rootfolders(args)

    model = Baseline()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
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
            csv_path=args.train_csv,
            vidmap_path=args.train_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='train'
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=args.val_csv,
            vidmap_path=args.val_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features,
            mode='val'
        ),
        batch_size=None, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    # criterion = torch.nn.KLDivLoss().cuda()
    criterion = torch.nn.L1Loss().cuda()
    accuracy = correlation
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    tb_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, log_training, tb_writer)

        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            # validate
            corr = validate(val_loader, model, criterion, accuracy, epoch, log_training, tb_writer)
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



def train(train_loader, model, criterion, optimizer, epoch, log, tb_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train() # switch to train mode

    t_start = time.time()
    for i, (img_feat, au_feat, labels) in enumerate(train_loader):
        # measure data load time
        data_time.update(time.time() - t_start)

        # print(img_feat.dtype, au_feat.dtype, labels.dtype)
        img_feat = img_feat.cuda()
        au_feat = au_feat.cuda()
        labels = labels.cuda()

        output = model(img_feat, au_feat)
        # log_softmax is numerically more stable than log(softmax(output)) [TESTED]
        # loss = criterion(F.log_softmax(output, dim=2), labels) # [B S 15]
        loss = criterion(output, labels) # [B S 15]

        losses.update(loss.item(), img_feat.size()[0])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - t_start)
        t_start = time.time() # reset timer
        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(output)
            # print(losses.val, losses.avg, losses.sum, losses.count)
            log.write(output + '\n')
            log.flush()
    
    tb_writer.add_scalar('loss/train', losses.avg, epoch)
    tb_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, accuracy, epoch, log, tb_writer):
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

            output = model(img_feat, au_feat) # [Clip S 15]
            # rearrange and remove extra padding in the end
            output = rearrange(output, 'Clip S C -> (Clip S) C')[:frame_count]
            labels = rearrange(labels, 'Clip S C -> (Clip S) C')[:frame_count]

            loss = criterion(output, labels) 
            mean_cor, cor = accuracy(output, labels) # mean and per-class correlation
            # update statistics
            losses.update(loss.item())
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

if __name__ == '__main__':
    main()