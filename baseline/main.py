# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import time
import os
from datetime import datetime
from einops import rearrange
from tensorboardX import SummaryWriter


from args import parser
from model import Baseline, Correlation
from eev_dataset import EEV_Dataset
from utils import AverageMeter

def main():
    global args
    args = parser.parse_args()

    args.store_name = 'Baseline_GRU'
    args.store_name = args.store_name + datetime.now().strftime('_%m-%d-%Y_%H-%M')
    args.start_epoch = 0

    check_rootfolders(args)

    model = Baseline()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # initialize datasets
    train_loader = torch.utils.data.DataLoader(
        dataset=EEV_Dataset(
            csv_path=args.train_csv,
            vidmap_path=args.train_vidmap,
            image_feat_path=args.image_features,
            audio_feat_path=args.audio_features
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
            audio_feat_path=args.audio_features
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        drop_last=True
    )

    criterion = torch.nn.KLDivLoss().cuda()

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    tb_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, log_training, tb_writer)



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
        loss = criterion(output.log(), labels) # [B S 15]

        losses.update(loss)
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
            log.write(output + '\n')
            log.flush()
    
    tb_writer.add_scalar('loss/train', losses.avg, epoch)
    tb_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_ckpt,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_ckpt, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()