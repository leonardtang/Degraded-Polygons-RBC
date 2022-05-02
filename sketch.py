"""
Main script to launch experiments on TU Berlin sketch dataset

Supports VGG-Net.
""" 

import argparse
import os
import time
import torch
import numpy as np
from meter_utils import AverageMeter, ProgressMeter
from torchvision import datasets, models, transforms
from torch import nn

# https://discuss.pytorch.org/t/nan-loss-coming-after-some-time/11568/17
torch.autograd.set_detect_anomaly(True)

# https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/3
class LazySketchDataset(torch.utils.data.Dataset):
    """
    Wrapper for TU Berlin Sketch dataset. Lazy loading to allow for separate transformations.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)
    
def save_checkpoint(state, filename):
    torch.save(state, filename)

def cosine_annealing(step, total_steps, lr_max, lr_min): 
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the top k predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        print("RES array", res)
        return res


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    start = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - start)

        x = images.cuda()
        y = target.cuda()

        logits = model(x)
        loss = criterion(logits, y)
        output, target = logits, y 

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - start)

        if i % args.print_freq == 0:
            progress.display(i)

        print('Train * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    
    return losses.avg, top1.avg, top5.avg


def val(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: '
    )

    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - start)

            if i % args.print_freq == 0:
                progress.display(i)

        print('Val * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(
        description="Trains classifier on TU Berlin Sketch Dataset", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", "-d", type=str, default="./data/png", choices=["./data/png"])
    parser.add_argument("--model", "-m", type=str, default="vgg16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--pretrained", "-p", action="store_true")
    parser.add_argument("--print-freq", "-f", type=int, default=10)
    parser.add_argument("--gpu", "-g", action="store_true")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    # Hyperparams adapted from http://cs231n.stanford.edu/reports/2017/pdfs/420.pdf
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", "-wd", type=float, default=0.01)
    args = parser.parse_args()

    if args.data == "./data/png":
        tub_train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # TODO: properly calculate image statistics
            transforms.Normalize([0.5] * 3, [0.5] * 3) 
        ])
        tub_test_transforms = transforms.Compose([
            transforms.Resize(255), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # TODO: properly calculate image statistics
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        dataset = datasets.ImageFolder(args.data)
        train_lazydata = LazySketchDataset(dataset, tub_train_transforms)
        val_lazydata = LazySketchDataset(dataset, tub_test_transforms)
        test_lazydata = LazySketchDataset(dataset, tub_test_transforms)

        train_size = 0.8
        num_train = len(dataset)
        indices = np.random.permutation(list(range(num_train)))
        split = int(np.floor(train_size * num_train))
        val_split = int(np.floor((train_size + (1 - train_size) / 2) * num_train))
        train_idx, val_idx, test_idx = indices[:split], indices[split:val_split], indices[val_split:]
        
        train_data = torch.utils.data.Subset(train_lazydata, indices=train_idx)
        val_data = torch.utils.data.Subset(val_lazydata, indices=val_idx)
        test_data = torch.utils.data.Subset(test_lazydata, indices=test_idx)

    else:
        raise Exception(f"{args.data} is not a supported dataset")

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True,
        # Unclear if we need drop_last, e.g. AugMix doesn't have this
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True,
        # Unclear if we need drop_last, e.g. AugMix doesn't have this
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True,
        # Unclear if we need drop_last, e.g. AugMix doesn't have this
        drop_last=True)

    if args.model == "vgg16":
        model = models.vgg16(pretrained=args.pretrained)
        features = []
        for feat in list(model.features):
            features.append(feat)
            if isinstance(feat, nn.Conv2d):
                features.append(nn.Dropout(p=0.5, inplace=True))

        model.features = nn.Sequential(*features)
        print(model)
    else:
        raise Exception(f"{args.model} is not a supported model")


    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing( 
            step,
            args.epochs * len(train_loader),
            1,  
            1e-6 / args.learning_rate
        )
    )

    # MAIN TRAINING LOOP
    best_acc1 = 0
    for epoch in range(args.epochs):
        train_losses_avg, train_top1_avg, train_top5_avg = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
        val_losses_avg, val_top1_avg, val_top5_avg = val(val_loader, model, criterion, args)

        with open(os.path.join(args.save_dir, "training_log.csv"), "a+") as f:
            f.write('%03d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_losses_avg, train_top1_avg, train_top5_avg,
                val_losses_avg, val_top1_avg, val_top5_avg
            )
        )

        best_acc1 = max(val_top5_avg, best_acc1)
        save_file = os.path.join(args.save_dir, "final_model.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, filename=save_file)

    best_cp = torch.load(save_file)
    best_model = model.load_state_dict(best_cp["state_dict"])
    test_losses_avgs, test_top1_avg, test_top5_avg = val(test_loader, best_model, criterion, args)
    print(f'Test * Acc@1 {test_top1_avg} Acc@5 {test_top5_avg}')
    
if __name__ == "__main__":
    main()

    

