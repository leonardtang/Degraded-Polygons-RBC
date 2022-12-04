import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# Create torch dataset from images
class ShapesDataset(torch.utils.data.Dataset):

    def __init__(self, metadata, data_dir, transform=None):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(os.path.join(data_dir, metadata))
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # Read images and targets
        img_path = os.path.join(self.data_dir, self.metadata.iloc[index, 0])
        # print("img_path", img_path)
        image = Image.open(img_path)
        label = self.metadata.iloc[index, 1]
        
        # # Convert image to channels first
        # image = np.transpose(image, (2, 0, 1))

        # Transform if requested
        if self.transform:
            image = self.transform(image)

        return image, label

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

        # print("RES array", res)
        return res


def train(train_loader, model, criterion, optimizer, scheduler):
    
    model.train()
    top_1, top_5 = 0, 0
    train_loss = 0.0

    for i, (images, target) in enumerate(train_loader):

        x = images.to(device)
        y = target.to(device)

        # print(f'**********IMAGE: {x[0]}')
        # print(f'**********TARGET: {y}')
        # print(f'**********TARGET TYPE: {type(y)}')

        logits = model(x)
        # print(f'**********LOGITS: {logits}')
        loss = criterion(logits, y)
        # print(f'**********LOSS: {loss}')
        output, target = logits, y 
        # input()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        train_loss += loss.item()
        top_1 += acc1[0]
        top_5 += acc5[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return train_loss / len(train_loader), top_1 / len(train_loader), top_5 / len(train_loader)


def val(test_loader, model, criterion, args):

    model.eval()
    top_1, top_5 = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            val_loss += loss.item()
            top_1 += acc1[0]
            top_5 += acc5[0]

    return val_loss/ len(test_loader), top_1 / len(test_loader), top_5 / len(test_loader)

# For v1 shapes dataset: mean = [0.8449, 0.8449, 0.8449] and std = [0.3620, 0.3620, 0.3620]
def dataset_stats(args):
    # Compute mean and STD of dataset
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    dummy_transforms = transforms.Compose([transforms.ToTensor()])
    dummy_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = dummy_transforms)
    dummy_loader = torch.utils.data.DataLoader(
        dummy_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True,
        # Unclear if we need drop_last, e.g. AugMix doesn't have this
        drop_last=True)

    for imgs, labels in tqdm(dummy_loader):
        psum    += imgs.sum(axis        = [0, 2, 3])
        psum_sq += (imgs ** 2).sum(axis = [0, 2, 3])

    # print("imgs[0]", imgs[0])
    # print("imgs[0].shape", imgs[0].shape)
    image_size = imgs[0].shape[1]
    # print("image_size", image_size)
    count = len(dummy_loader.dataset) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    print("Dataset Mean:", total_mean)
    print("Dataset STD:", total_std)
    return total_mean, total_std


def main():
    parser = argparse.ArgumentParser(
        description="Trains classifier on shapes dataset", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", "-d", type=str, default="./images/shapes", choices=["./images/shapes"])
    parser.add_argument("--model", "-m", type=str, default="resnet18")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--pretrained", "-p", action="store_true")
    parser.add_argument("--print-freq", "-f", type=int, default=10)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    # Hyperparams adapted from http://cs231n.stanford.edu/reports/2017/pdfs/420.pdf
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", "-wd", type=float, default=1e-4)
    args = parser.parse_args()
    print("*** Arguments: ***")
    print(" ".join(f'{k}={v}' for k, v in vars(args).items()))

    if args.data == "./images/shapes":
        mean, std = dataset_stats(args)
        tub_train_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(224), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ])
    
        tub_test_transforms = transforms.Compose([
            # transforms.CenterCrop(224),
            transforms.Resize(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Initialize train, val, test sets
        # Train on whole images, and see ability to generalize to edge/corner removed images
        train_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = tub_train_transforms)
        val_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = tub_test_transforms)
        test_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = tub_test_transforms)

        # Get train, val, test splits
        train_size = 0.6
        # total_num_train = len(train_data.metadata)
        total_num_train = len(train_data)
        indices = np.random.permutation(list(range(total_num_train)))
        split = int(np.floor(train_size * total_num_train))
        val_split = int(np.floor((train_size + (1 - train_size) / 2) * total_num_train))
        train_idx, val_idx, test_idx = indices[:split], indices[split:val_split], indices[val_split:]
        
        # Subset train, val, test sets
        train_data = torch.utils.data.Subset(train_data, indices=train_idx)
        val_data = torch.utils.data.Subset(val_data, indices=val_idx)
        test_data = torch.utils.data.Subset(test_data, indices=test_idx)

    else:
        raise Exception(f"{args.data} is not a supported dataset")

    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True)

    if args.model == "vgg16":
        model = models.vgg16(pretrained=args.pretrained)
        features = []
        for feat in list(model.features):
            features.append(feat)
            if isinstance(feat, nn.Conv2d):
                features.append(nn.Dropout(p=0.55, inplace=True))

        model.features = nn.Sequential(*features)
    elif args.model == "resnet18":
        model = models.resnet18(pretrained=args.pretrained)
    else:
        raise Exception(f"{args.model} is not a supported model")

    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        weight_decay=args.decay,
        # nesterov=True
    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: cosine_annealing( 
    #         step,
    #         args.epochs * len(train_loader),
    #         1,  
    #         1e-6 / args.learning_rate
    #     )
    # )

    lin_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_log = os.path.join(args.save_dir, "training_log.csv")
    # Clear out previous training run contents
    with open(train_log, 'w+'):
        pass

    # MAIN TRAINING LOOP
    best_top5 = 0
    print("\nTraining Model...")
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        train_losses_avg, train_top1_avg, train_top5_avg = train(train_loader, model, criterion, optimizer, None)
        print('Train * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=train_top1_avg, top5=train_top5_avg))
        val_losses_avg, val_top1_avg, val_top5_avg = val(val_loader, model, criterion, args)
        print('Val * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=val_top1_avg, top5=val_top5_avg))

        if lin_scheduler:
            lin_scheduler.step(val_losses_avg)

        os.makedirs(args.save_dir, exist_ok = True)          
        with open(train_log, "a") as f:
          f.write(f'{(epoch + 1)},{train_losses_avg},{train_top1_avg},{train_top5_avg},{val_losses_avg},{val_top1_avg},{val_top5_avg}\n')

        if val_top5_avg > best_top5:
            best_top5 = max(val_top5_avg, best_top5)
            save_file = os.path.join(args.save_dir, "final_model.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_top5': best_top5,
                'optimizer' : optimizer.state_dict(),
            }, filename=save_file)

    best_cp = torch.load(save_file)
    model.load_state_dict(best_cp["state_dict"])
    test_losses_avgs, test_top1_avg, test_top5_avg = val(test_loader, model, criterion, args)
    print(f'Test * Acc@1 {test_top1_avg:.3f} Acc@5 {test_top5_avg:.3f}')

    # Plot training curves
    df = pd.read_csv(train_log, header=None)
    df.columns = ['epoch', 'train_loss', 'train_acc1', 'train_acc5', 'val_loss', 'val_acc1', 'val_acc5']

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='train')
    plt.plot(df['epoch'], df['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc1'], label='train')
    plt.plot(df['epoch'], df['val_acc1'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("acc-loss-v3.png")
    
if __name__ == "__main__":
    main()