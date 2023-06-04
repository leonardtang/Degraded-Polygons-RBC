import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from models.modeling import MlpMixer, CONFIGS
from shape_generator import classes
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm
from transformers import TrainingArguments, Trainer
from vit_pytorch import SimpleViT


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

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
        
        img_path = os.path.join(f'{self.data_dir}/pngs', self.metadata.iloc[index, 0])
        image = Image.open(img_path)
        label = classes[self.metadata.iloc[index, 1]]

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

        res.append(torch.max(output, 1)[1])

        return res


def train(train_loader, model, criterion, optimizer):
    
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

        acc1, acc5, _ = accuracy(output, target, topk=(1, 5))
        train_loss += loss.item()
        top_1 += acc1[0]
        top_5 += acc5[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return train_loss / len(train_loader), top_1 / len(train_loader), top_5 / len(train_loader)


def val(test_loader, model, criterion):

    model.eval()
    top_1, top_5 = 0, 0
    val_loss = 0.0
    per_class_acc = [0 for _ in classes]

    with torch.no_grad():
        confusion_matrix = torch.zeros(len(classes), len(classes))
        for _, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            
            acc1, acc5, preds = accuracy(output, target, topk=(1, 5))
            val_loss += loss.item()
            top_1 += acc1[0]
            top_5 += acc5[0]

            ### DEBUGGING CODE
            # for i, t in enumerate(target):
            #     if preds[i] == t: continue
            #     if t == 4:  
            #         print("Pred instead of 4", preds[i])
            #     if t == 5:  
            #         print("Pred instead of 5", preds[i])
            #     if t == 6:  
            #         print("Pred instead of 6", preds[i])

            for t, p in zip(torch.flatten(target), torch.flatten(preds)):
                confusion_matrix[t.long(), p.long()] += 1

    per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    return val_loss/ len(test_loader), top_1 / len(test_loader), top_5 / len(test_loader), per_class_acc.tolist()


def dataset_stats(args, metadata_path):
    """
    Compute mean and STD of dataset
    For shapes-1000 dataset: mean = [0.9857, 0.9857, 0.9857] and std = [0.1188, 0.1188, 0.1188]
    """

    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    dummy_transforms = transforms.Compose([transforms.ToTensor()])
    dummy_data = ShapesDataset(metadata=metadata_path, data_dir=args.data, transform=dummy_transforms)
    dummy_loader = torch.utils.data.DataLoader(
        dummy_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True)

    for imgs, _ in tqdm(dummy_loader):
        psum    += imgs.sum(axis        = [0, 2, 3])
        psum_sq += (imgs ** 2).sum(axis = [0, 2, 3])

    image_size = imgs[0].shape[1]
    count = len(dummy_loader.dataset) * image_size * image_size

    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    print("Dataset Mean:", total_mean)
    print("Dataset STD:", total_std)
    return total_mean, total_std


def get_noedges_loader(args, sampler=None):
    noedges_mean, noedges_std = dataset_stats(args, "metadata_noedges.csv")
    noedges_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(noedges_mean, noedges_std)
        ])
    noedges_data = ShapesDataset(metadata='metadata_noedges.csv', data_dir=args.data, transform=noedges_transforms)

    if sampler:
        noedges_loader = torch.utils.data.DataLoader(
            noedges_data, 
            batch_size=args.batch_size,
            num_workers=args.num_workers, 
            sampler=sampler)
    else:
        noedges_loader = torch.utils.data.DataLoader(
            noedges_data, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)

    return noedges_loader


def get_nocorners_loader(args, sampler=None):
    nocorners_mean, nocorners_std = dataset_stats(args, "metadata_nocorners.csv")
    noedges_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(nocorners_mean, nocorners_std)
        ])
    nocorners_data = ShapesDataset(metadata='metadata_nocorners.csv', data_dir=args.data, transform=noedges_transforms)
    
    if sampler:
        nocorners_loader = torch.utils.data.DataLoader(
            nocorners_data, 
            batch_size=args.batch_size,
            num_workers=args.num_workers, 
            sampler=sampler)
    else:
        nocorners_loader = torch.utils.data.DataLoader(
            nocorners_data, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)

    return nocorners_loader


def main():
    parser = argparse.ArgumentParser(
        description="Trains classifier on shapes dataset", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", "-d", type=str, default="./images224/shapes")
    parser.add_argument("--model", "-m", type=str, default="resnet18")
    parser.add_argument("--num-workers", "-w", type=int, default=16)
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--pretrained-imagenet", "-pi", action="store_true")
    parser.add_argument("--pretrained-path", "-pp", type=str)
    parser.add_argument("--print-freq", "-f", type=int, default=10)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    # Hyperparams adapted from http://cs231n.stanford.edu/reports/2017/pdfs/420.pdf
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--evaluate", "-e", action="store_true")
    args = parser.parse_args()
    print("*** Arguments: ***")
    print(" ".join(f'{k}={v}' for k, v in vars(args).items()))

    # Choose only one pretraining souce
    assert not (args.pretrained_imagenet and args.pretrained_path)

    if args.model == "vgg16":
        if not args.evaluate:
            if args.pretrained_imagenet:
                model = models.vgg16(pretrained=True)
            else:
                model = models.vgg16()
            features = []
            for feat in list(model.features):
                features.append(feat)
                if isinstance(feat, nn.Conv2d):
                    features.append(nn.Dropout(p=0.55, inplace=True))
            
            model.features = nn.Sequential(*features)
            model.fc = torch.nn.Linear(512, len(classes))
        else:
            model = models.vgg16()
            model.fc = torch.nn.Linear(512, len(classes))

    elif args.model == "resnet18":
        if not args.evaluate:
            if args.pretrained_imagenet:
                model = models.resnet18(pretrained=True)
            elif args.pretrained_path:
                model = models.resnet18()
                if args.pretrained_path.endswith("FractalDB-10000_res18.pth"):
                    model.fc = torch.nn.Linear(512, 10000)        
                elif args.pretrained_path.endswith("FractalDB-1000_res18.pth"):
                    model.fc = torch.nn.Linear(512, 1000)
                try:
                    model.load_state_dict(torch.load(args.pretrained_path))
                except:
                    model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
            else:
                model = models.resnet18()
            model.fc = torch.nn.Linear(512, len(classes))
        else:
            model = models.resnet18()
            model.fc = torch.nn.Linear(512, len(classes))

    elif args.model == "resnet50":
        if not args.evaluate:
            if args.pretrained_imagenet:
                model = models.resnet50(pretrained=True)
            elif args.pretrained_path:
                raise Exception('Not implemented')
                # model = models.resnet50()
                # if args.pretrained_path.endswith("FractalDB-10000_res18.pth"):
                #     model.fc = torch.nn.Linear(2048, 10000)        
                # elif args.pretrained_path.endswith("FractalDB-1000_res18.pth"):
                #     model.fc = torch.nn.Linear(2048, 1000)
                # try:
                #     model.load_state_dict(torch.load(args.pretrained_path))
                # except:
                #     model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
            else:
                model = models.resnet50()
            model.fc = torch.nn.Linear(2048, len(classes))
        else:
            model = models.resnet50()
            model.fc = torch.nn.Linear(2048, len(classes))

    elif args.model == "mlpmixer":  # work in progress
        if not args.evaluate:
            if args.pretrained_imagenet:
                if args.pretrained_path:  # assumes imagenet1k
                    model = MLPMixer(num_classes=1000)
                    model.load_from(np.load(args.pretrained_path))
                    # re-initialize final layer
                    model.layers[-1] = nn.Linear(model.layers[-1].in_features, len(classes))
        else:
            model = MLPMixer(num_classes=len(classes))

    elif args.model == "vit":  
        # TODO(ltang): think carefully about which ViT to base off of
        
        model = SimpleViT(
            image_size = 224,
            patch_size = 32,
            num_classes = len(classes),
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )

    else:
        raise Exception(f"{args.model} is not a supported model")

    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        # nesterov=True
    )

    sub_save_dir = os.path.join(args.save_dir, args.data.split("/")[-1])
    os.makedirs(sub_save_dir, exist_ok=True)
    print("\nSub Save Dir", sub_save_dir)
    
    save_suffix = ""
    if args.pretrained_imagenet:
        save_suffix = "imagenet"
    elif args.pretrained_path:
        save_suffix = args.pretrained_path.split("/")[-1].split(".")[0]

    save_file = os.path.join(sub_save_dir, f"final_model_{save_suffix}.pth")
    print("\nSave File:", save_file)

    ### Main training loop
    if not args.evaluate:
        if args.data.startswith("./images224/shapes"):
            whole_mean, whole_std = dataset_stats(args, "metadata_whole.csv")
            tub_train_transforms = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.Resize(224), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.ToTensor(),
                transforms.Normalize(whole_mean, whole_std) 
            ])
        
            tub_test_transforms = transforms.Compose([
                # transforms.CenterCrop(224),
                # transforms.Resize(224), 
                transforms.ToTensor(),
                transforms.Normalize(whole_mean, whole_std)
            ])

            # Initialize train, val, test sets
            # Train on whole images, and see ability to generalize to edge/corner removed images
            train_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = tub_train_transforms)
            val_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = tub_test_transforms)
            test_data = ShapesDataset(metadata='metadata_whole.csv', data_dir = args.data, transform = tub_test_transforms)

            # Get train, val, test splits
            train_size = 0.6
            total_num_train = len(train_data)
            print("Total Number of Dataset Examples:", total_num_train)

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
        
        lin_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        train_log = os.path.join(sub_save_dir, "training_log.csv")
        with open(train_log, 'w+'): pass

        best_top1 = 0
        print("\nTraining Model...")

        for epoch in range(args.epochs):
            print(f"Epoch: {epoch}")

            train_losses_avg, train_top1_avg, train_top5_avg = train(train_loader, model, criterion, optimizer)
            print('Train * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(top1=train_top1_avg, top5=train_top5_avg))
            val_losses_avg, val_top1_avg, val_top5_avg, val_per_class_acc = val(val_loader, model, criterion)
            print('Val * Acc@1 {top1:.3f} Acc@5 {top5:.3f} Per-Class Acc {per_class}'.format(top1=val_top1_avg, top5=val_top5_avg, per_class=["{0:0.3f}".format(i) for i in val_per_class_acc]))

            if lin_scheduler:
                lin_scheduler.step(val_losses_avg)
          
            with open(train_log, "a") as f:
                f.write(f'{(epoch + 1)},{train_losses_avg},{train_top1_avg},{train_top5_avg},{val_losses_avg},{val_top1_avg},{val_top5_avg}\n')

            if val_top1_avg > best_top1:
                best_top1 = max(val_top1_avg, best_top1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'best_top1': best_top1,
                    'optimizer' : optimizer.state_dict(),
                }, filename=save_file)

        best_cp = torch.load(save_file)
        model.load_state_dict(best_cp["state_dict"])
        _, test_top1_avg, test_top5_avg, test_per_class_acc = val(test_loader, model, criterion)
        print(f'Test * Acc@1 {test_top1_avg:.3f} Acc@5 {test_top5_avg:.3f} Per-Class Acc {["{0:0.3f}".format(i) for i in test_per_class_acc]}')

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


    ### Evaluate on no corner and no edge datasets
    if not args.evaluate:
        best_cp = torch.load(save_file)
        model.load_state_dict(best_cp["state_dict"])
    else:
        model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

    noedges_loader = get_noedges_loader(args)
    _, test_top1_avg, test_top5_avg, test_per_class_acc = val(noedges_loader, model, criterion)
    print(f'No Edges * Acc@1 {test_top1_avg:.3f} Acc@5 {test_top5_avg:.3f} Per-Class Acc {["{0:0.3f}".format(i) for i in test_per_class_acc]}')

    nocorners_loader = get_nocorners_loader(args)
    _, test_top1_avg, test_top5_avg, test_per_class_acc = val(nocorners_loader, model, criterion)
    print(f'No Corners * Acc@1 {test_top1_avg:.3f} Acc@5 {test_top5_avg:.3f} Per-Class Acc {["{0:0.3f}".format(i) for i in test_per_class_acc]}')
    
if __name__ == "__main__":
    main()