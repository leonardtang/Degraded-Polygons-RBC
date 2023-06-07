import argparse
import copy
import os
import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from shape_generator import classes, classes_to_string
from train import get_noedges_loader, get_nocorners_loader
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model == "vgg16":
        model = models.vgg16()
        model.fc = torch.nn.Linear(512, len(classes))
        model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

    elif args.model == "resnet18":
        model = models.resnet18()
        model.fc = torch.nn.Linear(512, len(classes))
        model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

    else:
        raise Exception(f"{args.model} is not a supported model")
        
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
    
    num_indices = len(get_noedges_loader(args, sampler=None)) * args.batch_size
    sampler_a = list(SubsetRandomSampler(list(range(num_indices))))
    sampler_b = copy.deepcopy(sampler_a)
    noedges_loader = get_noedges_loader(args, sampler=sampler_a)
    nocorners_loader = get_nocorners_loader(args, sampler=sampler_b)

    for images, labels in noedges_loader:
        grayscale_noedges_cam = cam(input_tensor=images, targets=None)
        for i in range(len(labels)):
            label = labels[i]
            noedges_cam = grayscale_noedges_cam[i, :]
            image = images[i].type(torch.float32)
            image = torch.permute(image, (1, 2, 0))
            image = image.detach().numpy()
            
            viz = show_cam_on_image(image, noedges_cam, use_rgb=True)
            im = Image.fromarray(viz)
            shape_label = classes_to_string[label.detach().item()]
            im.save(f"{args.out_dir}/{i}_noedges_label_{shape_label}.png")

        break

    for images, labels in nocorners_loader:
        grayscale_nocorners_cam = cam(input_tensor=images, targets=None)
        for i in range(len(labels)):
            label = labels[i]
            nocorners_cam = grayscale_nocorners_cam[i, :]
            image = images[i].type(torch.float32)
            image = torch.permute(image, (1, 2, 0))
            image = image.detach().numpy()
            
            viz = show_cam_on_image(image, nocorners_cam, use_rgb=True)
            im = Image.fromarray(viz)
            shape_label = classes_to_string[label.detach().item()]
            im.save(f"{args.out_dir}/{i}_nocorners_label_{shape_label}.png")

        break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True, default="./images/shapes")
    parser.add_argument("--out-dir", "-o", type=str, required=True, default="./viz/shapes")
    parser.add_argument("--model", "-m", type=str, default="resnet18")
    parser.add_argument("--pretrained-path", "-pp", type=str, required=True)
    parser.add_argument("--num-workers", "-w", type=int, default=4)
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--use-cuda", "-c", action="store_false")
    args = parser.parse_args()

    print("*** Arguments: ***")
    print("\n".join(f'{k}={v}' for k, v in vars(args).items()))

    main(args)
