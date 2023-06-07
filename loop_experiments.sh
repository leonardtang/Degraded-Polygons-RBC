#!/bin/sh

# python train.py -d images224/shapes/ -m resnet18 -b 128 -e --pretrained-path checkpoints224/shapes/final_model_resnet18.pth
# python train.py -d images224/shapes/ -m vit -b 128 -e --pretrained-path checkpoints224/shapes/final_model_vit.pth
python train.py -d images224/shapes/ -m resnet50 -b 128 -e --pretrained-path checkpoints224/shapes/final_model_resnet50.pth

for r in 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7
do
    python train.py -d images224/shapes_1000_$r/ -m resnet50 -b 128 -e --pretrained-path checkpoints224/shapes/final_model_resnet50.pth
done