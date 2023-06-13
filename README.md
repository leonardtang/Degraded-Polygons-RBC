# Degraded-Polygons-RBC

Repository for the paper

[Degraded Polygons Raise Fundamental Questions of Neural Network Perception](https://arxiv.org/abs/2306.04955)

by Leonard Tang and Dan Ley

### Shape Generation

Datasets of degraded polygons can be generated using the following command

```
python3 shape_generator.py --num-shapes-per-class, -n <N>
                           --img-size, -s <S>
                           --min-radius-div, -m <M>
                           --remove-prop, -r, <R>
                           --thickness, -t, <T>
                           --save-dir, -d <D>
                           [--generate-whole, -w]
```

Parameters:
- ```N``` &mdash; number of shapes to generate per class (default: 1000)
- ```S``` &mdash; length of square images (default: 224)
- ```M``` &mdash; minimum radius of polygon circumcircles (default: 5)
- ```R``` &mdash; proportion of polygons to remove (default: 0.3)
- ```T``` &mdash; thickness of polygons (default: 2)
- ```D``` &mdash; save directory of dataset (default: './images224/shapes')

Use the ```--generate-whole``` flag to save non-degraded polygons

### Training Vision Models

The vision models used in our experiments can be trained with the following command

```
python3 train.py --data, -d <DATA>
                 --model, -m <MODEL>
                 --learning-rate, -lr <LEARNING_RATE>
                 --momentum <MOMENTUM>
                 --batch-size, -b <BATCH_SIZE>
                 --epochs, -e <EPOCHS>
                 --decay, -wd <DECAY>
                 --save-dir <SAVE_DIR>
                 --num-workers, -w <NUM_WORKERS>
                 --pretrained-path, --p <PRETRAINED_PATH>
                 [--pretrained-imagenet, -pi]
                 [--evaluate, -e]
```

Parameters:
- ```DATA``` &mdash; directory of dataset (default: './images224/shapes')
- ```MODEL``` &mdash; name of model to train e.g. 'resnet18', 'resnet50', 'mlpmixer', or 'vit' (default: 'resnet18')
- ```LEARNING_RATE``` &mdash; initial learning rate of optimizer (default: 0.01)
- ```MOMENTUM``` &mdash; momentum of optimizer (default: 0.9)
- ```BATCH_SIZE``` &mdash; batch size of optimizer (default: 512)
- ```EPOCHS``` &mdash; number of training epochs (default: 120)
- ```DECAY``` &mdash; weight decay value (default: 0.0001)
- ```SAVE_DIR``` &mdash; save directory of model checkpoints (default: './checkpoints224')
- ```NUM_WORKERS``` &mdash; number of workers for PyTorch (default: 16)
- ```PRETRAINED_PATH``` &mdash; optional path to pretrained models

Use the ```--pretrained-imagenet``` flag to use a model pretrained on ImageNet-1K, and the ```--evaluate``` flag to skip training and go straight to evaluation (accuracy on whole shapes, accuracy on degraded shapes, breakdown per class, etc.)
