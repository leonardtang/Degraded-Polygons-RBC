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
- ```N``` -- number of shapes to generate per class (default: 1000)
- ```S``` -- length of square images (default: 224)
- ```M``` -- minimum radius of polygon circumcircles (default: 5)
- ```R``` -- proportion of polygons to remove (default: 0.3)
- ```T``` -- thickness of polygons (default: 2)
- ```D``` -- save directory of dataset (default: './images224/shapes')

Use the ```--generate-whole``` flag to save non-degraded polygons
