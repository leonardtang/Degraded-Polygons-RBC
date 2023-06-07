#!/bin/sh

for r in 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7
do
    python shape_generator.py -d images224/shapes_1000_$r -r $r
done

python shape_generator.py -d images224/shapes -w