import numpy as np
from svg.path import parse_path
from svg.path.path import Line, Arc, QuadraticBezier, CubicBezier
from xml.dom import minidom

ex_svg = "data/svg/airplane/1.svg"

# read the SVG file
doc = minidom.parse(ex_svg)
path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
doc.unlink()

# print the line draw commands
splines = []
for path_string in path_strings:
    path = parse_path(path_string)
    # print(path_string, len(path))
    for e in path:
        if isinstance(e, (Line,Arc,QuadraticBezier,CubicBezier)):
            x0 = e.start.real
            y0 = e.start.imag
            x1 = e.end.real
            y1 = e.end.imag
            print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))
            splines.append(e)





# hi