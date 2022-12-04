import argparse
import cv2
import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CIRCLE = 1

def compute_obstruction_radius(num_sides, remove_prop):
    """
    For fixed remove_prop on the global image, determine appropriate local obstruction radius
    """
    r = remove_prop / (2 * num_sides)
    return r

def generate_metadata(save_dir):
    """
    Generate metadata DF for ShapesDataset
    """
    
    for shape_type in ['whole', 'nocorners', 'noedges']:
        
        df = pd.DataFrame(columns=['filename', 'label'])
        for path in tqdm(os.listdir(save_dir)):
            if path.endswith(f'{shape_type}.png'):
                df = pd.concat([df, pd.DataFrame([[path, int(path.split('_')[0])]], columns=['filename', 'label'])])

        df.to_csv(f'{save_dir}/metadata_{shape_type}.csv', index=False)


def main(classes, num_per_class, img_size, min_radius, thickness, bg_color, fg_color, save_dir, prop_to_remove):

    for num_sides in classes:
        for k in tqdm(range(num_per_class)):
            
            img = np.zeros((img_size, img_size, 3), np.uint8)
            img[:] = bg_color
            
            # Randomly select a point on the image to be shape center
            x = np.random.randint(min_radius + math.ceil(thickness / 2) + 1, img_size - min_radius - math.ceil(thickness / 2))
            y = np.random.randint(min_radius + math.ceil(thickness / 2) + 1, img_size - min_radius - math.ceil(thickness / 2))
            
            max_radius = min(x, y, img_size - x, img_size - y)
            radius = np.random.randint(min_radius, max_radius)
            
            # Initialize angle of one corner point
            angle = np.random.randint(0, 360)
            angle = angle * np.pi / 180
            
            if num_sides > 2:
                
                # Original generated shape
                angles = [angle + 2 * np.pi * i / num_sides for i in range(num_sides)]
                points = np.array([(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles], np.int32)
                img = cv2.polylines(img, [points], True, fg_color, thickness)
        
                cv2.imwrite(f'{save_dir}/pngs/{num_sides}_{k}_whole.png', img)
                
                img_nocorners = img.copy()
                for corner_x, corner_y in points:
                    # Obstruct corners by overlaying white circle on top
                    cv2.circle(img_nocorners, (corner_x, corner_y), math.ceil(compute_obstruction_radius(num_sides, prop_to_remove)), bg_color, -1)
                
                cv2.imwrite(f'{save_dir}/pngs/{num_sides}_{k}_nocorners.png', img_nocorners)

                img_noedges = img.copy()
                for i in range(num_sides):
                    # Obstruct midpoint of edges
                    cv2.circle(
                        img_noedges, 
                        (int((points[i][0] + points[(i + 1) % num_sides][0]) / 2), int((points[i][1] + points[(i + 1) % num_sides][1]) / 2)), 
                        math.ceil(compute_obstruction_radius(num_sides, prop_to_remove)), 
                        bg_color, 
                        -1,
                    )

                cv2.imwrite(f'{save_dir}/pngs/{num_sides}_{k}_noedges.png', img_noedges)

            elif num_sides == CIRCLE:

                # Only generate whole image for circle, since there are no corners/midpoints
                img = cv2.circle(img, (x, y), radius, fg_color, thickness)
                cv2.imwrite(f'{save_dir}/pngs/{num_sides}_{k}_whole.png', img)

            else:
                raise Exception(f"Invalid number of sides: {num_sides}")

    generate_metadata(save_dir)

if __name__ == "__main__":
    classes = [CIRCLE, 3, 4, 5, 6, 7, 8]
    bg_color = WHITE
    fg_color = BLACK
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shapes-per-class", "-n", default=1000)
    parser.add_argument("--img-size", "-s", default=224)
    parser.add_argument("--min-radius-div", "-m", default=5)
    parser.add_argument("--remove-prop", "-r", default=0.3)
    parser.add_argument("--thickness", "-t", default=2)
    parser.add_argument("--save-dir", "-d", default="./images/shapes_1000")

    args = parser.parse_args()
    print_args = "\n".join(f'{k}={v}' for k, v in vars(args).items())
    
    if not os.path.exists(args.save_dir):
        os.makedirs(f"{args.save_dir}/pngs")

    with open(f"{args.save_dir}/generator_args.txt", "w+") as f:
        f.write(print_args)
    
    min_radius = int(args.img_size / args.min_radius_div)
    main(classes, args.num_shapes_per_class, args.img_size, min_radius, args.thickness, bg_color, fg_color, args.save_dir, args.remove_prop)