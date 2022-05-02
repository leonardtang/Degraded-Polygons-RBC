"""
Construct non-recoverable images in the style of the Gollin figure test.
"""

import argparse
import itertools
import math
import operator
import os
import re
import cv2 as cv
import numpy as np


def file_walk(path, file_extension=".png"):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension): 
                yield os.path.join(dirpath, filename)

def parse_images(in_dir):
    images = []
    file_paths = []
    i = 0
    for file_path in file_walk(in_dir):
        print(f"Parsing file number {i}")
        img = cv.imread(file_path)
        if img is not None: 
            images.append(img)
        else:
            raise Exception("Image type not supported by OpenCV")
        file_paths.append(file_path)
        i += 1
    return images, file_paths
    

def find_corners(img):
    img = np.float32(img)
    return cv.cornerHarris(img, blockSize=2, ksize=3, k=0.03)


def non_overlapping(breakranges):

    no_overlap = []
    breakranges = sorted(breakranges, key=lambda pair:pair[0])

    for i in range(len(breakranges)):
        prev_end = breakranges[i - 1][1]
        curr_start = breakranges[i][0]

        if prev_end < curr_start:
            no_overlap.append((prev_end, curr_start))

    return no_overlap


def remove_corner_neighbors(img, global_corners, corner_radius=5):
    """
    Remove neighborhood of corner within corner_radius
    """
    corner_x, corner_y = global_corners
    edit_img = np.copy(img)
    for x in range(-corner_radius + 1, corner_radius):
        for y in range(-corner_radius + 1, corner_radius):
            transform_x = corner_x + x
            transform_y = corner_y + y
            edit_img[transform_x, transform_y] = 255
    return edit_img


def remove_corners(in_dir, out_dir_name="corners", corner_radii=[5, 10, 25, 50], dst_thresh=0.1):
    images, fnames = parse_images(in_dir)
    removed = {c_radius: [] for c_radius in corner_radii}

    for i, im in enumerate(images):
        print(f"Removing corners image {i}")

        file_path = fnames[i]
        # Corner and contour algorithms need white on black
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im_inv = cv.bitwise_not(imgray)
        # Erode image for sharper corners
        corner_img = cv.erode(im_inv, np.ones((4,4), np.uint8), iterations=1)
        dst = find_corners(corner_img)
        corner_coords = np.where(dst > dst_thresh * dst.max())
        for rad in corner_radii:
            # Pass in black on white image since we are setting white values to get rid of corner regions
            im = remove_corner_neighbors(imgray, corner_coords, corner_radius=rad)
            removed[rad].append((im, file_path))

    for radius, edit_list in removed.items():
        for edited_img, fpath in edit_list:
            # edit_img = np.full(images[0].shape, fill_value=-1, dtype="uint8")
            # for cont in edit_contours:
            #     edit_img[cont[:, :, 1], cont[:, :, 0], :] = 0
            edit_img = dedup_border(edited_img)
            fpath = re.sub(r'sketches/', r'{}/bwidth-{}/'.format(out_dir_name, int(radius)), fpath)
            if not os.path.isdir(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath), exist_ok=False)
            cv.imwrite(fpath, edit_img)


def find_breakrange(cont, global_corners, num_breaks, width, corner_radius=4):
    """
    Split cont (curve) into multiple curves, constrained by corners
    """
    corner_x, corner_y = global_corners
    
    # Squeeze out mysterious middle dimension
    cont = cont.squeeze()
    cont_x, cont_y = cont[:,0].reshape((-1, 1)), cont[:,1].reshape((-1, 1))
    x_delta = np.abs(cont_x - corner_x)
    y_delta = np.abs(cont_y - corner_y)
    # Want to check if *any* of the contour points are close to a corner
    split_cont_idx = np.unique(np.where((np.amin(x_delta, axis=1) < corner_radius) & (np.amin(y_delta, axis=1) < corner_radius))[0])
    split_cont_idx = np.insert(split_cont_idx, 0, 0, axis=0)

    # Deterministic breakpoints
    cont_bp_range_idx = []
    for i, end_id in enumerate(split_cont_idx[1:]):

        cont_split = cont[split_cont_idx[i - 1]:int(end_id)]
        if len(cont_split) <= 10:
            continue

        bp_idx = [(b + 1) / (num_breaks + 1) * len(cont_split) for b in range(num_breaks)]
        bp_range_idx = [
            list(range(
                max(int(bp_id - len(cont_split) * width / 2), int(0.1 * len(cont_split))), 
                min(int(bp_id + len(cont_split) * width / 2), int(0.9 * len(cont_split)))
            ))
            for bp_id in bp_idx
        ]
        cont_bp_range_idx.extend(bp_range_idx)

    # Remove all mid-parts of contour by masking such pixels as False
    remain_cont = np.ones(len(cont), bool)
    for range_idx in cont_bp_range_idx:
        remain_cont[range_idx] = 0

    return remain_cont


def dedup_border(img):
    """
    Dilate to fill borders; erode to reduce to original thickness 
    """
    white_on_black = cv.bitwise_not(img)
    v_dilate = cv.dilate(white_on_black, np.ones((4,1), np.uint8), iterations=1)
    dilated = cv.dilate(v_dilate, np.ones((1,4), np.uint8), iterations=1)
    deduped = cv.erode(dilated, np.ones((5,5), np.uint8), iterations=1)
    thickened  = cv.dilate(deduped, np.ones((2,2), np.uint8), iterations=1)
    img = cv.bitwise_not(thickened)

    # Remove small objects: https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_objects
    return img


def contract_corners():
    """
    Turn corner blob into corner point
    """
    pass


def remove_midedge(in_dir, out_dir_name="edges", num_breaks=1, breakwidths=[0.05, 0.1, 0.15, 0.2], dst_thresh=0.05):

    if num_breaks * max(breakwidths) >= 1:
        raise Exception("Too many breakpoints with too wide breakwidths")

    images, fnames = parse_images(in_dir)
    full_contours = {b_width: [] for b_width in breakwidths}
    
    for i, im in enumerate(images):
        print(f"Removing mid-edge image {i}")

        file_path = fnames[i]
        # Corner and contour algorithms need white on black
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im_inv = cv.bitwise_not(imgray)
        dst = find_corners(im_inv)
        corner_coords = np.where(dst > dst_thresh * dst.max())

        _, thresh = cv.threshold(im_inv, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        for width in breakwidths:
            sparse_contours = []
            for cont in contours: 
                remain_cont = find_breakrange(cont, corner_coords, num_breaks, width, corner_radius=5)
                # Separate continuous chunks of 0 into split arrays (e.g. 00011000 --> [000], [000])
                keep_split = [[i for i, value in it] for key, it in itertools.groupby(enumerate(remain_cont), key=operator.itemgetter(1)) if key != 0]
                for split_idx in keep_split:
                    sparse_contours.append(cont[split_idx])

            full_contours[width].append((sparse_contours, file_path))        

    for width, edit_list in full_contours.items():
        for edit_contours, fpath in edit_list:
            edit_img = np.full(images[0].shape, fill_value=-1, dtype="uint8")
            for cont in edit_contours:
                edit_img[cont[:, :, 1], cont[:, :, 0], :] = 0
            edit_img = dedup_border(edit_img)
            fpath = re.sub(r'sketches/', r'{}/bwidth-{}/'.format(out_dir_name, int(width * 100)), fpath)
            if not os.path.isdir(os.path.dirname(fpath)):
                os.makedirs(os.path.dirname(fpath), exist_ok=False)
            cv.imwrite(fpath, edit_img)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", type=str, default="./data/png")
    parser.add_argument("--corner-out", "-c", type=str, default="corners")
    parser.add_argument("--edge-out", "-e", type=str, default="edges")
    args = parser.parse_args()
    
    remove_midedge(args.data_dir, args.edge_out)
    remove_corners(args.data_dir, args.corner_out)


if __name__ == "__main__":
    main()
