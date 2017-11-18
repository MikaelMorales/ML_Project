#!/usr/bin/python
import numpy as np
from classification import value_to_class
from image_helpers import img_crop, build_color_mask

def get_patches(mask, patch_size, foreground_threshold):
    gt_patch = img_crop(mask, patch_size, patch_size)
    gt_patch = np.asarray([gt_patch[j] for j in range(len(gt_patch))])
    return np.asarray([value_to_class(np.mean(gt_patch[i]), foreground_threshold) for i in range(len(gt_patch))])

def compute_accuracy(img, predicted_im, expected_color_mask, patch_size, foreground_threshold):
    color_mask = build_color_mask(img, predicted_im)
    got_patches = get_patches(color_mask, patch_size, foreground_threshold)
    exp_patches = get_patches(expected_color_mask, patch_size, foreground_threshold)
    return np.sum(got_patches == exp_patches) / len(exp_patches)
