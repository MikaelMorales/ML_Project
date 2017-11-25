#!/usr/bin/python
from image_helpers import load_image, img_crop
from classification import *
import numpy as np

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename, patchSize):
    img = load_image(filename)
    img_patches = img_crop(img, patchSize, patchSize)
    X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X

def extract_features_from_patches(img_patches):
    return np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])

def extract_features_from_gt_patches(gt_patches, foreground_threshold):
    return np.asarray([value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])