#!/usr/bin/python
from image_helpers import *
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

# Extracr 8-dimensional features consisting of average RGB color as well as variance and indexes
def extract_features_8d(img,idx,patch_size, length):
    feat = extract_features(img)
    n = length / patch_size
    feat_i = (idx / n ) % 25
    feat = np.append(feat, feat_i)
    feat_j = idx % n
    feat = np.append(feat,feat_j)
    return feat

# Extract features for a given image
def extract_img_features(filename, patchSize):
    img = load_image(filename)
    img_patches = img_crop(img, patchSize, patchSize)
    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X


# Extract features for a given image
def extract_img_features_8d(filename, patchSize):
    img = load_image(filename)
    img_patches = img_crop(img, patchSize, patchSize)
    X = np.asarray([extract_features_8d(img_patches[i],i,patchSize,400) for i in range(len(img_patches))])
    return X