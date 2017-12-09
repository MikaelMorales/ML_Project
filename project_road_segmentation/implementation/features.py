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

def extract_img_features_with_neighbors(filename, patchSize):
    img = load_image(filename)
    img_patches_with_dim = create_linearized_patches_with_dimensions([img], patchSize)
    return extract_features_img_patches_with_neighbors(img_patches_with_dim)[0]

def extract_features_from_patches(img_patches):
    return np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])

def extract_features_from_gt_patches(gt_patches, foreground_threshold):
    return np.asarray([value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

def extract_features_img_patches_with_neighbors(img_patches_with_dim):
    Xm_v = extract_features_from_patches([i[0] for i in img_patches_with_dim])
    return np.asarray([extract_features_with_neighbors(img_patches_with_dim[i], i, Xm_v) for i in range(len(img_patches_with_dim))])

def extract_features_with_neighbors(img_patches_with_dim, idx, Xm_v):
    ppl = img_patches_with_dim[1] # patches per line
    ppc = img_patches_with_dim[2] # patches per column

    l = idx / ppl
    l = int(l)
    l = l % ppc
    c = idx % ppl # column index

    feat = extract_features(img_patches_with_dim[0])

    if l > 0:
        feat = np.append(feat, Xm_v[idx - ppl, :3])
        if c > 0:
            feat = np.append(feat, Xm_v[idx - (ppl + 1), :3])
        else:
            feat = np.append(feat, np.zeros(3))

        if c < ppl - 1:
            feat = np.append(feat, Xm_v[idx - (ppl - 1), :3])
        else:
            feat = np.append(feat, np.zeros(3))
    else:
        feat = np.append(feat, np.zeros(9))

    if l < ppc - 1:
        feat = np.append(feat, Xm_v[idx + ppl, :3])
        if c > 0:
            feat = np.append(feat, Xm_v[idx + (ppl - 1), :3])
        else:
            feat = np.append(feat, np.zeros(3))

        if c < img_patches_with_dim[1] - 1:
            feat = np.append(feat, Xm_v[idx + (ppl + 1), :3])
        else:
            feat = np.append(feat, np.zeros(3))
    else:
        feat = np.append(feat, np.zeros(9))

    if c > 0:
        feat = np.append(feat, Xm_v[idx - 1, :3])
    else:
        feat = np.append(feat, np.zeros(3))

    if c < img_patches_with_dim[1] - 1:
        feat = np.append(feat, Xm_v[idx + 1, :3])
    else:
        feat = np.append(feat, np.zeros(3))

    return feat
