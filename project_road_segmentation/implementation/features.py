#!/usr/bin/python
from image_helpers import load_image, img_crop
from classification import value_to_class
import numpy as np

def extract_features(img):
    '''Extract 6-dimensional features consisting of average RGB color as well as variance'''
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_2d(img):
    '''Extract 2-dimensional features consisting of average gray color as well as variance'''
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_img_features(filename, patchSize):
    '''Extract features for a given file'''
    img = load_image(filename)
    img_patches = img_crop(img, patchSize, patchSize)
    X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X

def extract_img_features_with_neighbors(filename, patchSize):
    '''Utility method to extract patches from a file taking into account the
    neighbors of each patch'''
    img = load_image(filename)
    img_patches_with_dim = create_linearized_patches_with_dimensions([img], patchSize)
    return extract_features_img_patches_with_neighbors(img_patches_with_dim)[0]

def extract_features_from_patches(img_patches):
    '''Utility method to extract patches from a list of images'''
    return np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])

def extract_features_from_gt_patches(gt_patches, foreground_threshold):
    '''Utility method to extract patches from a list of groundtruth and convert them to label'''
    return np.asarray([value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

def extract_features_img_patches_with_neighbors(img_patches_with_dim):
    '''Utility method to extract patches from a list of images by taking the
    neighbors of the patch into account'''
    Xm_v = extract_features_from_patches([i[0] for i in img_patches_with_dim])
    return np.asarray([extract_features_with_neighbors(img_patches_with_dim[i], i, Xm_v) for i in range(len(img_patches_with_dim))])

def extract_features_with_neighbors(img_patches_with_dim, idx, Xm_v):
    '''Extract the mean of each channels (RGB) of every neighboring patches
    surrouding the patch at index idx'''
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
