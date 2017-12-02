#!/usr/bin/python
from PIL import Image
import os,sys
import matplotlib.image as mpimg
import numpy as np

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def load_n_images_groundtruth(image_dir, gt_dir, n):
    # Loaded a set of images
    files = os.listdir(image_dir)
    n = min(n, len(files)) # Load maximum 20 images
    imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])

    gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
    return imgs, gt_imgs

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def create_linearized_patches(X, patchSize):
    img_patches = [img_crop(X[i], patchSize, patchSize) for i in range(X.shape[0])]
    return np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

def create_linearized_patches_with_dimensions(X, patchSize):
    img_patches = [(img_crop(X[i], patchSize, patchSize), int(X[i].shape[0]/patchSize), int(X[i].shape[1]/patchSize)) for i in range(X.shape[0])]
    res =  np.asarray([(img_patches[i][0][j], img_patches[i][1], img_patches[i][2]) for i in range(len(img_patches)) for j in range(len(img_patches[i][0]))])
    return res

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def load_test_images():
    imgs = np.asarray([load_image('../test_set_images/test_'+str(i)+'/test_'+str(i)+'.png') for i in range(1, 51)])
    return imgs