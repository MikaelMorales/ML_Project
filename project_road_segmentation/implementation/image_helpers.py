#!/usr/bin/python
from PIL import Image
import os,sys
import matplotlib.image as mpimg
import numpy as np

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def load_n_images_groundtruth(image_dir, gt_dir, n, rotate=False):
    ''' Load n images with their groundtruth'''
    # Loaded a set of images
    files = os.listdir(image_dir)
    n = min(n, len(files)) # Load maximum 20 images

    if not rotate:
        imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
        gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
    else:
        imgs = []
        gt_imgs = []
        for i in range(n):
            img = load_image(image_dir + files[i])
            imgs.append(img)
            gt_img = load_image(gt_dir + files[i])
            gt_imgs.append(gt_img)
            for k in range(1, 4):
                imgs.append(np.rot90(img, k))
                gt_imgs.append(np.rot90(gt_img, k))

        imgs = np.asarray(imgs)
        gt_imgs = np.asarray(gt_imgs)

    return imgs, gt_imgs

def img_float_to_uint8(img):
    '''Convert image to uint8'''
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    '''Concatenate an image and its groundtruth'''
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
    '''Crop the image to generate a list of patches of size (w,h)'''
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

def img_crop_with_padding(im, old, w, h):
    '''Crop the image to generate a list of patches of size (5*w,5*h)'''
    list_patches = []
    imgwidth = old.shape[0]
    imgheight = old.shape[1]
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im_patch = im[j:j+(5*w), i:i+(5*h), :]
            list_patches.append(im_patch)
    return list_patches

def create_linearized_patches(X, patchSize):
    ''' Utility method that generate patches from a given image and linearize the result'''
    img_patches = [img_crop(X[i], patchSize, patchSize) for i in range(X.shape[0])]
    return np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

def create_linearized_patches_with_dimensions(X, patchSize):
    img_patches = [(img_crop(X[i], patchSize, patchSize), int(X[i].shape[0]/patchSize), int(X[i].shape[1]/patchSize)) for i in range(X.shape[0])]
    res =  np.asarray([(img_patches[i][0][j], img_patches[i][1], img_patches[i][2]) for i in range(len(img_patches)) for j in range(len(img_patches[i][0]))])
    return res

def label_to_img(imgwidth, imgheight, w, h, labels):
    '''Convert array of labels to an image'''
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    '''Overlay the two given images'''
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
    '''Load the entire set of test images'''
    imgs = np.asarray([load_image('../test_set_images/test_'+str(i)+'/test_'+str(i)+'.png') for i in range(1, 51)])
    return imgs

def post_processing(predicted_img):
    '''Update the value of patches if it is surrounded by patches of the same value
    Example: If a patche has value 0 and is surrounded by patches with value 1,
             this patch will take value 1.'''
    processed = predicted_img
    l, c = predicted_img.shape
    for i in range(l):
        for j in range(c):
            if is_alone(i, j, l, c, predicted_img, predicted_img[i][j]):
                if predicted_img[i][j] == 1:
                    processed[i][j] = 0
                else:
                    processed[i][j] = 1

    return processed

def is_alone(i, j, l, c, predicted_labels, value):
    '''Return True if the patche at position (i,j) is surrounded by patches with the same value as the
    parameter value, otherwise it returns False'''
    if i > 0:
        if predicted_labels[i-1][j] == value: # the patch above
            return False
        if j > 0 and predicted_labels[i-1][j-1] == value: # the patch on the left diagonal above
            return False
        if j < (c-1) and predicted_labels[i-1][j+1] == value: # the patch on the right diagonal above
            return False

    if i < (l-1):
        if predicted_labels[i+1][j] == value: # the patch bellow
            return False
        if j > 0 and predicted_labels[i+1][j-1] == value: # the patch on the left diagonal below
            return False
        if j < (c-1) and predicted_labels[i+1][j+1] == value: # the patch on the right diagonal below
            return False

    if j > 0 and predicted_labels[i][j-1] == value: # the patch on the left
        return False

    if j < (c-1) and predicted_labels[i][j+1] == value: # the patch on the right
        return False

    return True

