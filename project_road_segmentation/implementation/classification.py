#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from image_helpers import *

def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def predict_and_display_image(model, img, gt, real_img):
    Zi = model.predict(img)
    
    w = gt.shape[0]
    h = gt.shape[1]
    predicted_im = label_to_img(w, h, model.patchSize, model.patchSize, Zi)
    cimg = concatenate_images(real_img, predicted_im)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')

    new_img = make_img_overlay(real_img, predicted_im)

    plt.imshow(new_img)