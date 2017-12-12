import numpy as np
import matplotlib.pyplot as plt
import os,sys
from image_helpers import *
from classification import *
from cnn_model import CNN
from classification import predict_test_set_images

def main():
    #Global variables
    image_dir = "../training/images/"
    gt_dir = "../training/groundtruth/"
    files = os.listdir(image_dir)

    patch_size = 16
    foregroud_threshold = 0.25

    # Loading a set of images with their groundtruth
    imgs, gt_imgs = load_n_images_groundtruth(image_dir, gt_dir, 100, rotate=True)

    model = CNN(patch_size, foregroud_threshold)

    try:
        model.train(imgs, gt_imgs)
    except KeyboardInterrupt:
        pass

    model.save_weights('ahahah.h5')

    predict_test_set_images('test.csv', model, cnn=True)

if __name__ == "__main__":
    main()
