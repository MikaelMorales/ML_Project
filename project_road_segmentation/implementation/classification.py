#!/usr/bin/python
import numpy as np
import matplotlib.image as mpimg
from image_helpers import *
from mask_to_submission import *

def value_to_class(v, foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def predict_test_set_images(filename, model, cnn=False):
    '''Predict the entire test_set_images using the model and apply some
    post-processing before saving the images in the folder predicition_groundtruth.
    This method also generate the csv file used in the kaggle submission'''
    imgs = load_test_images()
    imgs_path = ['../test_set_images/test_'+str(i)+'/test_'+str(i)+'.png' for i in range(1, 51)]
    prediction_filenames = []
    for i in range(len(imgs)):
        prediction_filenames.append('predictions_groundtruth/prediction' + str(i+1) + '.png')

    for i in range(len(imgs)):
        print('Predicting test image number ' + str(i+1))
        if not cnn:
            Zi = model.predict(imgs_path[i])
        else:
            Zi = model.predict(load_image(imgs_path[i]))
        
        w = imgs[i].shape[0]
        h = imgs[i].shape[1]
        # Post processing
        Zi = Zi.reshape((int(h/model.patchSize), int(w/model.patchSize)))
        new_labels = post_processing(Zi).reshape(-1)
        # Generate groundtruth from label and save the image
        predicted_im = label_to_img(w, h, model.patchSize, model.patchSize, new_labels)
        color_mask = np.zeros((w, h, 3), dtype=np.float)
        color_mask[:,:,0] = predicted_im
        color_mask[:,:,1] = predicted_im
        color_mask[:,:,2] = predicted_im

        mpimg.imsave(prediction_filenames[i], color_mask)
    
    print('Generating csv file: ' + filename)
    masks_to_submission('predictions_csv/'+filename, *prediction_filenames)