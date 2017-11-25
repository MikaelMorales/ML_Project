# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from image_helpers import *
from features import *

class SVM:
    
    def __init__(self, patchSize, foreground_threshold, C=1e6):
        self.patchSize = patchSize
        self.foreground_threshold = foreground_threshold
        self.model = GridSearchCV(SVC(), {'kernel':['rbf'], 'C':[C]}, cv=4, n_jobs=-1)
    
    def grid_search(self, C, gamma):
        self.model = GridSearchCV(SVC(), {'kernel':['rbf'], 'C':C, 'gamma':gamma}, cv=4, n_jobs=-1)

    def train(self, Y, X):
        print('Training...')
        # Extract patches from input images and linearized them
        img_patches = create_linearized_patches(X, self.patchSize)
        gt_patches = create_linearized_patches(Y, self.patchSize)
        
        # Compute features for each image patch
        X = extract_features_from_patches(img_patches)
        Y = extract_features_from_gt_patches(gt_patches, self.foreground_threshold)
        
        self.model.fit(X, Y)
        
        print(str(self.model.cv_results_['mean_test_score']))
        
    def predict(self, img):
        print('Classifying...')
        X = extract_img_features(img, self.patchSize)
        return self.model.predict(X)