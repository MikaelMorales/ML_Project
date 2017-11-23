# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from image_helpers import *
from features import *
from classification import *

class ClassificationModel:
    
    def __init__(self):
        self.patchSize = 16
        self.foreground_threshold = 0.25

    def classificationMethod(self):
        self.method = SVC(C=1e5)
    
    def build_poly(self, X):
        """
        Fit the dataset using a polynomial basis.
        """
        poly = PolynomialFeatures(4, interaction_only=False)
        return poly.fit_transform(X)
    
    def train(self, Y, X):
        # Extract patches from input images
        patch_size = self.patchSize
        img_patches = [img_crop(X[i], patch_size, patch_size) for i in range(X.shape[0])]
        gt_patches = [img_crop(Y[i], patch_size, patch_size) for i in range(X.shape[0])]

        # Linearize list of patches
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
        
        # Compute features for each image patch
        X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
        Y = np.asarray([value_to_class(np.mean(gt_patches[i]), self.foreground_threshold) for i in range(len(gt_patches))])
           
        X = self.build_poly(X)
        self.method.fit(X, Y)
        
        print('Training completed')
        
    def classify(self, X):
        img_patches = [img_crop(X[i], self.patchSize, self.patchSize) for i in range(X.shape[0])]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
        X = self.build_poly(X)
        Z = self.method.predict(X)

        return Z.reshape(X.shape[0], -1)