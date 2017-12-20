# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from image_helpers import *
from features import *

class SVM:

    def __init__(self, patchSize, foreground_threshold, C=1e6, with_neighbors=False):
        self.patchSize = patchSize
        self.foreground_threshold = foreground_threshold
        self.model = GridSearchCV(SVC(), {'kernel':['rbf'], 'C':[C]}, cv=4, n_jobs=-1)
        self.with_neighbors = with_neighbors

    def train(self, Y, X):
        # Extract patches from input images and linearized them
        if self.with_neighbors:
            img_patches = create_linearized_patches_with_dimensions(X, self.patchSize)
            X = extract_features_img_patches_with_neighbors(img_patches)
        else:
            img_patches = create_linearized_patches(X, self.patchSize)
            X = extract_features_from_patches(img_patches)

        gt_patches = create_linearized_patches(Y, self.patchSize)

        # Compute features for each image patch
        Y = extract_features_from_gt_patches(gt_patches, self.foreground_threshold)

        self.model.fit(X, Y)

        print('Accuracy=' + str(np.mean(self.model.cv_results_['mean_test_score'])))

    def predict(self, img):
        if self.with_neighbors:
            X = extract_img_features_with_neighbors(img, self.patchSize)
        else:
            X = extract_img_features(img, self.patchSize)
        return self.model.predict(X)