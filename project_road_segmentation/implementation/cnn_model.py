import keras
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from image_helpers import *
from features import *

class CNN:
    def __init__(self, patchSize, foreground_threshold):
        self.patchSize = patchSize
        self.foreground_threshold = foreground_threshold 
        self.num_classes = 2        
        self.model = Sequential()
        self.setModel()
    
    def save_weights(self, file):
        self.model.save_weights('cnn_weights/'+file)
        
    def load_weights(self, file):
        self.model.load_weights('cnn_weights/'+file)

    def setModel(self):
        #input_shape = (16, 16, 3)
        input_shape = (48, 48, 3)

        pool_size = (2,2)
        nb_classes = 2
        
        self.model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        
    def train(self, X, Y):
        X_new = np.asarray([np.lib.pad(X[i],((self.patchSize, self.patchSize), (self.patchSize, self.patchSize), (0,0)), 'reflect') for i in range(len(X))])
        img_patches = [img_crop_with_padding(X_new[i], X[i], self.patchSize, self.patchSize) for i in range(X_new.shape[0])]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        #img_patches = create_linearized_patches(X, self.patchSize)        

        gt_patches = create_linearized_patches(Y, self.patchSize)        
        Y = extract_features_from_gt_patches(gt_patches, self.foreground_threshold)
        Y = to_categorical(Y, num_classes=self.num_classes)
        
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=2, verbose=1, mode='auto')

        self.model.fit(img_patches, Y, 
                       validation_split=0.2,
                       batch_size=128,
                       epochs=100,
                       callbacks=[stop_callback])
    
    def predict(self, img):
        X = np.lib.pad(img, ((self.patchSize, self.patchSize), (self.patchSize, self.patchSize), (0,0)), 'reflect')
        img_patches = [img_crop_with_padding(X, img, self.patchSize, self.patchSize)]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
        
        #img_patches = [img_crop(img, self.patchSize, self.patchSize)]
        #img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

        Z = self.model.predict(img_patches)
        return np.argmax(Z, axis=1)
    