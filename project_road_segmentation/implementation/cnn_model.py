import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential

class CNN:
    def __init__(self, patchSize):
        self.patchSize = patchSize
        self.model = Sequential()
        self.setModel()
    
    def save_weights(self, file):
        self.model.save_weights(file)
        
    def setModel(self):
        input_shape = (self.patchSize, self.patchSize, 3)
        num_classes = 2
        
        self.model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
        
    def train(self, Y, X):
        # Extract patches from input images and linearized them
        img_patches = create_linearized_patches(X, self.patchSize)
        gt_patches = create_linearized_patches(Y, self.patchSize)
        
        # Compute features for each image patch
        X = extract_features_from_patches(img_patches)
        Y = extract_features_from_gt_patches(gt_patches, self.foreground_threshold)
        
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')

        self.model.fit(X, Y, 
                       validation_split=0.2,
                       batch_size=128,
                       epochs=10,
                       callbacks=[stop_callback])

    def predict(self, img):
        print('Classifying...')
        X = extract_img_features(img, self.patchSize)
        return self.model.predict(X)