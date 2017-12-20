import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
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
        '''Save the weights of the model in the folder cnn_weights
        to be able warm start'''
        self.model.save_weights('cnn_weights/'+file)

    def load_weights(self, file):
        '''Load the given weights of the model,
        the weights needs to be inside cnn_weights'''
        self.model.load_weights('cnn_weights/'+file)

    def setModel(self):
        ''' Initialize the CNN with the predefined structure'''
        # Window size of 80x80x3
        input_shape = (80, 80, 3)
        # Binary classification
        num_classes = 2

        # Model structure
        self.model.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    def train(self, X, Y):
        '''Train the CNN with the given features and label'''
        # Mirror boundaries: pad by patchsize*2
        X_new = np.asarray([np.lib.pad(X[i],((self.patchSize*2, self.patchSize*2), (self.patchSize*2, self.patchSize*2), (0,0)), 'reflect') for i in range(len(X))])
        # Generate windows of size 80x80x3 and linearize the results
        img_patches = [img_crop_with_padding(X_new[i], X[i], self.patchSize, self.patchSize) for i in range(X_new.shape[0])]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

        # Extract patches from groundtruth and convert to labels
        gt_patches = create_linearized_patches(Y, self.patchSize)
        Y = extract_features_from_gt_patches(gt_patches, self.foreground_threshold)
        Y = to_categorical(Y, num_classes=self.num_classes)

        # Avoid running the model if the accuracy doesn't improve more than 0.0001
        stop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')

        # Train the model using 20 percent of the dataset for validation and using batches of size 128.
        self.model.fit(img_patches, Y,
                       validation_split=0.2,
                       batch_size=128,
                       epochs=100,
                       callbacks=[stop])
                       
    def predict(self, img):
        # Mirror boundaries: pad by patchsize*2
        X = np.lib.pad(img, ((self.patchSize*2, self.patchSize*2), (self.patchSize*2, self.patchSize*2), (0,0)), 'reflect')
        # Generate windows of size 80x80x3 and linearize the results
        img_patches = [img_crop_with_padding(X, img, self.patchSize, self.patchSize)]
        img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

        Z = self.model.predict(img_patches)
        return np.argmax(Z, axis=1) # Invert function to_categorical from keras
