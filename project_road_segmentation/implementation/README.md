# Machine Learning Road Segmentation Project

### Mikael Morales Gonzales, Charles Parzy-Turlat, Vincent Petri

This is the repository for the road segmentation project we worked on during the Fall semester 2017.

## Libraries

This project was solved using sklearn first to train and predict using an Support Vector Machine(SVM) classifier.

We then implemented a Convolutionnal Neural Network(CNN). For this we used [Keras](https://keras.io/), a high level neural network API running over a [Tensorflow](https://www.tensorflow.org/) backend to do the computations.
To run the code it is necessary to install the following libraries:
* TensorFlow 1.4.0
* Keras 2.1.2

## Run the prediction

In order to run the prediction without training you should:
* Make sure the `test_set_images` folder is located at the same level as the `implementation` folder
* Go into the implementation folder and run the following command:
```
python run.py
```
The result file will be saved as `run_results.csv`.

## Model Training

If you want to train the model you can use the two notebooks:
* `SupportVectorMachine.ipynb` for the SVM model.
* `CNN.ipynb` for the CNN model.

## Presentation of the files

* `cnn_weights` contains the weights we used to get our best prediction on Kaggle
* `classification.py` defines the function we used to do our predictions
* `cnn_model.py` defines the CNN model methods to train and do predict
* `SVM_model.py`defines the SVM model methods to train and do prediction
* `features.py`defines the function to extract features for our SVM model
* `image_helpers.py` defines the function to interract with the images (eg. load the images as numpy arrays)
