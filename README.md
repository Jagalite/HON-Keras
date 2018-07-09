# Machine Learning - TensorFlow & Keras

This exmaple was created to be as minimalistic as possible. main.py should be taken and modified if your project when you want to implement something. It was written to be as modifiable as possible and very easy to understand. Most tutorials used preloaded datasets that is hard to get you started because it doesn't show how you load images, clean them and prep them for the ML models. main.py goes through all this. From loading the images, training the models, saving the training and predicting using new images. This example should provide you with the bare minimum to start a ML project in tensorflow. This uses keras, tensorflow's/Google's offial high-level ML api. https://keras.io/#installation

## Making Models - Dependencies
- install python3
- install pip3
- pip3 install keras
- pip3 install tensorflow

## main.py
If you are starting out in learning this stuff, a good place to start is main.py. It was written to be as easy to understand as possible as well as easily modifiable.You just need to unzip the training data in the dataset directory. Then run: python3 main.py

## Running a more sophisticated algorithm
augment_main.py is very similar to main.py. The augmented file will augment each photo randomly(rotate, stretch, distort) in order to deal with the low dataset size in pursuits if a better result.

# Starting a service to upload your own photos
Please Switch off master and use branch(uploadImage) for the sections below.
We will start up a python flask serivce that will run the model against a picture in a specific directory. This image will be uploaded via a nodejs service called from an front end angular app. Change all the ip addresses (/backend/server.js and /frontend/src/app/app.component.html) and you should be set to go.

### Starting all the services
Python
- FLASK_APP=predict.py flask run --host=192.168.0.11

Frontend
- ng serve --host 192.168.0.11

Backend
- node server.js

### dependencies
- install nodejs
- pip3 install flask
