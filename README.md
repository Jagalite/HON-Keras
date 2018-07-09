# Machine Learning - TensorFlow & Keras

## Making Models
- install python3
- install pip3
- pip3 install keras
- pip3 install tensorflow

## main.py 
If you are starting out in learning this stuff, a good place to start is main.py. It was written to be as easy to understand as possible as well as easily modifiable.You just need to unzip the training data in the dataset directory. Then run: python3 main.py

## Running a more sophisticated algorithm
augment_main.py is very similar to main.py. The augmented file will augment each photo randomly(rotate, stretch, distort) in order to deal with the low dataset size and get better results. 

## Predicting with your own image
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
