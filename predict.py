from keras.models import model_from_json
import numpy
import os
import argparse
from PIL import Image
import numpy as np

from flask import Flask
app = Flask(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument("imagePath")
# args = parser.parse_args()
# img = Image.open(args.imagePath)

# load json and create model
json_file = open('/home/server3/keras/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/home/server3/keras/model.h5")
print("Loaded model from disk")

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#=================Make a prediction=====================================

def predict(imagePath):
    height = 32
    width = 32
    
    img = Image.open(imagePath)
    img = img.resize((height, width))
    data = np.asarray( img, dtype="int32" )
    data = np.asarray([data])
    prediction = model.predict(data)
    
    # for i in range(len(prediction[0])):
    #     print("Score: " + str(i+1) + " - Confidence: " + '{:.11%}'.format(prediction[0][i]))
    
    return prediction
    

@app.route("/predict")
def getPredictions():
    predictionArray =  predict("/home/server3/keras/ImageUpload/BackEnd/upload.jpg")
    returnString = ""
    for i in range(len(predictionArray[0])):
        returnString += ("(Score: " + str(i) + " - Confidence: " + str(int(predictionArray[0][i]*100)) + "%) ------ ")
    return returnString
    
predict("/home/server3/keras/ImageUpload/BackEnd/upload.jpg")
    
