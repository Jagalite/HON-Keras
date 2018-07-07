from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

import numpy as np

#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#Parameters for a run - tweak to improve accuracy - marked by in the code #tweak
height = 100
weight = 100

#Split full dataset to training and testing data, ie 0-900 for taining and 900-1000 for testing
splitIndex = 900

batch_size = 32

#defines how percise the perdictions need to be.
#ie: num_classes = 10 means outputs have only 10 categories 1,2,3,4....10
#ie: num_classes = 100 means outputs have 100 categories 1.0,1.1,1.2,1.3,...9.8,9.9, 10.0 or 1,2,3,4....100
num_classes = 10

epochs = 10

def create( infilename ) :
    img = Image.open( infilename )
    img = img.resize((height, weight))
    data = np.asarray( img, dtype="int32" )
    data = data.flatten()
    return data

# Step 1 - Generates image file paths and loads scores
def getFilePathsAndScores(gender):
    imagePaths = []
    labelsScores = []
    
    for i in range(1, 1001):
        imagePath = "dataset/" + gender + "/" + str(i).zfill(4) + ".jpg"
        labelPath = "dataset/" + gender + "/" + str(i).zfill(4) + ".txt"
        
        imagePaths.append(imagePath)
            
        #read first line of each file into this stucture: [[1.2],[5.5],[9.0]]
        with open(labelPath, errors='ignore') as textFile:
            labelsScores.append([float(textFile.readline().strip())])
    
    return imagePaths, labelsScores

# Step 2 - Karas/Tensorflow requires input to be NumPy Arrays
def loadImagesAndLabels(listOfImagePaths, listOfScoreLabels):
    listOfImageData = []
    for imagePath in listOfImagePaths:
        img = Image.open( imagePath )
        img = img.resize((height, weight)) #tweak
        data = np.asarray( img, dtype="int32" )
        listOfImageData.append(data)
        
    return listOfImageData, listOfScoreLabels
    
# Step 3 - remove data-lebels that are bad. ex discard some images that come back as a tuple of size 2 rather then 3.
def cleanUpData(listOfImageData, listOfScoreLabels):
    cleanData = []
    cleanLabels = []
    for i in range(len(listOfImageData)):
        if(len(listOfImageData[i].shape) == 3):
            cleanData.append(listOfImageData[i])
            cleanLabels.append(listOfScoreLabels[i])
    return cleanData, cleanLabels
    
# Step 4 - finalize data and labels
def finalizeInputs(listOfImageData, listOfScoreLabels):
    listOfImageData = np.array(listOfImageData)
    listOfScoreLabels = listOfScoreLabels
    
    finalData = []
    finalScores = []
    
    for i in range(len(listOfImageData)):
        finalData.append(listOfImageData[i].flatten())
        finalScores.append(np.array(listOfScoreLabels[i]))
        
    finalData = np.array(finalData)
    finalScores = np.array(finalScores)
    
    
    finalScores = to_categorical(finalScores, num_classes=num_classes) #tweak
    
    return finalData, finalScores
    

# Step 5 - Split data into training and testing, used to train and test models
def splitData(splitIndex, trainingData, trainingLabes):
    return trainingData[0:splitIndex], trainingLabes[0:splitIndex], trainingData[splitIndex:], trainingLabes[splitIndex:]

#Make each run consistent
np.random.seed(1337)

# Load, clean, finalize, and split data
data, labels = getFilePathsAndScores("female") #Step 1
data, labels = loadImagesAndLabels(data, labels) #Step 2
data, labels = cleanUpData(data, labels) #Step 3
data, labels = finalizeInputs(data, labels) #Step 4
trainingData, trainingLabels, testingData, testingLabels = splitData(splitIndex, data, labels) #tweak #Step 5

# Create the Model/network
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=height*weight*3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainingData, trainingLabels, epochs=epochs, batch_size=batch_size, validation_data=(testingData, testingLabels))

# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
    
# data_train = data_train.reshape(((1,) + data_train.shape))
# datagen.fit(data_train)

# fits the model on batches with real-time data augmentation:
#model.fit_generator(datagen.flow(data_train, labels_train, batch_size=32),steps_per_epoch=len(data_train), epochs=epochs)

# Evaluate Model Accuracy on Test data
score = model.evaluate(testingData, testingLabels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

