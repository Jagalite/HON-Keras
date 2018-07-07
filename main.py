from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from PIL import Image

import numpy as np

height = 300
weight = 300

batch_size = 32
num_classes = 100
epochs = 10

def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.resize((height, weight))
    data = np.asarray( img, dtype="int32" )
    data = data.flatten()
    return data

def loadFiles(gender):
    numpyImages = []
    labelsScores = []
    
    for i in range(1, 1001):
        
        imagePath = "dataset/" + gender + "/" + str(i).zfill(4) + ".jpg"
        labelPath = "dataset/" + gender + "/" + str(i).zfill(4) + ".txt"
        
        imageArray = load_image(imagePath)
        if(len(imageArray) == height*weight*3   ):
            numpyImages.append(imageArray) #image
            
            with open(labelPath, errors='ignore') as textFile:
                labelsScores.append([float(textFile.readline().strip())])
    
    
    return np.array(numpyImages), np.array(labelsScores)
    
np.random.seed(1337)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=height*weight*3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data, labels = loadFiles("female")

labels = to_categorical(labels, num_classes=num_classes)

data_train = data[0:900]
labels_train = labels[0:900]

data_test = data[900:]
labels_test = labels[900:]

model.fit(data_train, labels_train, epochs=epochs, batch_size=batch_size, validation_data=(data_test, labels_test))

score = model.evaluate(data_test, labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

