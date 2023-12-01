import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
from helper import decode_weights,encode_weights,ReadandProcessData
import numpy as np

# function for reading weights from text file
def read_from_file(fileName):
    with open(fileName,'r') as file:
        data = file.read()
    return data

# load model weights
model_weights = decode_weights(read_from_file("model_weights.txt"))

model = model_from_json(read_from_file("model_config.json"))
    
# compile the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
# set weights to the model
model.set_weights(model_weights)

# Reading data from the dataset
X_train,y_train = ReadandProcessData("data")
# Train the model on the data present on the client
model.fit(X_train,y_train,epochs = 5)

# taking updated weights from the model
model_weights = {}
for layer in model.layers:
    if layer.trainable_weights:
        model_weights[layer.name] = encode_weights(layer.get_weights())

# setting number of data points client contains.
model_weights["number_of_train"] = X_train.shape[0]

# writing the updated weights dictionary to a text file 
with open('model_weights_updated.txt', 'w') as file:
    # Write the string to the file
    json.dump(model_weights,file)