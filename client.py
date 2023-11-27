import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
from helper import decode_weights,encode_weights,ReadandProcessData
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow_model_optimization as tfmot

class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, shuffle=True):
        self.data = data  # Your dataset (e.g., list of file paths)
        self.labels = labels  # Your corresponding labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        
        X_batch = []  # Use a list instead of a NumPy array
        y_batch = np.zeros((len(batch_indexes),), dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            # Load and preprocess your data here (e.g., read image, apply transformations)
            # Example:
            # img = load_and_preprocess_image(self.data[idx])
            # Determine image size dynamically
            img = self.data[idx]
            X_batch.append(img)
            y_batch[i] = self.labels[idx]  # Assign your labels to y_batch
        
        # Convert X_batch to a NumPy array
        X_batch = np.array(X_batch, dtype=np.float32)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# function for reading weights from text file
def read_from_file(fileName):
    with open(fileName,'r') as file:
        data = file.read()
    return data

# load model weights
model_weights = decode_weights(read_from_file("model_weights.txt"))

# load model from json file
with tfmot.quantization.keras.quantize_scope():
    model = model_from_json(read_from_file("model_config.json"))
    
# compile the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
# set weights to the model
model.set_weights(model_weights)


# Reading data from the dataset
X_train,y_train = ReadandProcessData("test")
train_gen = DataGenerator(X_train,y_train,8)
# Train the model on the data present on the client
model.fit(train_gen,epochs = 5,batch_size=2)


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