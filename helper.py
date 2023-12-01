import codecs
import pickle
import numpy as np
import os
from PIL import Image

# function to read from a file
def read_from_file(fileName):
    with open(fileName, 'r') as file:
        data = file.read()
    return data

# function to write to a file
def write_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(data)

# function to decode the base64 encoded string
def decode_weights(b64_str):
    return pickle.loads(codecs.decode(b64_str.encode(), "base64"))

# function to encode the weights into a base64 string
def encode_weights(weights):
    return codecs.encode(pickle.dumps(weights), "base64").decode()

# function to read and process data from folder structure
def ReadandProcessData(path):
    root_dir = '{}'.format(path)
    class_labels = os.listdir(root_dir)
    images = []
    labels = []
    target_width = 224
    target_height = 224
    for label in class_labels:
        class_dir = os.path.join(root_dir, label)
        for file_name in os.listdir(class_dir):
            if 'Copy' in file_name:
                continue  # skip files with 'copy' in the filename
            image_path = os.path.join(class_dir, file_name)
            image = Image.open(image_path)
            image = image.resize((target_width, target_height))
            images.append(np.array(image))
            labels.append(label)
    X = np.array(images)
    y = np.array(labels)
    X = X / 255.0
    for i in range(len(y)):
        y[i] = class_labels.index(y[i])
    y = y.astype(np.int8)
    return X,y