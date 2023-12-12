# FL-Client
The Federated Learning Client is an essential component of the federated learning system that runs on individual devices or nodes. This client facilitates the training of machine learning models locally using locally available data while preserving data privacy. The model updates from all participating clients are sent to the Federated Learning Server for aggregation and model improvement.
<br/> You can find the link to server <a href="https://github.com/bargavkoduri/BTP-Server">here</a>

# Set-Up
To run the client make sure you have node, python and tensorflow installed on your system.
<br/>
To install other dependencies type
<br/>
``` npm i ```

# Usage
1) Change the SERVER_URL in constants.js file to the URL where server is hosted.
2) To start the client run ```node client.js```.

<br/>
The dataset should be present the following : <br/>
 (i) The main folder is named "train," and it contains subfolders, each corresponding to a specific class label.
<br/>
 (ii) Each subfolder's name represents a distinct class in the dataset.

(iii) Inside each class subfolder, there are image files (or any other data format depending on the dataset) that belong to the corresponding class.
  ```
train/
|-- class_label_1/
|   |-- image_1.jpg
|   |-- image_2.jpg
|   |-- ...
|-- class_label_2/
|   |-- image_3.jpg
|   |-- image_4.jpg
|   |-- ...
|-- ...
```
