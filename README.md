The code simulates a system that combines anomaly detection using Convolutional Neural Networks (CNN) with blockchain technology for validation. It works as follows:

The generate_patient_data function creates synthetic data for 1000 patients with 20 features, adding some anomalies randomly. This data is then preprocessed using the StandardScaler to normalize it and split into training and test sets with train_test_split.

A simple CNN model is defined in the create_cnn_model function. The model is designed to take the data, reshape it to fit the CNN structure, and detect anomalies by learning patterns in the data. The model has two convolutional layers, followed by max-pooling layers, and ends with a dense layer to output a binary classification (normal or anomalous).

Next, a Blockchain class simulates the proof of work mining process. It generates blocks by mining them with a defined difficulty and records transactions. The mine_block function uses a nonce to find a valid block hash. The add_transaction method adds data (in this case, the patient data) to the block.

The simulate_blockchain_anomaly_detection function simulates the anomaly detection process by iterating through the patient data in blocks of 10 samples. For each block, the CNN model predicts whether the data points are anomalous, and the blockchain validates the block by mining it. The result includes the block's information and the percentage of anomalies detected in that block.

Finally, visualize_results plots the percentage of anomalies detected in each block. The run_simulation function ties everything together by generating the data, preprocessing it, training the model, and then simulating the blockchain mining and anomaly detection process.

When you run this code, it will train the CNN on synthetic data, simulate blockchain block creation with anomaly detection, and then visualize the results. The flowchart of the framework with components as shown as below.

-----------------         ------------------------         -------------------
![image](https://github.com/user-attachments/assets/1c5882f6-61ad-41b3-a684-04df60e03ed0)



import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import hashlib
import random
import time

# Generate synthetic patient data (1000 samples, 20 features)
def generate_patient_data(num_samples=1000, num_features=20):
    # Randomly generate patient data with some anomalies
    data = np.random.randn(num_samples, num_features)
    labels = np.zeros(num_samples)
    
    # Introduce anomalies at random positions
    anomaly_indices = random.sample(range(num_samples), int(num_samples * 0.05))
    labels[anomaly_indices] = 1  # Anomalies are labeled as '1'
    
    return data, labels

# Preprocess the data (scaling and train-test split)
def preprocess_data(data, labels):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Define a simple CNN model for anomaly detection
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),  # Properly define input layer
        tf.keras.layers.Reshape((input_shape[0], 1, 1)),  # Reshaping for CNN (height, width, channels)
        tf.keras.layers.Conv2D(32, (1, 3), activation='relu', padding='same'),  # Use (1, 3) kernel
        tf.keras.layers.MaxPooling2D((1, 2), padding='same'),  # Pooling with (1, 2) size
        tf.keras.layers.Conv2D(64, (1, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((1, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Blockchain proof of work simulation
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = []
        self.difficulty = difficulty
        self.pending_transactions = []
        self.mining_reward = 10
    
    def mine_block(self, previous_hash):
        nonce = 0
        while True:
            block_data = str(self.pending_transactions) + str(previous_hash) + str(nonce)
            block_hash = hashlib.sha256(block_data.encode()).hexdigest()
            if block_hash[:self.difficulty] == '0' * self.difficulty:
                self.pending_transactions = []  # Reset transactions
                return {'previous_hash': previous_hash, 'transactions': self.pending_transactions, 'nonce': nonce, 'hash': block_hash}
            nonce += 1
    
    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

# Simulate anomaly detection in blockchain blocks
def simulate_blockchain_anomaly_detection(data, labels, model):
    blockchain = Blockchain(difficulty=4)
    previous_hash = '0' * 64  # Genesis block hash
    
    anomaly_detection_results = []
    
    for i in range(0, len(data), 10):  # Simulate 10 transactions per block
        block_data = data[i:i+10]
        block_labels = labels[i:i+10]
        
        # Use CNN to predict anomalies in the block
        predictions = model.predict(block_data)
        anomalies_in_block = np.sum(predictions) / len(predictions)  # Percentage of anomalies
        
        # Add block to blockchain with proof of work validation
        blockchain.add_transaction(block_data)
        mined_block = blockchain.mine_block(previous_hash)
        
        # Record the anomaly detection and the mining process
        anomaly_detection_results.append({'block': mined_block, 'anomaly_percentage': anomalies_in_block})
        
        previous_hash = mined_block['hash']
    
    return anomaly_detection_results

# Visualization function
def visualize_results(results):
    anomaly_percentages = [result['anomaly_percentage'] for result in results]
    
    # Plot the anomaly detection results across blocks
    plt.figure(figsize=(10, 6))
    plt.plot(anomaly_percentages, label="Anomaly Percentage per Block")
    plt.title("Anomaly Detection in Blockchain Blocks")
    plt.xlabel("Block Number")
    plt.ylabel("Anomaly Percentage")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main simulation pipeline
def run_simulation():
    # Step 1: Generate patient data
    data, labels = generate_patient_data()
    
    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)
    
    # Step 3: Create and train the CNN model
    model = create_cnn_model((X_train.shape[1], 1))  # Adjust input shape to (features, 1)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Step 4: Simulate blockchain mining with anomaly detection
    results = simulate_blockchain_anomaly_detection(X_test, y_test, model)
    
    # Step 5: Visualize the anomaly detection results
    visualize_results(results)

# Run the simulation
run_simulation()



Results

Epoch 1/10
/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/input_layer.py:26: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.
  warnings.warn(
25/25 ━━━━━━━━━━━━━━━━━━━━ 3s 25ms/step - accuracy: 0.9377 - loss: 0.4481 - val_accuracy: 0.9500 - val_loss: 0.2254
Epoch 2/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.9539 - loss: 0.2054 - val_accuracy: 0.9500 - val_loss: 0.2024
Epoch 3/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9426 - loss: 0.2172 - val_accuracy: 0.9500 - val_loss: 0.1976
Epoch 4/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9463 - loss: 0.2031 - val_accuracy: 0.9500 - val_loss: 0.1994
Epoch 5/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9472 - loss: 0.2060 - val_accuracy: 0.9500 - val_loss: 0.2014
Epoch 6/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9575 - loss: 0.1678 - val_accuracy: 0.9500 - val_loss: 0.2054
Epoch 7/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9481 - loss: 0.1980 - val_accuracy: 0.9500 - val_loss: 0.2064
Epoch 8/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9455 - loss: 0.1990 - val_accuracy: 0.9500 - val_loss: 0.2156
Epoch 9/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9604 - loss: 0.1568 - val_accuracy: 0.9500 - val_loss: 0.2192
Epoch 10/10
25/25 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9432 - loss: 0.1978 - val_accuracy: 0.9500 - val_loss: 0.2143
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step

![image](https://github.com/user-attachments/assets/35d56f7f-0e00-470b-b248-bc5fb692c3b0)

